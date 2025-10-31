# FILE: eval/eval_planner.py
# SOTA Evaluation Script for VisualPlannerDiffusion

import os
import sys
import time
import logging
from pathlib import Path
import random
import csv
from typing import Dict, List, Tuple, Any

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pandas as pd
import imageio
import cv2
from torchvision import transforms

# --- SOTA Metrics ---
try:
    import torchmetrics
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not found. `pip install torchmetrics[image]` for full metrics.")
    # Define dummy classes if not available
    class PeakSignalNoiseRatio: pass
    class StructuralSimilarityIndexMeasure: pass
    class LearnedPerceptualImagePatchSimilarity: pass

# --- Project Imports ---
# Add project root for imports if necessary
try:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Import components from our SOTA plan
    from models.planner import VisualPlannerDiffusion
    from utils.planner_dataset import HierarchicalPlannerDataset
    from train.train_planner import PlannerLightningModule # To load checkpoint
except ImportError as e:
    print(f"Error: Failed to import project modules. {e}")
    print("Ensure __init__.py files exist and PYTHONPATH is set, or run from project root.")
    sys.exit(1)

log = logging.getLogger(__name__)

# --- Helper Functions ---

def set_seed(seed: int):
    """Sets seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# FILE: eval/eval_planner.py

# --- START: ROBUST PATCH (Second Layer of Checkpoint Surgery) ---
# Replace the old load_planner_model function with this one.

def load_planner_model(cfg: DictConfig, device: torch.device) -> VisualPlannerDiffusion:
    """
    SOTA loading: Loads a PyTorch Lightning checkpoint, extracts the
    VisualPlannerDiffusion model, performs robust "checkpoint surgery" on key names,
    applies torch.compile, and sets to eval mode.
    """
    log.info(f"Loading Planner checkpoint from: {cfg.planner_ckpt_path}")
    checkpoint_path = Path(cfg.planner_ckpt_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    try:
        lightning_ckpt = torch.load(checkpoint_path, map_location="cpu")
        hparams = lightning_ckpt['hyper_parameters']
        model_cfg = hparams['model']
        scheduler_cfg = hparams['scheduler']
        
        planner = VisualPlannerDiffusion(
            image_size=model_cfg['image_size'],
            vit_model_name=model_cfg['vit_model_name'],
            vit_feature_dim=model_cfg['vit_feature_dim'],
            freeze_vit=model_cfg['freeze_vit'],
            progress_embed_dim=model_cfg['progress_embed_dim'],
            unet_block_out_channels=tuple(model_cfg['unet_block_out_channels']),
            unet_down_block_types=tuple(model_cfg['unet_down_block_types']),
            unet_up_block_types=tuple(model_cfg['unet_up_block_types']),
            unet_attention_head_dim=model_cfg['unet_attention_head_dim'],
            condition_drop_prob=model_cfg['condition_drop_prob'],
            num_diffusion_timesteps=scheduler_cfg['timesteps']
        )
        
        # --- Stage 1: Strip the 'model.' prefix from Lightning ---
        model_state_dict_cleaned = {
            k.replace("model.", ""): v
            for k, v in lightning_ckpt['state_dict'].items()
            if k.startswith("model.")
        }
        
        # --- Stage 2: Perform surgery on internal CLIP model prefixes ---
        # This fixes the mismatch between different 'transformers' versions or saving methods.
        final_state_dict = {}
        for k, v in model_state_dict_cleaned.items():
            new_key = k
            # Rule: If a key for the vision encoder is in the old format, rename it.
            # e.g., "vision_encoder.vision_embeddings.*" -> "vision_encoder.vision_model.embeddings.*"
            if k.startswith("vision_encoder.vision_"):
                new_key = k.replace("vision_encoder.vision_", "vision_encoder.vision_model.", 1)
                
            final_state_dict[new_key] = v

        # Load the final, surgically-repaired state dict.
        planner.load_state_dict(final_state_dict) # Use strict=True to confirm it's a perfect match
        log.info("Successfully performed checkpoint surgery and loaded VisualPlannerDiffusion state dict.")

    except Exception as e:
        log.exception(f"An unknown error occurred while loading the model: {e}")
        raise

    planner = planner.to(device).eval()

    # SOTA Optimization: Apply torch.compile
    if cfg.inference.use_torch_compile and hasattr(torch, "compile"):
        log.info(f"Applying torch.compile (mode='{cfg.inference.torch_compile_mode}')...")
        try:
            planner = torch.compile(planner, mode=cfg.inference.torch_compile_mode)
        except Exception as e:
            log.warning(f"torch.compile failed: {e}. Continuing without compilation.")
    
    return planner
# --- END: ROBUST PATCH ---



def get_validation_dataloader(cfg: DictConfig) -> Tuple[DataLoader, Any]:
    """Instantiates the dataset and returns the validation split dataloader."""
    log.info(f"Loading dataset from: {cfg.dataset_path}")
    full_dataset = HierarchicalPlannerDataset(
        dataset_path=cfg.dataset_path,
        subgoal_horizon_k=cfg.dataset.subgoal_horizon_k,
        image_size=tuple(cfg.data_preprocessing.image_size),
        img_mean=tuple(cfg.data_preprocessing.img_mean),
        img_std=tuple(cfg.data_preprocessing.img_std),
        use_random_aug=False # Never use augmentation for evaluation
    )

    # Split dataset (e.g., 95% train, 5% val) with a fixed seed
    total_len = len(full_dataset)
    val_len = int(total_len * cfg.evaluation.val_split)
    train_len = total_len - val_len
    log.info(f"Splitting dataset with seed {cfg.seed}: Train={train_len}, Val={val_len}")
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(cfg.seed) # Consistent split
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        num_workers=cfg.evaluation.num_workers,
        pin_memory=(cfg.inference.device == 'cuda'),
        persistent_workers=(cfg.evaluation.num_workers > 0),
        drop_last=False
    )
    # Return the dataloader AND the underlying validation Subset
    return val_dataloader, val_dataset

def denormalize_image(img_tensor: torch.Tensor, cfg: DictConfig) -> np.ndarray:
    """Converts a normalized CHW tensor back to HWC uint8 NumPy image."""
    # Ensure tensor is on CPU and detached
    img_tensor = img_tensor.detach().cpu()
    
    mean = torch.tensor(cfg.data_preprocessing.img_mean).view(3, 1, 1)
    std = torch.tensor(cfg.data_preprocessing.img_std).view(3, 1, 1)
    
    img_denorm = torch.clamp(img_tensor * std + mean, 0, 1) # CHW, float[0,1]
    img_np = img_denorm.permute(1, 2, 0).numpy() # HWC, float[0,1]
    img_np = (img_np * 255).astype(np.uint8) # HWC, uint8
    return img_np

def add_text_to_frame(frame: np.ndarray, text: str) -> np.ndarray:
    """Adds text to the bottom-left corner of an image frame."""
    try:
        return cv2.putText(
            img=np.ascontiguousarray(frame), # Ensure contiguous
            text=text,
            org=(10, frame.shape[0] - 10), # Bottom-left
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA
        )
    except Exception:
        log.warning("Failed to add text to frame. Is OpenCV installed?")
        return frame

# --- Core Evaluation Functions ---

def run_quantitative_evaluation(
    planner: VisualPlannerDiffusion,
    dataloader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
    output_dir: Path
):
    """Runs batched evaluation and computes quantitative metrics (LPIPS, SSIM, PSNR)."""
    if not TORCHMETRICS_AVAILABLE:
        log.error("torchmetrics is not installed. Skipping quantitative evaluation.")
        return

    log.info("Starting quantitative evaluation (LPIPS, SSIM, PSNR)...")
    
    # Initialize SOTA metrics from torchmetrics
    # data_range=1.0 because images are in [0, 1] range after denormalization
    # We will compute LPIPS on normalized [-1, 1] data (as it was trained)
    # We will compute SSIM/PSNR on denormalized [0, 1] data (standard)
    
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    
    # Store per-item results for CSV
    results_list = []
    
    mean_cfg = torch.tensor(cfg.data_preprocessing.img_mean, device=device).view(1, 3, 1, 1)
    std_cfg = torch.tensor(cfg.data_preprocessing.img_std, device=device).view(1, 3, 1, 1)

    # Denormalize helper for metrics
    def denorm_for_metrics(img_tensor):
        # Input is normalized tensor (B, C, H, W)
        return torch.clamp(img_tensor * std_cfg + mean_cfg, 0, 1)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Quantitative Eval"):
            try:
                # Move batch to device
                current_img = batch['current_image'].to(device)
                goal_img = batch['goal_image'].to(device)
                progress = batch['progress'].to(device)
                gt_subgoal_img = batch['gt_subgoal_image'].to(device) # Normalized
                
                # 1. Generate subgoal
                generated_subgoal = planner.sample(
                    current_image=current_img,
                    goal_image=goal_img,
                    progress=progress,
                    num_inference_steps=cfg.inference.planner_inference_steps
                ) # Normalized
                
                # --- Denormalize for SSIM/PSNR (range [0, 1]) ---
                generated_0_1 = denorm_for_metrics(generated_subgoal)
                gt_0_1 = denorm_for_metrics(gt_subgoal_img)

                # 2. Update metrics in batch
                # LPIPS expects normalized [-1, 1], but torchmetrics' version handles [0, 1] if normalize=True
                # Let's use the normalized [-1, 1] tensors directly for LPIPS, as it's more standard
                lpips_metric.update(generated_subgoal, gt_subgoal_img)
                ssim_metric.update(generated_0_1, gt_0_1)
                psnr_metric.update(generated_0_1, gt_0_1)
                
                # 3. Store per-item metrics (slower, but good for CSV)
                # This part is optional if you only want aggregate stats
                for i in range(current_img.shape[0]):
                    # Use .item() to get float values
                    lpips_val = lpips_metric(generated_subgoal[i:i+1], gt_subgoal_img[i:i+1]).item()
                    ssim_val = ssim_metric(generated_0_1[i:i+1], gt_0_1[i:i+1]).item()
                    psnr_val = psnr_metric(generated_0_1[i:i+1], gt_0_1[i:i+1]).item()
                    
                    results_list.append({
                        "progress": batch['progress'][i].item(),
                        "lpips": lpips_val,
                        "ssim": ssim_val,
                        "psnr": psnr_val
                    })

            except Exception as e:
                log.warning(f"Error during quantitative batch: {e}. Skipping batch.")
                continue

    # Compute final aggregate metrics
    final_lpips = lpips_metric.compute().item()
    final_ssim = ssim_metric.compute().item()
    final_psnr = psnr_metric.compute().item()
    
    log.info("--- Quantitative Evaluation Summary ---")
    log.info(f"Mean LPIPS: {final_lpips:.4f} (Lower is better)")
    log.info(f"Mean SSIM:  {final_ssim:.4f} (Higher is better)")
    log.info(f"Mean PSNR:  {final_psnr:.2f} dB (Higher is better)")
    log.info("---------------------------------------")

    # Save results to CSV
    csv_path = output_dir / "quantitative_metrics.csv"
    try:
        df = pd.DataFrame(results_list)
        df.to_csv(csv_path, index_label="sample_index")
        log.info(f"Saved detailed quantitative metrics to {csv_path}")
    except Exception as e:
        log.error(f"Failed to save metrics CSV: {e}")

def run_qualitative_rollouts(
    planner: VisualPlannerDiffusion,
    val_dataset: Subset,
    device: torch.device,
    cfg: DictConfig,
    output_dir: Path
):
    """Generates SOTA composite videos for N full episodes."""
    log.info(f"Starting qualitative video rollouts for {cfg.evaluation.num_video_rollouts} episodes...")
    
    # 1. Map episode indices to their corresponding dataset indices
    # This is a SOTA way to handle the unstructured Subset
    episode_map: Dict[int, List[int]] = {}
    for subset_idx in range(len(val_dataset)):
        # Get the original dataset index
        original_idx = val_dataset.indices[subset_idx]
        # Get the (ep_idx, t) from the full dataset
        ep_idx, timestep_t = val_dataset.dataset.valid_samples[original_idx]
        
        if ep_idx not in episode_map:
            episode_map[ep_idx] = []
        episode_map[ep_idx].append((timestep_t, subset_idx)) # Store (t, subset_idx)
        
    # Sort the timesteps for each episode
    for ep_idx in episode_map:
        episode_map[ep_idx].sort(key=lambda x: x[0]) # Sort by timestep_t
        
    # 2. Select N episodes to render
    available_ep_indices = list(episode_map.keys())
    if not available_ep_indices:
        log.warning("No valid episodes found in validation split. Skipping video rollouts.")
        return
        
    num_to_render = min(len(available_ep_indices), cfg.evaluation.num_video_rollouts)
    selected_ep_indices = random.sample(available_ep_indices, num_to_render)
    log.info(f"Selected episodes {selected_ep_indices} for video generation.")

    img_size = tuple(cfg.data_preprocessing.image_size)
    
    # 3. Loop over selected episodes and generate videos
    for ep_idx in selected_ep_indices:
        video_path = output_dir / f"vidhis_planner_rollout_ep_{ep_idx}.mp4"
        log.info(f"Generating video for episode {ep_idx} -> {video_path}")
        
        frames = []
        episode_samples = episode_map[ep_idx] # List of (t, subset_idx)
        
        # Get the goal image for this episode (it's the same for all steps)
        try:
            sample_data = val_dataset[episode_samples[0][1]] # Get data for first step
            goal_img_tensor = sample_data['goal_image'].to(device)
            goal_img_np = denormalize_image(goal_img_tensor, cfg)
        except Exception as e:
            log.error(f"Failed to load goal image for ep {ep_idx}: {e}. Skipping episode.")
            continue
            
        with imageio.get_writer(video_path, fps=cfg.evaluation.video_fps) as writer:
            for timestep_t, subset_idx in tqdm(episode_samples, desc=f"Video Ep {ep_idx}"):
                try:
                    # 1. Get data for this timestep
                    data = val_dataset[subset_idx]
                    current_img_tensor = data['current_image'].to(device).unsqueeze(0) # (1, C, H, W)
                    progress_tensor = data['progress'].to(device).unsqueeze(0) # (1,)
                    gt_subgoal_tensor = data['gt_subgoal_image'].to(device)
                    
                    # 2. Run Planner inference
                    with torch.no_grad():
                        generated_subgoal_tensor = planner.sample(
                            current_image=current_img_tensor,
                            goal_image=goal_img_tensor.unsqueeze(0), # (1, C, H, W)
                            progress=progress_tensor,
                            num_inference_steps=cfg.inference.planner_inference_steps
                        )
                    
                    # 3. Denormalize all images to HWC uint8
                    current_np = denormalize_image(current_img_tensor.squeeze(0), cfg)
                    generated_np = denormalize_image(generated_subgoal_tensor.squeeze(0), cfg)
                    gt_subgoal_np = denormalize_image(gt_subgoal_tensor, cfg)
                    
                    # 4. Create composite SOTA frame
                    # Ensure all images are the same size
                    if current_np.shape != tuple(img_size + (3,)):
                        current_np = cv2.resize(current_np, img_size, interpolation=cv2.INTER_AREA)
                    if generated_np.shape != tuple(img_size + (3,)):
                        generated_np = cv2.resize(generated_np, img_size, interpolation=cv2.INTER_AREA)
                    if gt_subgoal_np.shape != tuple(img_size + (3,)):
                        gt_subgoal_np = cv2.resize(gt_subgoal_np, img_size, interpolation=cv2.INTER_AREA)
                    if goal_img_np.shape != tuple(img_size + (3,)):
                        goal_img_np = cv2.resize(goal_img_np, img_size, interpolation=cv2.INTER_AREA)

                    # Add labels
                    current_np = add_text_to_frame(current_np, f"Current (t={timestep_t})")
                    generated_np = add_text_to_frame(generated_np, "Generated Subgoal")
                    gt_subgoal_np = add_text_to_frame(gt_subgoal_np, f"Ground Truth (t+{cfg.dataset.subgoal_horizon_k})")
                    goal_img_np = add_text_to_frame(goal_img_np, "Final Goal")
                    
                    composite_frame = np.hstack([current_np, generated_np, gt_subgoal_np, goal_img_np])
                    
                    # 5. Write frame to video
                    writer.append_data(composite_frame)
                
                except Exception as e:
                    log.warning(f"Error processing frame for ep {ep_idx}, t {timestep_t}: {e}")
                    continue
        log.info(f"Finished video for episode {ep_idx}.")

# --- Hydra Main Entry Point ---

@hydra.main(version_base=None, config_path="./configs", config_name="eval_planner_config")
def main(cfg: DictConfig):
    """Main function to run the planner evaluation."""
    output_dir = Path.cwd() # Hydra sets CWD to the output directory
    log.info("----------- ViDHiS Planner Evaluation -----------")
    log.info(f"Output Directory: {output_dir}")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("-------------------------------------------------")
    
    # --- Setup ---
    set_seed(cfg.seed)
    device = torch.device(cfg.inference.device)
    if device.type == 'cpu':
        log.warning("Running evaluation on CPU. LPIPS metric will be slow.")
        try:
            num_cores = os.cpu_count() or 1
            effective_cores = min(num_cores, cfg.inference.get("cpu_threads", num_cores))
            torch.set_num_threads(effective_cores)
            os.environ['OMP_NUM_THREADS'] = str(effective_cores)
            os.environ['MKL_NUM_THREADS'] = str(effective_cores)
            log.info(f"Set PyTorch/OMP/MKL CPU threads to: {effective_cores}")
        except Exception as e:
            log.warning(f"Failed to set CPU thread counts: {e}")

    # --- Load Data ---
    try:
        val_dataloader, val_dataset = get_validation_dataloader(cfg)
    except Exception as e:
        log.exception(f"Failed to load dataset: {e}. Aborting.")
        return

    # --- Load Model ---
    try:
        planner = load_planner_model(cfg, device)
    except Exception as e:
        log.exception(f"Failed to load planner model: {e}. Aborting.")
        return

    # --- Run Evaluations ---
    if cfg.evaluation.run_quantitative:
        run_quantitative_evaluation(planner, val_dataloader, device, cfg, output_dir)
    else:
        log.info("Skipping quantitative evaluation.")
        
    if cfg.evaluation.run_qualitative:
        run_qualitative_rollouts(planner, val_dataset, device, cfg, output_dir)
    else:
        log.info("Skipping qualitative video rollouts.")

    log.info("----------- Evaluation Finished -----------")

if __name__ == "__main__":
    if not TORCHMETRICS_AVAILABLE:
        print("\n\nCRITICAL WARNING: `torchmetrics` is not installed.")
        print("Please run `pip install torchmetrics[image] lpips` to enable quantitative evaluation.")
        print("LPIPS also requires `pip install lpips`.")
        print("Continuing without metrics, but results will be incomplete.\n\n")
    if not hasattr(cv2, 'putText'):
        print("\n\nCRITICAL WARNING: `opencv-python` is not installed.")
        print("Please run `pip install opencv-python-headless` to enable text on video frames.\n\n")

    main()