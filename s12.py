# FILE: run_planner_visualization.py
# A simple, standalone script to generate a video of the trained planner.

import torch
import numpy as np
from pathlib import Path
import imageio
import cv2
from tqdm import tqdm
import sys


# --- Add project root for imports ---
try:
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from models.planner import VisualPlannerDiffusion
    from utils.planner_dataset import HierarchicalPlannerDataset
except ImportError as e:
    print(f"Error importing project modules: {e}. Make sure this script is in your project's root directory.")
    sys.exit(1)

# ==============================================================================
# --- 1. CONFIGURATION: SET YOUR PATHS AND PARAMETERS HERE ---
# ==============================================================================

CHECKPOINT_PATH = "/content/drive/MyDrive/pda/Models/backup_epoch_7.ckpt"

# REQUIRED: Path to the dataset you want to visualize an episode from.
# Using your validation set is a good choice.
DATASET_PATH = "/content/drive/MyDrive/pda/data/validation/sota_dataset/expert_validation_run_99914b93.lmdb"

# Choose which episode you want to create a video for.
EPISODE_TO_VISUALIZE = 0  # 0 for the first episode, 1 for the second, etc.

# Set the output path for the video.
OUTPUT_VIDEO_PATH = f"./planner_visualization_ep_{EPISODE_TO_VISUALIZE}.mp4"

# Set the device to run on ('cuda' or 'cpu').
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_VIDEO_STEPS = 30 # Set to None to render the full episode
# Set the number of inference steps for the planner.
INFERENCE_STEPS = 30
# FILE: run_planner_visualization.py

# --- START: DEFINITIVE ROBUST PATCH ---
# Replace your existing load_planner_model function with this one.

def load_planner_model(ckpt_path: str, device: str) -> VisualPlannerDiffusion:
    """
    SOTA loading: Loads a PyTorch Lightning checkpoint, extracts the
    VisualPlannerDiffusion model, performs robust, multi-stage "checkpoint surgery"
    on key names, and sets it to eval mode.
    """
    print(f"Loading Planner checkpoint from: {ckpt_path}")
    checkpoint_path = Path(ckpt_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    try:
        lightning_ckpt = torch.load(checkpoint_path, map_location="cpu")
        hparams = lightning_ckpt['hyper_parameters']
        model_cfg = hparams['model']
        scheduler_cfg = hparams['scheduler']
        
        # 1. Instantiate the new model with the correct architecture from the checkpoint
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
        
        # 2. Get the raw state dict from the checkpoint
        raw_state_dict = lightning_ckpt['state_dict']
        
        # 3. Perform surgical renaming of the keys
        final_state_dict = {}
        for k, v in raw_state_dict.items():
            # Rule 1: Remove the top-level "model." prefix from PyTorch Lightning
            new_key = k.replace("model.", "", 1)
            
            # Rule 2: Remove the extra, internally nested ".model" from the vision encoder keys
            # Example: "vision_encoder.vision_model.model..." -> "vision_encoder.vision_model..."
            if "vision_encoder.vision_model.model" in new_key:
                new_key = new_key.replace("vision_encoder.vision_model.model.", "vision_encoder.vision_model.", 1)

            # Only add keys that are part of the VisualPlannerDiffusion model,
            # ignoring others like 'lpips_loss'.
            if new_key in planner.state_dict():
                final_state_dict[new_key] = v

        # 4. Load the perfectly renamed state dict. Use strict=False for robustness.
        incompatible_keys = planner.load_state_dict(final_state_dict, strict=False)
        
        # Provide a clear report on the loading process
        if incompatible_keys.missing_keys:
            print(f"WARNING: The following keys were in the model but not in the checkpoint: {incompatible_keys.missing_keys}")
        if incompatible_keys.unexpected_keys:
            print(f"WARNING: The following keys were in the checkpoint but not in the model (this is expected for things like lpips_loss): {incompatible_keys.unexpected_keys}")

        print("Successfully performed checkpoint surgery and loaded VisualPlannerDiffusion state dict.")

    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    return planner.to(device).eval()

# --- END: DEFINITIVE ROBUST PATCH ---

def denormalize_image(img_tensor: torch.Tensor, mean, std) -> np.ndarray:
    """Converts a normalized CHW tensor back to HWC uint8 NumPy image."""
    img_tensor = img_tensor.detach().cpu()
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    img_denorm = torch.clamp(img_tensor * std + mean, 0, 1)
    img_np = img_denorm.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    return img_np

def add_text_to_frame(frame: np.ndarray, text: str) -> np.ndarray:
    """Adds text to an image frame."""
    return cv2.putText(
        img=np.ascontiguousarray(frame), text=text, org=(10, frame.shape[0] - 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255),
        thickness=1, lineType=cv2.LINE_AA
    )

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # --- 1. Load Model ---
    planner = load_planner_model(CHECKPOINT_PATH, DEVICE)
    
    # --- 2. Load Dataset and Find Episode ---
    print(f"Loading dataset from: {DATASET_PATH}")
    # We need to get the mean/std from the dataset hparams in the checkpoint
    hparams = torch.load(CHECKPOINT_PATH, map_location="cpu")['hyper_parameters']
    dataset_cfg = hparams['dataset']
    
    full_dataset = HierarchicalPlannerDataset(
        dataset_path=DATASET_PATH,
        subgoal_horizon_k=dataset_cfg['subgoal_horizon_k'],
        image_size=tuple(dataset_cfg['image_size']),
        img_mean=tuple(dataset_cfg['img_mean']),
        img_std=tuple(dataset_cfg['img_std']),
    )

    print(f"Searching for samples from episode {EPISODE_TO_VISUALIZE}...")
    episode_samples = []
    for i in range(len(full_dataset)):
        ep_idx, timestep_t = full_dataset.valid_samples[i]
        if ep_idx == EPISODE_TO_VISUALIZE:
            episode_samples.append((timestep_t, i)) # Store (timestep, dataset_index)
    
    episode_samples.sort(key=lambda x: x[0]) # Sort by timestep
    if MAX_VIDEO_STEPS is not None and len(episode_samples) > MAX_VIDEO_STEPS:
        print(f"Limiting video to the first {MAX_VIDEO_STEPS} steps of the episode.")
        episode_samples = episode_samples[:MAX_VIDEO_STEPS]
    if not episode_samples:
        print(f"ERROR: No samples found for episode {EPISODE_TO_VISUALIZE}. "
              f"Please choose an episode index between 0 and {len(full_dataset.expert_reader.episode_metadata) - 1}.")
        sys.exit(1)

    print(f"Found {len(episode_samples)} steps for episode {EPISODE_TO_VISUALIZE}. Starting video generation...")

    # --- 3. Generate Video ---
    with imageio.get_writer(OUTPUT_VIDEO_PATH, fps=10) as writer:
        # Get the goal image for this episode (it's the same for all steps)
        first_sample_data = full_dataset[episode_samples[0][1]]
        goal_img_tensor = first_sample_data['goal_image'].to(DEVICE)
        goal_img_np = denormalize_image(goal_img_tensor, dataset_cfg['img_mean'], dataset_cfg['img_std'])
        goal_img_np_labeled = add_text_to_frame(goal_img_np.copy(), "Final Goal")

        for timestep_t, dataset_idx in tqdm(episode_samples, desc=f"Generating Video for Ep {EPISODE_TO_VISUALIZE}"):
            # Get data for this step
            data = full_dataset[dataset_idx]
            current_img_tensor = data['current_image'].to(DEVICE).unsqueeze(0)
            progress_tensor = data['progress'].to(DEVICE).unsqueeze(0)
            gt_subgoal_tensor = data['gt_subgoal_image']

            # Run Planner inference
            with torch.no_grad():
                generated_subgoal_tensor = planner.sample(
                    current_image=current_img_tensor,
                    goal_image=goal_img_tensor.unsqueeze(0),
                    progress=progress_tensor,
                    num_inference_steps=INFERENCE_STEPS
                )

            # Denormalize images for visualization
            mean, std = dataset_cfg['img_mean'], dataset_cfg['img_std']
            current_np = denormalize_image(current_img_tensor.squeeze(0), mean, std)
            generated_np = denormalize_image(generated_subgoal_tensor.squeeze(0), mean, std)
            gt_subgoal_np = denormalize_image(gt_subgoal_tensor, mean, std)
            
            # Add text labels to each panel
            current_np_labeled = add_text_to_frame(current_np, f"Current (t={timestep_t})")
            generated_np_labeled = add_text_to_frame(generated_np, "Generated Subgoal")
            gt_subgoal_np_labeled = add_text_to_frame(gt_subgoal_np, f"Ground Truth (t+{dataset_cfg['subgoal_horizon_k']})")

            # Create the composite frame
            composite_frame = np.hstack([current_np_labeled, generated_np_labeled, gt_subgoal_np_labeled, goal_img_np_labeled])
            
            # Write frame to video
            writer.append_data(composite_frame)

    print(f"\nSUCCESS! Video saved to: {OUTPUT_VIDEO_PATH}")