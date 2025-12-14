2025-12-13 13:47:08.810659: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1765633628.842920    5738 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1765633628.854016    5738 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1765633628.879551    5738 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1765633628.879586    5738 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1765633628.879594    5738 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1765633628.879601    5738 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2025-12-13 13:47:08.886766: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Seed set to 42
[2025-12-13 13:47:14,734][train_unified_planner][INFO] - === Starting UnifiedDiffusionPlanner Training ===
[2025-12-13 13:47:14,734][train_unified_planner][INFO] - Working Dir: /content/dgpo
[2025-12-13 13:47:14,739][train_unified_planner][INFO] - WandB logger enabled (offline mode)
[2025-12-13 13:47:14,742][models.unified_diffusion_planner][INFO] - Loading SigLIP backbone: google/siglip-base-patch16-224
[2025-12-13 13:47:15,850][models.unified_diffusion_planner][INFO] - SigLIP: Bottom layers FROZEN, top 3 layers UNFROZEN for geometric adaptation
[2025-12-13 13:47:15,850][models.unified_diffusion_planner][INFO] - SigLIP: Gradient checkpointing ENABLED for memory efficiency
[2025-12-13 13:47:15,850][models.unified_diffusion_planner][INFO] - SigLIP: 224x224, 196 patches
[2025-12-13 13:47:16,230][models.unified_diffusion_planner][INFO] -   - Phase head: 5 phases (auxiliary loss)
[2025-12-13 13:47:16,232][models.unified_diffusion_planner][INFO] - UnifiedDiffusionPlanner initialized with 88,081,165 trainable / 159,700,237 total parameters
[2025-12-13 13:47:16,233][models.unified_diffusion_planner][INFO] -   - Vision encoder: 51,432,192 trainable
[2025-12-13 13:47:16,233][models.unified_diffusion_planner][INFO] -   - Action head: 36,382,728 params
[2025-12-13 13:47:16,233][models.unified_diffusion_planner][INFO] -   - Separate heads: True
Using 16bit Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
`Trainer(val_check_interval=1.0)` was configured so validation will run at the end of the training epoch..
[2025-12-13 13:47:17,717][train_unified_planner][INFO] - Resuming from checkpoint: /content/drive/MyDrive/pda/models/unified/unified_planner_backup_epoch_002.ckpt
[2025-12-13 13:47:17,717][train_unified_planner][INFO] - Starting training...
wandb: WARNING The anonymous setting has no effect and will be removed in a future version.
wandb: WARNING `resume` will be ignored since W&B syncing is set to `offline`. Starting a new run with run id 9ovj66td.
wandb: Tracking run with wandb version 0.23.1
wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
wandb: Run data is saved locally in /content/drive/MyDrive/pda/logs/wandb/offline-run-20251213_134719-9ovj66td
[2025-12-13 13:47:22,750][train_unified_planner][INFO] - Loading Training Dataset from: /content/pda_data/train/training_set.lmdb
[2025-12-13 13:47:22,751][utils.unified_planner_dataset][INFO] - Initializing UnifiedPlannerDataset with K=8
[2025-12-13 13:47:22,846][OpenGL.acceleratesupport][INFO] - No OpenGL_accelerate module loaded: No module named 'OpenGL_accelerate'
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information./usr/local/lib/python3.12/dist-packages/google/protobuf/internal/well_known_types.py:178: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  self.FromDatetime(datetime.datetime.utcnow())

[2025-12-13 13:47:23,416][utils.expert_dataset][INFO] - Loading index from /content/pda_data/train/training_set_index.json...
/usr/local/lib/python3.12/dist-packages/google/protobuf/internal/well_known_types.py:178: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  self.FromDatetime(datetime.datetime.utcnow())
[2025-12-13 13:47:23,433][utils.expert_dataset][INFO] - Loaded 400 episodes, 82486 total valid chunks (Virtually Indexed).
[2025-12-13 13:47:23,448][utils.unified_planner_dataset][INFO] - Found 81686 valid samples from 400 episodes
[2025-12-13 13:47:23,449][train_unified_planner][INFO] -   Train samples: 81686
[2025-12-13 13:47:23,449][train_unified_planner][INFO] - Loading Validation Dataset from: /content/pda_data/val/validation_set.lmdb
[2025-12-13 13:47:23,450][utils.unified_planner_dataset][INFO] - Initializing UnifiedPlannerDataset with K=8
[2025-12-13 13:47:23,451][utils.expert_dataset][INFO] - Loading index from /content/pda_data/val/validation_set_index.json...
[2025-12-13 13:47:23,453][utils.expert_dataset][INFO] - Loaded 40 episodes, 8011 total valid chunks (Virtually Indexed).
[2025-12-13 13:47:23,455][utils.unified_planner_dataset][INFO] - Found 7931 valid samples from 40 episodes
[2025-12-13 13:47:23,456][train_unified_planner][INFO] -   Val samples: 7931
Restoring states from the checkpoint path at /content/drive/MyDrive/pda/models/unified/unified_planner_backup_epoch_002.ckpt
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/callbacks/model_checkpoint.py:566: The dirpath has changed from '/content/drive/MyDrive/pda/logs/checkpoints' to '/content/dgpo/checkpoint/checkpoints', therefore `best_model_score`, `kth_best_model_path`, `kth_value`, `last_model_path` and `best_k_models` won't be reloaded. Only `best_model_path` will be reloaded.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
[2025-12-13 13:47:46,415][train_unified_planner][INFO] - Optimizer Parameter Groups:
[2025-12-13 13:47:46,416][train_unified_planner][INFO] -   backbone: 50 params, lr=1.00e-05
[2025-12-13 13:47:46,416][train_unified_planner][INFO] -   head_decay: 80 params, lr=1.00e-04
[2025-12-13 13:47:46,416][train_unified_planner][INFO] -   head_no_decay: 96 params, lr=1.00e-04
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/model_summary/model_summary.py:242: Precision 16-mixed is not supported by the model summary.  Estimated model size in MB will not be accurate. Using 32 bits instead.
┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name  ┃ Type                    ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model │ UnifiedDiffusionPlanner │  159 M │ train │     0 │
└───┴───────┴─────────────────────────┴────────┴───────┴───────┘
Trainable params: 88.1 M                                                        
Non-trainable params: 71.6 M                                                    
Total params: 159 M                                                             
Total estimated model params size (MB): 638                                     
Modules in train mode: 190                                                      
Modules in eval mode: 160                                                       
Total FLOPs: 0                                                                  
Restored all states from the checkpoint at /content/drive/MyDrive/pda/models/unified/unified_planner_backup_epoch_002.ckpt
Sanity Checking: |          | 0/? [00:00<?, ?it/s][2025-12-13 13:47:46,725][lmdb_utils][INFO] -  Opened LMDB at /content/pda_data/val/validation_set.lmdb | mode=RO | file-mode
/content/dgpo/utils/unified_planner_dataset.py:332: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  gt_phase_t = int(gt_phases[t])
/usr/local/lib/python3.12/dist-packages/torch/utils/data/_utils/pin_memory.py:57: DeprecationWarning: The argument 'device' of Tensor.pin_memory() is deprecated. Please do not pass this argument. (Triggered internally at /pytorch/aten/src/ATen/native/Memory.cpp:46.)
  return data.pin_memory(device)
/usr/local/lib/python3.12/dist-packages/torch/utils/data/_utils/pin_memory.py:57: DeprecationWarning: The argument 'device' of Tensor.is_pinned() is deprecated. Please do not pass this argument. (Triggered internally at /pytorch/aten/src/ATen/native/Memory.cpp:31.)
  return data.pin_memory(device)
/usr/local/lib/python3.12/dist-packages/torch/utils/data/sampler.py:74: UserWarning: `data_source` argument is not used and will be removed in 2.2.0.You may still have custom implementation that utilizes it.
  warnings.warn(
[2025-12-13 13:47:56,907][utils.samplers][INFO] - Initializing EpisodeAwareSampler (DDP: False, Rank: 0)
/usr/lib/python3.12/multiprocessing/popen_fork.py:66: DeprecationWarning: This process (pid=5738) is multi-threaded, use of fork() may lead to deadlocks in the child.
  self.pid = os.fork()
[2025-12-13 13:47:57,014][lmdb_utils][INFO] -  Opened LMDB at /content/pda_data/train/training_set.lmdb | mode=RO | file-mode
[2025-12-13 13:47:57,014][lmdb_utils][INFO] -  Opened LMDB at /content/pda_data/train/training_set.lmdb | mode=RO | file-mode
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/fit_loop.py:534: Found 160 module(s) in eval mode at the start of training. This may lead to unexpected behavior during training. If this is intentional, you can ignore this warning.
Training: |          | 0/? [00:00<?, ?it/s][2025-12-13 13:47:57,020][train_unified_planner][INFO] - Fitting action normalizer on training dataset...
[2025-12-13 13:47:57,023][lmdb_utils][INFO] -  Opened LMDB at /content/pda_data/train/training_set.lmdb | mode=RO | file-mode
[2025-12-13 13:48:05,379][models.unified_diffusion_planner][INFO] - ActionNormalizer fitted: min=[-0.026746049523353577, -0.07258637994527817, -0.033010274171829224, -0.007153845392167568, -0.011443411000072956, -0.026500077918171883, 0.9995606541633606, -0.009999999776482582], max=[0.05671416223049164, 0.04525541514158249, 0.03450450301170349, 0.006272112485021353, 0.006951103452593088, 0.019817998632788658, 1.000100016593933, 1.0099999904632568]
[2025-12-13 13:48:05,380][train_unified_planner][INFO] -   Fitted on 100 samples
[2025-12-13 13:48:05,380][train_unified_planner][INFO] -   Action min: [-0.026746049523353577, -0.07258637994527817, -0.033010274171829224]
[2025-12-13 13:48:05,380][train_unified_planner][INFO] -   Action max: [0.05671416223049164, 0.04525541514158249, 0.03450450301170349]
Epoch 2: 100% 850/850 [28:40<00:00,  2.02s/it, v_num=66td, train/loss_step=0.0452, diff_step=0.0317, pose_step=0.0299, grip_step=0.0443, phase_step=0.136] [2025-12-13 14:16:45,678][train_unified_planner][INFO] - End of epoch 2: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-13 14:17:21,792][train_unified_planner][INFO] - Failsafe backup saved to /content/drive/MyDrive/pda/models/backups/unified/unified_planner_backup_epoch_002.ckpt
Epoch 2: 100% 850/850 [29:16<00:00,  2.07s/it, v_num=66td, train/loss_step=0.0452, diff_step=0.0317, pose_step=0.0299, grip_step=0.0443, phase_step=0.136, train/loss_epoch=0.0658, diff_epoch=0.0529, pose_epoch=0.0521, grip_epoch=0.0581, phase_epoch=0.130]
======================================================================
📊 EPOCH 2 COMPLETE
======================================================================
  Train Loss:     0.065817
    - Diffusion:  0.052861
    - Pose:       0.052107
    - Grip:       0.058140
    - Phase:      0.129560
  Learning Rate:  0.00e+00
======================================================================

Epoch 3: 100% 850/850 [28:48<00:00,  2.03s/it, v_num=66td, train/loss_step=0.0643, diff_step=0.0577, pose_step=0.0543, grip_step=0.0814, phase_step=0.066, train/loss_epoch=0.0658, diff_epoch=0.0529, pose_epoch=0.0521, grip_epoch=0.0581, phase_epoch=0.130] [2025-12-13 14:46:10,415][train_unified_planner][INFO] - End of epoch 3: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-13 14:46:56,917][train_unified_planner][INFO] - Failsafe backup saved to /content/drive/MyDrive/pda/models/backups/unified/unified_planner_backup_epoch_003.ckpt

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/83 [00:00<?, ?it/s]
Validation DataLoader 0:  60% 50/83 [03:25<02:15,  4.10s/it]
Validation DataLoader 0: 100% 83/83 [05:38<00:00,  4.08s/it]
Epoch 3: 100% 850/850 [35:15<00:00,  2.49s/it, v_num=66td, train/loss_step=0.0643, diff_step=0.0577, pose_step=0.0543, grip_step=0.0814, phase_step=0.066, train/loss_epoch=0.0658, diff_epoch=0.0529, pose_epoch=0.0521, grip_epoch=0.0581, phase_epoch=0.130, val/total_error=0.00615, val/diff_loss=0.0372]Epoch 3, global step 4676: 'val/total_error' reached 0.00615 (best 0.00615), saving model to '/content/dgpo/checkpoint/checkpoints/udp-epoch=03-val/total_error=0.0062.ckpt' as top 3
Epoch 3: 100% 850/850 [36:46<00:00,  2.60s/it, v_num=66td, train/loss_step=0.0643, diff_step=0.0577, pose_step=0.0543, grip_step=0.0814, phase_step=0.066, train/loss_epoch=0.0617, diff_epoch=0.050, pose_epoch=0.0493, grip_epoch=0.0544, phase_epoch=0.117, val/total_error=0.00615, val/diff_loss=0.0372] 
======================================================================
📊 EPOCH 3 COMPLETE
======================================================================
  Train Loss:     0.061661
    - Diffusion:  0.049958
    - Pose:       0.049328
    - Grip:       0.054366
    - Phase:      0.117029
  Learning Rate:  0.00e+00
  Val Error:      0.006150
    - Pos Error:  0.003861
    - Rot Error:  0.001608
    - Grip Error: 0.090740
======================================================================

Epoch 4: 100% 850/850 [28:45<00:00,  2.03s/it, v_num=66td, train/loss_step=0.0852, diff_step=0.0684, pose_step=0.0701, grip_step=0.0565, phase_step=0.168, train/loss_epoch=0.0617, diff_epoch=0.050, pose_epoch=0.0493, grip_epoch=0.0544, phase_epoch=0.117, val/total_error=0.00615, val/diff_loss=0.0372][2025-12-13 15:22:54,239][train_unified_planner][INFO] - End of epoch 4: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-13 15:24:39,760][train_unified_planner][INFO] - Failsafe backup saved to /content/drive/MyDrive/pda/models/backups/unified/unified_planner_backup_epoch_004.ckpt
Epoch 4: 100% 850/850 [30:31<00:00,  2.15s/it, v_num=66td, train/loss_step=0.0852, diff_step=0.0684, pose_step=0.0701, grip_step=0.0565, phase_step=0.168, train/loss_epoch=0.0556, diff_epoch=0.0464, pose_epoch=0.046, grip_epoch=0.0492, phase_epoch=0.0917, val/total_error=0.00615, val/diff_loss=0.0372]
======================================================================
📊 EPOCH 4 COMPLETE
======================================================================
  Train Loss:     0.055617
    - Diffusion:  0.046442
    - Pose:       0.046045
    - Grip:       0.049220
    - Phase:      0.091747
  Learning Rate:  0.00e+00
  Val Error:      0.006150
    - Pos Error:  0.003861
    - Rot Error:  0.001608
    - Grip Error: 0.090740
======================================================================

Epoch 5: 100% 850/850 [28:45<00:00,  2.03s/it, v_num=66td, train/loss_step=0.0365, diff_step=0.0349, pose_step=0.0373, grip_step=0.0177, phase_step=0.0163, train/loss_epoch=0.0556, diff_epoch=0.0464, pose_epoch=0.046, grip_epoch=0.0492, phase_epoch=0.0917, val/total_error=0.00615, val/diff_loss=0.0372][2025-12-13 15:53:25,260][train_unified_planner][INFO] - End of epoch 5: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-13 15:54:59,957][train_unified_planner][INFO] - Failsafe backup saved to /content/drive/MyDrive/pda/models/backups/unified/unified_planner_backup_epoch_005.ckpt
[2025-12-13 15:54:59,961][train_unified_planner][INFO] - Cleaned up old backup: unified_planner_backup_epoch_002.ckpt

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/83 [00:00<?, ?it/s]
Validation DataLoader 0:  60% 50/83 [03:23<02:14,  4.07s/it]
Validation DataLoader 0: 100% 83/83 [05:35<00:00,  4.05s/it]
Epoch 5: 100% 850/850 [35:58<00:00,  2.54s/it, v_num=66td, train/loss_step=0.0365, diff_step=0.0349, pose_step=0.0373, grip_step=0.0177, phase_step=0.0163, train/loss_epoch=0.0556, diff_epoch=0.0464, pose_epoch=0.046, grip_epoch=0.0492, phase_epoch=0.0917, val/total_error=0.00729, val/diff_loss=0.0339]Epoch 5, global step 6376: 'val/total_error' reached 0.00729 (best 0.00615), saving model to '/content/dgpo/checkpoint/checkpoints/udp-epoch=05-val/total_error=0.0073.ckpt' as top 3
Epoch 5: 100% 850/850 [37:53<00:00,  2.67s/it, v_num=66td, train/loss_step=0.0365, diff_step=0.0349, pose_step=0.0373, grip_step=0.0177, phase_step=0.0163, train/loss_epoch=0.0524, diff_epoch=0.044, pose_epoch=0.0438, grip_epoch=0.0448, phase_epoch=0.0847, val/total_error=0.00729, val/diff_loss=0.0339]
======================================================================
📊 EPOCH 5 COMPLETE
======================================================================
  Train Loss:     0.052423
    - Diffusion:  0.043957
    - Pose:       0.043831
    - Grip:       0.044845
    - Phase:      0.084661
  Learning Rate:  0.00e+00
  Val Error:      0.007295
    - Pos Error:  0.003887
    - Rot Error:  0.001619
    - Grip Error: 0.099023
======================================================================

Epoch 6: 100% 850/850 [28:45<00:00,  2.03s/it, v_num=66td, train/loss_step=0.0515, diff_step=0.0502, pose_step=0.0502, grip_step=0.0501, phase_step=0.0137, train/loss_epoch=0.0524, diff_epoch=0.044, pose_epoch=0.0438, grip_epoch=0.0448, phase_epoch=0.0847, val/total_error=0.00729, val/diff_loss=0.0339][2025-12-13 16:31:19,270][train_unified_planner][INFO] - End of epoch 6: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-13 16:32:59,699][train_unified_planner][INFO] - Failsafe backup saved to /content/drive/MyDrive/pda/models/backups/unified/unified_planner_backup_epoch_006.ckpt
Epoch 6: 100% 850/850 [30:26<00:00,  2.15s/it, v_num=66td, train/loss_step=0.0515, diff_step=0.0502, pose_step=0.0502, grip_step=0.0501, phase_step=0.0137, train/loss_epoch=0.0493, diff_epoch=0.0419, pose_epoch=0.0422, grip_epoch=0.0401, phase_epoch=0.0737, val/total_error=0.00729, val/diff_loss=0.0339]
======================================================================
📊 EPOCH 6 COMPLETE
======================================================================
  Train Loss:     0.049292
    - Diffusion:  0.041924
    - Pose:       0.042180
    - Grip:       0.040132
    - Phase:      0.073678
  Learning Rate:  0.00e+00
  Val Error:      0.007295
    - Pos Error:  0.003887
    - Rot Error:  0.001619
    - Grip Error: 0.099023
======================================================================

Epoch 7: 100% 850/850 [28:46<00:00,  2.03s/it, v_num=66td, train/loss_step=0.0376, diff_step=0.0368, pose_step=0.0389, grip_step=0.0216, phase_step=0.00876, train/loss_epoch=0.0493, diff_epoch=0.0419, pose_epoch=0.0422, grip_epoch=0.0401, phase_epoch=0.0737, val/total_error=0.00729, val/diff_loss=0.0339][2025-12-13 17:01:46,206][train_unified_planner][INFO] - End of epoch 7: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-13 17:02:43,134][train_unified_planner][INFO] - Failsafe backup saved to /content/drive/MyDrive/pda/models/backups/unified/unified_planner_backup_epoch_007.ckpt

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/83 [00:00<?, ?it/s]
Validation DataLoader 0:  60% 50/83 [03:24<02:14,  4.08s/it]
Validation DataLoader 0: 100% 83/83 [05:36<00:00,  4.06s/it]
Epoch 7: 100% 850/850 [35:22<00:00,  2.50s/it, v_num=66td, train/loss_step=0.0376, diff_step=0.0368, pose_step=0.0389, grip_step=0.0216, phase_step=0.00876, train/loss_epoch=0.0493, diff_epoch=0.0419, pose_epoch=0.0422, grip_epoch=0.0401, phase_epoch=0.0737, val/total_error=0.00636, val/diff_loss=0.0317]Epoch 7, global step 8076: 'val/total_error' reached 0.00636 (best 0.00615), saving model to '/content/dgpo/checkpoint/checkpoints/udp-epoch=07-val/total_error=0.0064.ckpt' as top 3
Epoch 7: 100% 850/850 [37:09<00:00,  2.62s/it, v_num=66td, train/loss_step=0.0376, diff_step=0.0368, pose_step=0.0389, grip_step=0.0216, phase_step=0.00876, train/loss_epoch=0.0466, diff_epoch=0.0406, pose_epoch=0.0409, grip_epoch=0.039, phase_epoch=0.0595, val/total_error=0.00636, val/diff_loss=0.0317] 
======================================================================
📊 EPOCH 7 COMPLETE
======================================================================
  Train Loss:     0.046574
    - Diffusion:  0.040626
    - Pose:       0.040863
    - Grip:       0.038971
    - Phase:      0.059471
  Learning Rate:  0.00e+00
  Val Error:      0.006364
    - Pos Error:  0.003213
    - Rot Error:  0.001517
    - Grip Error: 0.089213
======================================================================

Epoch 8: 100% 850/850 [28:50<00:00,  2.04s/it, v_num=66td, train/loss_step=0.0424, diff_step=0.0408, pose_step=0.0405, grip_step=0.0428, phase_step=0.0161, train/loss_epoch=0.0466, diff_epoch=0.0406, pose_epoch=0.0409, grip_epoch=0.039, phase_epoch=0.0595, val/total_error=0.00636, val/diff_loss=0.0317][2025-12-13 17:38:59,468][train_unified_planner][INFO] - End of epoch 8: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-13 17:40:24,255][train_unified_planner][INFO] - Failsafe backup saved to /content/drive/MyDrive/pda/models/backups/unified/unified_planner_backup_epoch_008.ckpt
[2025-12-13 17:40:24,261][train_unified_planner][INFO] - Cleaned up old backup: unified_planner_backup_epoch_005.ckpt
Epoch 8: 100% 850/850 [30:14<00:00,  2.14s/it, v_num=66td, train/loss_step=0.0424, diff_step=0.0408, pose_step=0.0405, grip_step=0.0428, phase_step=0.0161, train/loss_epoch=0.0446, diff_epoch=0.0393, pose_epoch=0.0397, grip_epoch=0.0364, phase_epoch=0.0531, val/total_error=0.00636, val/diff_loss=0.0317]
======================================================================
📊 EPOCH 8 COMPLETE
======================================================================
  Train Loss:     0.044634
    - Diffusion:  0.039325
    - Pose:       0.039744
    - Grip:       0.036391
    - Phase:      0.053086
  Learning Rate:  0.00e+00
  Val Error:      0.006364
    - Pos Error:  0.003213
    - Rot Error:  0.001517
    - Grip Error: 0.089213
======================================================================

Epoch 9: 100% 850/850 [28:46<00:00,  2.03s/it, v_num=66td, train/loss_step=0.0308, diff_step=0.0304, pose_step=0.0328, grip_step=0.0142, phase_step=0.00366, train/loss_epoch=0.0446, diff_epoch=0.0393, pose_epoch=0.0397, grip_epoch=0.0364, phase_epoch=0.0531, val/total_error=0.00636, val/diff_loss=0.0317][2025-12-13 18:09:10,472][train_unified_planner][INFO] - End of epoch 9: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-13 18:10:55,337][train_unified_planner][INFO] - Failsafe backup saved to /content/drive/MyDrive/pda/models/backups/unified/unified_planner_backup_epoch_009.ckpt
[2025-12-13 18:10:55,341][train_unified_planner][INFO] - Cleaned up old backup: unified_planner_backup_epoch_006.ckpt

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/83 [00:00<?, ?it/s]
Validation DataLoader 0:  60% 50/83 [03:23<02:14,  4.07s/it]
Validation DataLoader 0: 100% 83/83 [05:36<00:00,  4.05s/it]
Epoch 9: 100% 850/850 [36:10<00:00,  2.55s/it, v_num=66td, train/loss_step=0.0308, diff_step=0.0304, pose_step=0.0328, grip_step=0.0142, phase_step=0.00366, train/loss_epoch=0.0446, diff_epoch=0.0393, pose_epoch=0.0397, grip_epoch=0.0364, phase_epoch=0.0531, val/total_error=0.00631, val/diff_loss=0.0306]Epoch 9, global step 9776: 'val/total_error' reached 0.00631 (best 0.00615), saving model to '/content/dgpo/checkpoint/checkpoints/udp-epoch=09-val/total_error=0.0063.ckpt' as top 3
Epoch 9: 100% 850/850 [37:59<00:00,  2.68s/it, v_num=66td, train/loss_step=0.0308, diff_step=0.0304, pose_step=0.0328, grip_step=0.0142, phase_step=0.00366, train/loss_epoch=0.0435, diff_epoch=0.0383, pose_epoch=0.0388, grip_epoch=0.0344, phase_epoch=0.0526, val/total_error=0.00631, val/diff_loss=0.0306]
======================================================================
📊 EPOCH 9 COMPLETE
======================================================================
  Train Loss:     0.043543
    - Diffusion:  0.038284
    - Pose:       0.038842
    - Grip:       0.034380
    - Phase:      0.052585
  Learning Rate:  0.00e+00
  Val Error:      0.006314
    - Pos Error:  0.003038
    - Rot Error:  0.001480
    - Grip Error: 0.078000
======================================================================

Epoch 10:  88% 750/850 [25:23<03:23,  2.03s/it, v_num=66td, train/loss_step=0.0424, diff_step=0.0399, pose_step=0.0351, grip_step=0.0739, phase_step=0.0247, train/loss_epoch=0.0435, diff_epoch=0.0383, pose_epoch=0.0388, grip_epoch=0.0344, phase_epoch=0.0526, val/total_error=0.00631, val/diff_loss=0.0306]
r/local/lib/python3.12/dist-packages/pytorch_lightning/loops/training_epoch_loop.py:224: You're resuming from a checkpoint that ended before the epoch ended and your dataloader is not resumable. This can cause unreliable results if further training is done. Consider using an end-of-epoch checkpoint or make your dataloader resumable by implementing the `state_dict` / `load_state_dict` interface.
Training: |          | 850/? [00:13<00:00, 63.30it/s, v_num=28v6, train/loss_step=0.0423, diff_step=0.0412, pose_step=0.0386, grip_step=0.0595, phase_step=0.0108, train/loss_epoch=0.0423, diff_epoch=0.0412, pose_epoch=0.0386, grip_epoch=0.0595, phase_epoch=0.0108]
======================================================================
📊 EPOCH 9 COMPLETE
======================================================================
  Train Loss:     0.042302
    - Diffusion:  0.041226
    - Pose:       0.038614
    - Grip:       0.059507
    - Phase:      0.010760
  Learning Rate:  1.00e-05
======================================================================

Epoch 10: 100% 638/638 [29:00<00:00,  2.73s/it, v_num=28v6, train/loss_step=0.0368, diff_step=0.0344, pose_step=0.036, grip_step=0.0233, phase_step=0.0239, train/loss_epoch=0.0423, diff_epoch=0.0412, pose_epoch=0.0386, grip_epoch=0.0595, phase_epoch=0.0108][2025-12-14 01:59:08,893][train_unified_planner][INFO] - End of epoch 10: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-14 01:59:36,059][train_unified_planner][INFO] - Failsafe backup saved to /content/drive/MyDrive/pda/models/backups/unified/unified_planner_backup_epoch_010.ckpt
Epoch 10: 100% 638/638 [29:27<00:00,  2.77s/it, v_num=28v6, train/loss_step=0.0368, diff_step=0.0344, pose_step=0.036, grip_step=0.0233, phase_step=0.0239, train/loss_epoch=0.0385, diff_epoch=0.0353, pose_epoch=0.0361, grip_epoch=0.0304, phase_epoch=0.0311]
======================================================================
📊 EPOCH 10 COMPLETE
======================================================================
  Train Loss:     0.038455
    - Diffusion:  0.035345
    - Pose:       0.036058
    - Grip:       0.030353
    - Phase:      0.031098
  Learning Rate:  1.00e-05
======================================================================

Epoch 11: 100% 638/638 [29:01<00:00,  2.73s/it, v_num=28v6, train/loss_step=0.0245, diff_step=0.0244, pose_step=0.0265, grip_step=0.00991, phase_step=0.000551, train/loss_epoch=0.0385, diff_epoch=0.0353, pose_epoch=0.0361, grip_epoch=0.0304, phase_epoch=0.0311][2025-12-14 02:28:37,818][train_unified_planner][INFO] - End of epoch 11: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-14 02:30:10,743][train_unified_planner][INFO] - Failsafe backup saved to /content/drive/MyDrive/pda/models/backups/unified/unified_planner_backup_epoch_011.ckpt

Validation: |          | 0/? [00:00<?, ?it/s][2025-12-14 02:30:10,779][lmdb_utils][INFO] -  Opened LMDB at /content/pda_data/val/validation_set.lmdb | mode=RO | file-mode

Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/62 [00:00<?, ?it/s]
Validation DataLoader 0:  81% 50/62 [04:38<01:06,  5.57s/it]
Validation DataLoader 0: 100% 62/62 [05:45<00:00,  5.58s/it]
Epoch 11: 100% 638/638 [36:23<00:00,  3.42s/it, v_num=28v6, train/loss_step=0.0245, diff_step=0.0244, pose_step=0.0265, grip_step=0.00991, phase_step=0.000551, train/loss_epoch=0.0385, diff_epoch=0.0353, pose_epoch=0.0361, grip_epoch=0.0304, phase_epoch=0.0311, val/total_error=0.00634, val/diff_loss=0.0281]Epoch 11, global step 11053: 'val/total_error' reached 0.00634 (best 0.00615), saving model to '/content/dgpo/checkpoint/checkpoints/udp-epoch=11-val/total_error=0.0063.ckpt' as top 3
Epoch 11: 100% 638/638 [38:58<00:00,  3.67s/it, v_num=28v6, train/loss_step=0.0245, diff_step=0.0244, pose_step=0.0265, grip_step=0.00991, phase_step=0.000551, train/loss_epoch=0.0374, diff_epoch=0.0347, pose_epoch=0.0355, grip_epoch=0.0294, phase_epoch=0.0269, val/total_error=0.00634, val/diff_loss=0.0281]
======================================================================
📊 EPOCH 11 COMPLETE
======================================================================
  Train Loss:     0.037398
    - Diffusion:  0.034711
    - Pose:       0.035473
    - Grip:       0.029374
    - Phase:      0.026875
  Learning Rate:  1.00e-05
  Val Error:      0.006340
    - Pos Error:  0.002867
    - Rot Error:  0.001429
    - Grip Error: 0.076538
======================================================================

Epoch 12: 100% 638/638 [29:35<00:00,  2.78s/it, v_num=28v6, train/loss_step=0.0336, diff_step=0.0334, pose_step=0.0358, grip_step=0.017, phase_step=0.00171, train/loss_epoch=0.0374, diff_epoch=0.0347, pose_epoch=0.0355, grip_epoch=0.0294, phase_epoch=0.0269, val/total_error=0.00634, val/diff_loss=0.0281] [2025-12-14 03:08:09,738][train_unified_planner][INFO] - End of epoch 12: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-14 03:10:35,784][train_unified_planner][INFO] - Failsafe backup saved to /content/drive/MyDrive/pda/models/backups/unified/unified_planner_backup_epoch_012.ckpt
Epoch 12: 100% 638/638 [32:01<00:00,  3.01s/it, v_num=28v6, train/loss_step=0.0336, diff_step=0.0334, pose_step=0.0358, grip_step=0.017, phase_step=0.00171, train/loss_epoch=0.0372, diff_epoch=0.0345, pose_epoch=0.0353, grip_epoch=0.0292, phase_epoch=0.0262, val/total_error=0.00634, val/diff_loss=0.0281]
======================================================================
📊 EPOCH 12 COMPLETE
======================================================================
  Train Loss:     0.037167
    - Diffusion:  0.034547
    - Pose:       0.035310
    - Grip:       0.029209
    - Phase:      0.026204
  Learning Rate:  1.00e-05
  Val Error:      0.006340
    - Pos Error:  0.002867
    - Rot Error:  0.001429
    - Grip Error: 0.076538
======================================================================

Epoch 13: 100% 638/638 [29:41<00:00,  2.79s/it, v_num=28v6, train/loss_step=0.033, diff_step=0.0312, pose_step=0.0319, grip_step=0.0266, phase_step=0.0184, train/loss_epoch=0.0372, diff_epoch=0.0345, pose_epoch=0.0353, grip_epoch=0.0292, phase_epoch=0.0262, val/total_error=0.00634, val/diff_loss=0.0281][2025-12-14 03:40:17,761][train_unified_planner][INFO] - End of epoch 13: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-14 03:42:36,941][train_unified_planner][INFO] - Failsafe backup saved to /content/drive/MyDrive/pda/models/backups/unified/unified_planner_backup_epoch_013.ckpt
[2025-12-14 03:42:36,947][train_unified_planner][INFO] - Cleaned up old backup: unified_planner_backup_epoch_010.ckpt

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/62 [00:00<?, ?it/s]
Validation DataLoader 0:  81% 50/62 [04:36<01:06,  5.53s/it]
Validation DataLoader 0: 100% 62/62 [05:41<00:00,  5.51s/it]
Epoch 13: 100% 638/638 [37:44<00:00,  3.55s/it, v_num=28v6, train/loss_step=0.033, diff_step=0.0312, pose_step=0.0319, grip_step=0.0266, phase_step=0.0184, train/loss_epoch=0.0372, diff_epoch=0.0345, pose_epoch=0.0353, grip_epoch=0.0292, phase_epoch=0.0262, val/total_error=0.0062, val/diff_loss=0.0265] Epoch 13, global step 12329: 'val/total_error' reached 0.00620 (best 0.00615), saving model to '/content/dgpo/checkpoint/checkpoints/udp-epoch=13-val/total_error=0.0062.ckpt' as top 3
Epoch 13: 100% 638/638 [40:17<00:00,  3.79s/it, v_num=28v6, train/loss_step=0.033, diff_step=0.0312, pose_step=0.0319, grip_step=0.0266, phase_step=0.0184, train/loss_epoch=0.0371, diff_epoch=0.0344, pose_epoch=0.0352, grip_epoch=0.0295, phase_epoch=0.027, val/total_error=0.0062, val/diff_loss=0.0265] 
======================================================================
📊 EPOCH 13 COMPLETE
======================================================================
  Train Loss:     0.037141
    - Diffusion:  0.034439
    - Pose:       0.035151
    - Grip:       0.029455
    - Phase:      0.027024
  Learning Rate:  1.00e-05
  Val Error:      0.006197
    - Pos Error:  0.002719
    - Rot Error:  0.001409
    - Grip Error: 0.072324
======================================================================

Epoch 14: 100% 638/638 [29:24<00:00,  2.77s/it, v_num=28v6, train/loss_step=0.0424, diff_step=0.039, pose_step=0.042, grip_step=0.0183, phase_step=0.0337, train/loss_epoch=0.0371, diff_epoch=0.0344, pose_epoch=0.0352, grip_epoch=0.0295, phase_epoch=0.027, val/total_error=0.0062, val/diff_loss=0.0265]   [2025-12-14 04:20:17,926][train_unified_planner][INFO] - End of epoch 14: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-14 04:22:53,605][train_unified_planner][INFO] - Failsafe backup saved to /content/drive/MyDrive/pda/models/backups/unified/unified_planner_backup_epoch_014.ckpt
[2025-12-14 04:22:53,610][train_unified_planner][INFO] - Cleaned up old backup: unified_planner_backup_epoch_011.ckpt
Epoch 14: 100% 638/638 [32:00<00:00,  3.01s/it, v_num=28v6, train/loss_step=0.0424, diff_step=0.039, pose_step=0.042, grip_step=0.0183, phase_step=0.0337, train/loss_epoch=0.036, diff_epoch=0.0334, pose_epoch=0.0342, grip_epoch=0.0279, phase_epoch=0.0255, val/total_error=0.0062, val/diff_loss=0.0265]
======================================================================
📊 EPOCH 14 COMPLETE
======================================================================
  Train Loss:     0.035986
    - Diffusion:  0.033432
    - Pose:       0.034220
    - Grip:       0.027917
    - Phase:      0.025541
  Learning Rate:  1.00e-05
  Val Error:      0.006197
    - Pos Error:  0.002719
    - Rot Error:  0.001409
    - Grip Error: 0.072324
======================================================================

Epoch 15: 100% 638/638 [29:17<00:00,  2.75s/it, v_num=28v6, train/loss_step=0.0266, diff_step=0.0261, pose_step=0.0261, grip_step=0.0254, phase_step=0.00505, train/loss_epoch=0.036, diff_epoch=0.0334, pose_epoch=0.0342, grip_epoch=0.0279, phase_epoch=0.0255, val/total_error=0.0062, val/diff_loss=0.0265][2025-12-14 04:52:11,232][train_unified_planner][INFO] - End of epoch 15: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-14 04:54:28,306][train_unified_planner][INFO] - Failsafe backup saved to /content/drive/MyDrive/pda/models/backups/unified/unified_planner_backup_epoch_015.ckpt
[2025-12-14 04:54:28,389][train_unified_planner][INFO] - Cleaned up old backup: unified_planner_backup_epoch_012.ckpt

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/62 [00:00<?, ?it/s]
Validation DataLoader 0:  81% 50/62 [04:40<01:07,  5.60s/it]
Validation DataLoader 0: 100% 62/62 [05:46<00:00,  5.59s/it]
Epoch 15: 100% 638/638 [37:25<00:00,  3.52s/it, v_num=28v6, train/loss_step=0.0266, diff_step=0.0261, pose_step=0.0261, grip_step=0.0254, phase_step=0.00505, train/loss_epoch=0.036, diff_epoch=0.0334, pose_epoch=0.0342, grip_epoch=0.0279, phase_epoch=0.0255, val/total_error=0.00546, val/diff_loss=0.0248]Epoch 15, global step 13605: 'val/total_error' reached 0.00546 (best 0.00546), saving model to '/content/dgpo/checkpoint/checkpoints/udp-epoch=15-val/total_error=0.0055.ckpt' as top 3
Epoch 15: 100% 638/638 [39:59<00:00,  3.76s/it, v_num=28v6, train/loss_step=0.0266, diff_step=0.0261, pose_step=0.0261, grip_step=0.0254, phase_step=0.00505, train/loss_epoch=0.0351, diff_epoch=0.0331, pose_epoch=0.0339, grip_epoch=0.0274, phase_epoch=0.0197, val/total_error=0.00546, val/diff_loss=0.0248]
======================================================================
📊 EPOCH 15 COMPLETE
======================================================================
  Train Loss:     0.035100
    - Diffusion:  0.033125
    - Pose:       0.033949
    - Grip:       0.027362
    - Phase:      0.019746
  Learning Rate:  1.00e-05
  Val Error:      0.005461
    - Pos Error:  0.002696
    - Rot Error:  0.001400
    - Grip Error: 0.066752
======================================================================

Epoch 16: 100% 638/638 [29:22<00:00,  2.76s/it, v_num=28v6, train/loss_step=0.0317, diff_step=0.0299, pose_step=0.0289, grip_step=0.0373, phase_step=0.0175, train/loss_epoch=0.0351, diff_epoch=0.0331, pose_epoch=0.0339, grip_epoch=0.0274, phase_epoch=0.0197, val/total_error=0.00546, val/diff_loss=0.0248] [2025-12-14 05:32:15,162][train_unified_planner][INFO] - End of epoch 16: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-14 05:35:32,235][train_unified_planner][INFO] - Failsafe backup saved to /content/drive/MyDrive/pda/models/backups/unified/unified_planner_backup_epoch_016.ckpt
[2025-12-14 05:35:32,242][train_unified_planner][INFO] - Cleaned up old backup: unified_planner_backup_epoch_013.ckpt
Epoch 16: 100% 638/638 [32:39<00:00,  3.07s/it, v_num=28v6, train/loss_step=0.0317, diff_step=0.0299, pose_step=0.0289, grip_step=0.0373, phase_step=0.0175, train/loss_epoch=0.0342, diff_epoch=0.0327, pose_epoch=0.0336, grip_epoch=0.0261, phase_epoch=0.0157, val/total_error=0.00546, val/diff_loss=0.0248]
======================================================================
📊 EPOCH 16 COMPLETE
======================================================================
  Train Loss:     0.034230
    - Diffusion:  0.032657
    - Pose:       0.033587
    - Grip:       0.026147
    - Phase:      0.015724
  Learning Rate:  1.00e-05
  Val Error:      0.005461
    - Pos Error:  0.002696
    - Rot Error:  0.001400
    - Grip Error: 0.066752
======================================================================

Epoch 17: 100% 638/638 [29:27<00:00,  2.77s/it, v_num=28v6, train/loss_step=0.0353, diff_step=0.0259, pose_step=0.0266, grip_step=0.0207, phase_step=0.0943, train/loss_epoch=0.0342, diff_epoch=0.0327, pose_epoch=0.0336, grip_epoch=0.0261, phase_epoch=0.0157, val/total_error=0.00546, val/diff_loss=0.0248][2025-12-14 06:05:00,036][train_unified_planner][INFO] - End of epoch 17: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-14 06:08:03,790][train_unified_planner][INFO] - Failsafe backup saved to /content/drive/MyDrive/pda/models/backups/unified/unified_planner_backup_epoch_017.ckpt
[2025-12-14 06:08:03,801][train_unified_planner][INFO] - Cleaned up old backup: unified_planner_backup_epoch_014.ckpt

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/62 [00:00<?, ?it/s]
Validation DataLoader 0:  81% 50/62 [04:35<01:06,  5.51s/it]
Validation DataLoader 0: 100% 62/62 [05:42<00:00,  5.52s/it]
Epoch 17: 100% 638/638 [38:16<00:00,  3.60s/it, v_num=28v6, train/loss_step=0.0353, diff_step=0.0259, pose_step=0.0266, grip_step=0.0207, phase_step=0.0943, train/loss_epoch=0.0342, diff_epoch=0.0327, pose_epoch=0.0336, grip_epoch=0.0261, phase_epoch=0.0157, val/total_error=0.00644, val/diff_loss=0.0258]Epoch 17, global step 14881: 'val/total_error' was not in top 3
Epoch 17: 100% 638/638 [38:16<00:00,  3.60s/it, v_num=28v6, train/loss_step=0.0353, diff_step=0.0259, pose_step=0.0266, grip_step=0.0207, phase_step=0.0943, train/loss_epoch=0.0346, diff_epoch=0.0325, pose_epoch=0.0333, grip_epoch=0.0268, phase_epoch=0.0212, val/total_error=0.00644, val/diff_loss=0.0258]
======================================================================
📊 EPOCH 17 COMPLETE
======================================================================
  Train Loss:     0.034596
    - Diffusion:  0.032479
    - Pose:       0.033293
    - Grip:       0.026780
    - Phase:      0.021171
  Learning Rate:  1.00e-05
  Val Error:      0.006439
    - Pos Error:  0.003162
    - Rot Error:  0.001542
    - Grip Error: 0.074454
======================================================================

Epoch 18:   0% 0/638 [00:00<?, ?it/s, v_num=28v6, train/loss_step=0.0353, diff_step=0.0259, pose_step=0.0266, grip_step=0.0207, phase_step=0.0943, train/loss_epoch=0.0346, diff_epoch=0.0325, pose_epoch=0.0333, grip_epoch=0.0268, phase_epoch=0.0212, val/total_error=0.00644, val/diff_loss=0.0258]

[ ]