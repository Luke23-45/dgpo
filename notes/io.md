Restored all states from the checkpoint at /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_006.ckpt
Sanity Checking: |          | 0/? [00:00<?, ?it/s][2025-11-27 13:04:23,947][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode
[2025-11-27 13:04:23,947][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode
/usr/local/lib/python3.12/dist-packages/torch/utils/data/_utils/pin_memory.py:57: DeprecationWarning: The argument 'device' of Tensor.pin_memory() is deprecated. Please do not pass this argument. (Triggered internally at /pytorch/aten/src/ATen/native/Memory.cpp:46.)
  return data.pin_memory(device)
/usr/local/lib/python3.12/dist-packages/torch/utils/data/_utils/pin_memory.py:57: DeprecationWarning: The argument 'device' of Tensor.is_pinned() is deprecated. Please do not pass this argument. (Triggered internally at /pytorch/aten/src/ATen/native/Memory.cpp:31.)
  return data.pin_memory(device)
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/fit_loop.py:527: Found 160 module(s) in eval mode at the start of training. This may lead to unexpected behavior during training. If this is intentional, you can ignore this warning.
Epoch 6: 100% 1248/1248 [29:17<00:00,  1.41s/it, v_num=o3qg, train/loss_step=0.130, train/loss_pose_step=0.092, train/loss_grip_step=0.000742, train/loss_phase_step=0.0754] [2025-11-27 13:33:51,477][train_semantic_planner][INFO] - End of epoch 6: Triggering atomic failsafe backup...
[2025-11-27 13:34:50,825][train_semantic_planner][INFO] - Failsafe backup for epoch 6 saved successfully to /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_006.ckpt.
[2025-11-27 13:34:50,834][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_006.ckpt
Epoch 7: 100% 1248/1248 [29:24<00:00,  1.41s/it, v_num=o3qg, train/loss_step=0.113, train/loss_pose_step=0.0982, train/loss_grip_step=0.0121, train/loss_phase_step=0.00508, train/loss_epoch=0.264, train/loss_pose_epoch=0.168, train/loss_grip_epoch=0.0647, train/loss_phase_epoch=0.0621, train/phase_acc=0.977][2025-11-27 14:04:15,564][train_semantic_planner][INFO] - End of epoch 7: Triggering atomic failsafe backup...
[2025-11-27 14:05:57,911][train_semantic_planner][INFO] - Failsafe backup for epoch 7 saved successfully to /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_007.ckpt.
[2025-11-27 14:05:57,918][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_007.ckpt

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:09<00:43,  1.01it/s]
Validation DataLoader 0:  37% 20/54 [00:24<00:41,  1.22s/it]
Validation DataLoader 0:  56% 30/54 [00:35<00:28,  1.18s/it]
Validation DataLoader 0:  74% 40/54 [00:46<00:16,  1.17s/it]
Validation DataLoader 0:  93% 50/54 [00:57<00:04,  1.16s/it]
Validation DataLoader 0: 100% 54/54 [01:01<00:00,  1.14s/it]
Epoch 8:  58% 730/1248 [17:12<12:12,  1.41s/it, v_num=o3qg, train/loss_step=0.627, train/loss_pose_step=0.538, train/loss_grip_step=0.0666, train/loss_phase_step=0.0439, train/loss_epoch=0.223, train/loss_pose_epoch=0.160, train/loss_grip_epoch=0.0401, train/loss_phase_epoch=0.0452, train/phase_acc=0.982, val/loss=0.130, val/pos_error_m=0.0192]

[ ]
025-11-27 16:36:41,341][train_semantic_planner][INFO] - End of epoch 7: Triggering atomic failsafe backup...
[2025-11-27 16:37:29,992][train_semantic_planner][INFO] - Failsafe backup for epoch 7 saved successfully to /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_007.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s][2025-11-27 16:37:31,196][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode
[2025-11-27 16:37:31,198][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode

Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:08<00:36,  1.20it/s]
Validation DataLoader 0:  37% 20/54 [00:37<01:03,  1.86s/it]
Validation DataLoader 0:  56% 30/54 [00:48<00:38,  1.60s/it]
Validation DataLoader 0:  74% 40/54 [00:58<00:20,  1.47s/it]
Validation DataLoader 0:  93% 50/54 [01:07<00:05,  1.35s/it]
Validation DataLoader 0: 100% 54/54 [01:11<00:00,  1.33s/it]
Epoch 8: 100% 1248/1248 [28:45<00:00,  1.38s/it, v_num=4157, train/loss_step=0.286, train/loss_pose_step=0.144, train/loss_grip_step=0.0559, train/loss_phase_step=0.171, val/loss=0.130, val/pos_error_m=0.0197, train/loss_epoch=0.0721, train/loss_pose_epoch=0.0551, train/loss_grip_epoch=0.0144, train/loss_phase_epoch=0.00515, train/phase_acc=1.000][2025-11-27 17:08:56,432][train_semantic_planner][INFO] - End of epoch 8: Triggering atomic failsafe backup...
[2025-11-27 17:11:13,427][train_semantic_planner][INFO] - Failsafe backup for epoch 8 saved successfully to /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_008.ckpt.
Epoch 9: 100% 1248/1248 [28:48<00:00,  1.38s/it, v_num=4157, train/loss_step=0.159, train/loss_pose_step=0.0818, train/loss_grip_step=0.0715, train/loss_phase_step=0.0111, val/loss=0.130, val/pos_error_m=0.0197, train/loss_epoch=0.210, train/loss_pose_epoch=0.159, train/loss_grip_epoch=0.0307, train/loss_phase_epoch=0.0391, train/phase_acc=0.985][2025-11-27 17:40:01,531][train_semantic_planner][INFO] - End of epoch 9: Triggering atomic failsafe backup...
[2025-11-27 17:42:03,074][train_semantic_planner][INFO] - Failsafe backup for epoch 9 saved successfully to /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_009.ckpt.
[2025-11-27 17:42:03,080][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_007.ckpt

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:08<00:37,  1.18it/s]
Validation DataLoader 0:  37% 20/54 [00:20<00:34,  1.02s/it]
Validation DataLoader 0:  56% 30/54 [00:28<00:22,  1.06it/s]
Validation DataLoader 0:  74% 40/54 [00:37<00:13,  1.08it/s]
Validation DataLoader 0:  93% 50/54 [00:47<00:03,  1.05it/s]
Validation DataLoader 0: 100% 54/54 [00:50<00:00,  1.06it/s]
Epoch 10: 100% 1248/1248 [28:48<00:00,  1.38s/it, v_num=4157, train/loss_step=0.110, train/loss_pose_step=0.089, train/loss_grip_step=0.0186, train/loss_phase_step=0.00399, val/loss=0.148, val/pos_error_m=0.0216, train/loss_epoch=0.200, train/loss_pose_epoch=0.154, train/loss_grip_epoch=0.0276, train/loss_phase_epoch=0.0375, train/phase_acc=0.986][2025-11-27 18:13:50,006][train_semantic_planner][INFO] - End of epoch 10: Triggering atomic failsafe backup...
[2025-11-27 18:15:43,610][train_semantic_planner][INFO] - Failsafe backup for epoch 10 saved successfully to /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_010.ckpt.
[2025-11-27 18:15:43,625][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_008.ckpt
Epoch 11: 100% 1248/1248 [28:50<00:00,  1.39s/it, v_num=4157, train/loss_step=0.131, train/loss_pose_step=0.0467, train/loss_grip_step=0.084, train/loss_phase_step=0.000431, val/loss=0.148, val/pos_error_m=0.0216, train/loss_epoch=0.195, train/loss_pose_epoch=0.148, train/loss_grip_epoch=0.0288, train/loss_phase_epoch=0.036, train/phase_acc=0.986][2025-11-27 18:44:34,117][train_semantic_planner][INFO] - End of epoch 11: Triggering atomic failsafe backup...
[2025-11-27 18:46:59,737][train_semantic_planner][INFO] - Failsafe backup for epoch 11 saved successfully to /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_011.ckpt.
[2025-11-27 18:46:59,748][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_009.ckpt

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:09<00:40,  1.09it/s]
Validation DataLoader 0:  37% 20/54 [00:21<00:35,  1.05s/it]
Validation DataLoader 0:  56% 30/54 [00:29<00:23,  1.00it/s]
Validation DataLoader 0:  74% 40/54 [00:39<00:13,  1.02it/s]
Validation DataLoader 0:  93% 50/54 [00:46<00:03,  1.06it/s]
Validation DataLoader 0: 100% 54/54 [00:49<00:00,  1.08it/s]
Epoch 12: 100% 1248/1248 [28:49<00:00,  1.39s/it, v_num=4157, train/loss_step=0.120, train/loss_pose_step=0.0715, train/loss_grip_step=0.0437, train/loss_phase_step=0.00921, val/loss=0.150, val/pos_error_m=0.0196, train/loss_epoch=0.187, train/loss_pose_epoch=0.139, train/loss_grip_epoch=0.0336, train/loss_phase_epoch=0.0295, train/phase_acc=0.989][2025-11-27 19:19:55,721][train_semantic_planner][INFO] - End of epoch 12: Triggering atomic failsafe backup...
[2025-11-27 19:21:41,512][train_semantic_planner][INFO] - Failsafe backup for epoch 12 saved successfully to /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_012.ckpt.
[2025-11-27 19:21:41,520][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_010.ckpt
Epoch 13: 100% 1248/1248 [28:49<00:00,  1.39s/it, v_num=4157, train/loss_step=0.317, train/loss_pose_step=0.246, train/loss_grip_step=0.0348, train/loss_phase_step=0.0718, val/loss=0.150, val/pos_error_m=0.0196, train/loss_epoch=0.186, train/loss_pose_epoch=0.136, train/loss_grip_epoch=0.0312, train/loss_phase_epoch=0.0373, train/phase_acc=0.987][2025-11-27 19:50:31,275][train_semantic_planner][INFO] - End of epoch 13: Triggering atomic failsafe backup...
[2025-11-27 19:52:53,127][train_semantic_planner][INFO] - Failsafe backup for epoch 13 saved successfully to /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_013.ckpt.
[2025-11-27 19:52:53,135][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_011.ckpt

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:07<00:34,  1.28it/s]
Validation DataLoader 0:  37% 20/54 [00:19<00:33,  1.01it/s]
Validation DataLoader 0:  56% 30/54 [00:29<00:23,  1.01it/s]
Validation DataLoader 0:  74% 40/54 [00:37<00:13,  1.07it/s]
Validation DataLoader 0:  93% 50/54 [00:46<00:03,  1.08it/s]
Validation DataLoader 0: 100% 54/54 [00:49<00:00,  1.10it/s]
Epoch 14:  54% 670/1248 [15:30<13:22,  1.39s/it, v_num=4157, train/loss_step=0.098, train/loss_pose_step=0.0977, train/loss_grip_step=1.35e-6, train/loss_phase_step=0.000628, val/loss=0.140, val/pos_error_m=0.0187, train/loss_epoch=0.180, train/loss_pose_epoch=0.132, train/loss_grip_epoch=0.0328, train/loss_phase_epoch=0.0304, train/phase_acc=0.989]

[ ]
2025-11-28 08:16:50,951][train_semantic_planner][INFO] - End of epoch 14: Triggering atomic failsafe backup...
[2025-11-28 08:17:10,785][train_semantic_planner][INFO] - Failsafe backup for epoch 14 saved successfully to /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_014.ckpt.
Epoch 15: 100% 1248/1248 [28:53<00:00,  1.39s/it, v_num=tsv7, train/loss_step=0.0545, train/loss_pose_step=0.0465, train/loss_grip_step=0.00447, train/loss_phase_step=0.00697, train/loss_epoch=0.075, train/loss_pose_epoch=0.0654, train/loss_grip_epoch=0.00781, train/loss_phase_epoch=0.00344, train/phase_acc=1.000][2025-11-28 08:46:04,227][train_semantic_planner][INFO] - End of epoch 15: Triggering atomic failsafe backup...
[2025-11-28 08:46:48,726][train_semantic_planner][INFO] - Failsafe backup for epoch 15 saved successfully to /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_015.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s][2025-11-28 08:46:50,121][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode
[2025-11-28 08:46:50,127][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode

Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:12<00:55,  1.25s/it]
Validation DataLoader 0:  37% 20/54 [00:22<00:38,  1.13s/it]
Validation DataLoader 0:  56% 30/54 [00:34<00:27,  1.15s/it]
Validation DataLoader 0:  74% 40/54 [00:45<00:15,  1.13s/it]
Validation DataLoader 0:  93% 50/54 [00:53<00:04,  1.08s/it]
Validation DataLoader 0: 100% 54/54 [00:57<00:00,  1.07s/it]
Epoch 16: 100% 1248/1248 [28:53<00:00,  1.39s/it, v_num=tsv7, train/loss_step=0.115, train/loss_pose_step=0.0805, train/loss_grip_step=0.0267, train/loss_phase_step=0.0149, train/loss_epoch=0.170, train/loss_pose_epoch=0.125, train/loss_grip_epoch=0.0325, train/loss_phase_epoch=0.0247, train/phase_acc=0.991, val/loss=0.142, val/pos_error_m=0.0206]  [2025-11-28 09:17:46,554][train_semantic_planner][INFO] - End of epoch 16: Triggering atomic failsafe backup...
[2025-11-28 09:19:13,679][train_semantic_planner][INFO] - Failsafe backup for epoch 16 saved successfully to /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_016.ckpt.
[2025-11-28 09:19:13,684][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_014.ckpt
Epoch 17: 100% 1248/1248 [28:52<00:00,  1.39s/it, v_num=tsv7, train/loss_step=0.0345, train/loss_pose_step=0.0255, train/loss_grip_step=0.0085, train/loss_phase_step=0.000877, train/loss_epoch=0.164, train/loss_pose_epoch=0.118, train/loss_grip_epoch=0.0327, train/loss_phase_epoch=0.0267, train/phase_acc=0.990, val/loss=0.142, val/pos_error_m=0.0206][2025-11-28 09:48:06,626][train_semantic_planner][INFO] - End of epoch 17: Triggering atomic failsafe backup...
[2025-11-28 09:49:51,722][train_semantic_planner][INFO] - Failsafe backup for epoch 17 saved successfully to /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_017.ckpt.
[2025-11-28 09:49:51,729][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_015.ckpt

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:10<00:47,  1.07s/it]
Validation DataLoader 0:  37% 20/54 [00:20<00:34,  1.01s/it]
Validation DataLoader 0:  56% 30/54 [00:30<00:24,  1.01s/it]
Validation DataLoader 0:  74% 40/54 [00:39<00:13,  1.00it/s]
Validation DataLoader 0:  93% 50/54 [00:47<00:03,  1.04it/s]
Validation DataLoader 0: 100% 54/54 [00:51<00:00,  1.04it/s]
Epoch 18:  11% 140/1248 [03:15<25:49,  1.40s/it, v_num=tsv7, train/loss_step=0.168, train/loss_pose_step=0.148, train/loss_grip_step=0.0185, train/loss_phase_step=0.003, train/loss_epoch=0.155, train/loss_pose_epoch=0.113, train/loss_grip_epoch=0.0306, train/loss_phase_epoch=0.0234, train/phase_acc=0.991, val/loss=0.136, val/pos_error_m=0.0213]    

2025-11-28 15:38:42,826][train_semantic_planner][INFO] - Failsafe backup for epoch 17 saved successfully to /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_017.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s][2025-11-28 15:38:45,195][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode
[2025-11-28 15:38:45,197][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode

Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:26<01:56,  2.64s/it]
Validation DataLoader 0:  37% 20/54 [00:36<01:01,  1.80s/it]
Validation DataLoader 0:  56% 30/54 [00:45<00:36,  1.51s/it]
Validation DataLoader 0:  74% 40/54 [00:56<00:19,  1.41s/it]
Validation DataLoader 0:  93% 50/54 [01:07<00:05,  1.35s/it]
Validation DataLoader 0: 100% 54/54 [01:10<00:00,  1.31s/it]
Epoch 18: 100% 1248/1248 [31:37<00:00,  1.52s/it, v_num=fy9v, train/loss_step=0.0647, train/loss_pose_step=0.0418, train/loss_grip_step=0.0219, train/loss_phase_step=0.002, val/loss=0.137, val/pos_error_m=0.0281, train/loss_epoch=0.0491, train/loss_pose_epoch=0.0472, train/loss_grip_epoch=0.00141, train/loss_phase_epoch=0.000988, train/phase_acc=1.000]   [2025-11-28 16:11:36,143][train_semantic_planner][INFO] - End of epoch 18: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-11-28 16:12:58,882][train_semantic_planner][INFO] - Failsafe backup for epoch 18 saved successfully to /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_018.ckpt.
Epoch 19: 100% 1248/1248 [31:43<00:00,  1.53s/it, v_num=fy9v, train/loss_step=0.0563, train/loss_pose_step=0.0479, train/loss_grip_step=0.00594, train/loss_phase_step=0.00484, val/loss=0.137, val/pos_error_m=0.0281, train/loss_epoch=0.155, train/loss_pose_epoch=0.114, train/loss_grip_epoch=0.0306, train/loss_phase_epoch=0.0221, train/phase_acc=0.992][2025-11-28 16:44:42,904][train_semantic_planner][INFO] - End of epoch 19: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-11-28 16:46:28,878][train_semantic_planner][INFO] - Failsafe backup for epoch 19 saved successfully to /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_019.ckpt.
[2025-11-28 16:46:28,891][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_017.ckpt

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:08<00:36,  1.21it/s]
Validation DataLoader 0:  37% 20/54 [00:20<00:35,  1.04s/it]
Validation DataLoader 0:  56% 30/54 [00:31<00:25,  1.04s/it]
Validation DataLoader 0:  74% 40/54 [00:39<00:13,  1.00it/s]
Validation DataLoader 0:  93% 50/54 [00:49<00:03,  1.02it/s]
Validation DataLoader 0: 100% 54/54 [00:52<00:00,  1.03it/s]


Epoch 20: 100% 1248/1248 [31:41<00:00,  1.52s/it, v_num=fy9v, train/loss_step=0.354, train/loss_pose_step=0.198, train/loss_grip_step=0.152, train/loss_phase_step=0.00698, val/loss=0.134, val/pos_error_m=0.020, train/loss_epoch=0.144, train/loss_pose_epoch=0.106, train/loss_grip_epoch=0.0286, train/loss_phase_epoch=0.0181, train/phase_acc=0.993][2025-11-28 17:19:05,052][train_semantic_planner][INFO] - End of epoch 20: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-11-28 17:20:58,908][train_semantic_planner][INFO] - Failsafe backup for epoch 20 saved successfully to /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_020.ckpt.
[2025-11-28 17:20:58,917][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_018.ckpt
Epoch 21: 100% 1248/1248 [31:41<00:00,  1.52s/it, v_num=fy9v, train/loss_step=0.450, train/loss_pose_step=0.196, train/loss_grip_step=0.215, train/loss_phase_step=0.078, val/loss=0.134, val/pos_error_m=0.020, train/loss_epoch=0.139, train/loss_pose_epoch=0.0993, train/loss_grip_epoch=0.0291, train/loss_phase_epoch=0.0203, train/phase_acc=0.993]     [2025-11-28 17:52:40,852][train_semantic_planner][INFO] - End of epoch 21: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-11-28 17:54:31,381][train_semantic_planner][INFO] - Failsafe backup for epoch 21 saved successfully to /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_021.ckpt.
[2025-11-28 17:54:31,389][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_019.ckpt

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:09<00:42,  1.04it/s]
Validation DataLoader 0:  37% 20/54 [00:21<00:36,  1.08s/it]
Validation DataLoader 0:  56% 30/54 [00:31<00:24,  1.04s/it]
Validation DataLoader 0:  74% 40/54 [00:40<00:14,  1.02s/it]
Validation DataLoader 0:  93% 50/54 [00:49<00:03,  1.00it/s]
Validation DataLoader 0: 100% 54/54 [00:52<00:00,  1.02it/s]


Epoch 22: 100% 1248/1248 [31:44<00:00,  1.53s/it, v_num=fy9v, train/loss_step=0.0751, train/loss_pose_step=0.0515, train/loss_grip_step=0.0184, train/loss_phase_step=0.0102, val/loss=0.142, val/pos_error_m=0.0166, train/loss_epoch=0.143, train/loss_pose_epoch=0.097, train/loss_grip_epoch=0.0344, train/loss_phase_epoch=0.0239, train/phase_acc=0.991] [2025-11-28 18:28:38,035][train_semantic_planner][INFO] - End of epoch 22: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-11-28 18:30:24,322][train_semantic_planner][INFO] - Failsafe backup for epoch 22 saved successfully to /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_022.ckpt.
[2025-11-28 18:30:24,334][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_020.ckpt
Epoch 23:  39% 490/1248 [12:29<19:19,  1.53s/it, v_num=fy9v, train/loss_step=0.108, train/loss_pose_step=0.083, train/loss_grip_step=0.0204, train/loss_phase_step=0.00865, val/loss=0.142, val/pos_error_m=0.0166, train/loss_epoch=0.140, train/loss_pose_epoch=0.0956, train/loss_grip_epoch=0.032, train/loss_phase_epoch=0.0253, train/phase_acc=0.991]    

Epoch 23: 100% 1248/1248 [31:44<00:00,  1.53s/it, v_num=fy9v, train/loss_step=0.152, train/loss_pose_step=0.123, train/loss_grip_step=0.0149, train/loss_phase_step=0.0286, val/loss=0.142, val/pos_error_m=0.0166, train/loss_epoch=0.140, train/loss_pose_epoch=0.0956, train/loss_grip_epoch=0.032, train/loss_phase_epoch=0.0253, train/phase_acc=0.991]  [2025-11-28 19:02:08,439][train_semantic_planner][INFO] - End of epoch 23: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-11-28 19:03:49,835][train_semantic_planner][INFO] - Failsafe backup for epoch 23 saved successfully to /content/drive/MyDrive/pda/models/v1/backups/backup_epoch_023.ckpt.
[2025-11-28 19:03:49,840][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_021.ckpt

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:09<00:41,  1.05it/s]
Validation DataLoader 0:  37% 20/54 [00:19<00:33,  1.02it/s]
Validation DataLoader 0:  56% 30/54 [00:31<00:24,  1.04s/it]
Validation DataLoader 0:  74% 40/54 [00:40<00:14,  1.02s/it]
Validation DataLoader 0:  93% 50/54 [00:49<00:03,  1.01it/s]
Validation DataLoader 0: 100% 54/54 [00:53<00:00,  1.02it/s]
Epoch 24: 100% 1248/1248 [31:41<00:00,  1.52s/it, v_num=fy9v, train/loss_step=0.143, train/loss_pose_step=0.0378, train/loss_grip_step=0.024, train/loss_phase_step=0.162, val/loss=0.144, val/pos_error_m=0.0205, train/loss_epoch=0.126, train/loss_pose_epoch=0.0924, train/loss_grip_epoch=0.0265, train/loss_phase_epoch=0.015, train/phase_acc=0.995]  [2025-11-28 19:36:25,975][train_semantic_planner][INFO] - End of epoch 24: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.


025-12-01 16:46:17,324][train_semantic_planner][INFO] - End of epoch 23: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-01 16:46:48,368][train_semantic_planner][INFO] - Failsafe backup for epoch 23 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/backup_epoch_023.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s][2025-12-01 16:46:48,563][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode
[2025-12-01 16:46:48,570][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode

Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:11<00:50,  1.14s/it]
Validation DataLoader 0:  37% 20/54 [00:22<00:37,  1.11s/it]
Validation DataLoader 0:  56% 30/54 [00:33<00:26,  1.11s/it]
Validation DataLoader 0:  74% 40/54 [00:43<00:15,  1.08s/it]
Validation DataLoader 0:  93% 50/54 [00:52<00:04,  1.06s/it]
Validation DataLoader 0: 100% 54/54 [00:56<00:00,  1.04s/it]
Epoch 24: 100% 1248/1248 [30:51<00:00,  1.48s/it, v_num=0fbc, train/loss_step=0.163, train/loss_pose_step=0.0356, train/loss_grip_step=0.0178, train/loss_phase_step=0.220, val/loss=0.145, val/pos_error_m=0.021, train/loss_epoch=0.0526, train/loss_pose_epoch=0.0482, train/loss_grip_epoch=0.00101, train/loss_phase_epoch=0.00659, train/phase_acc=1.000]  [2025-12-01 17:18:39,648][train_semantic_planner][INFO] - End of epoch 24: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-01 17:19:17,068][train_semantic_planner][INFO] - Failsafe backup for epoch 24 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/backup_epoch_024.ckpt.
Epoch 25: 100% 1248/1248 [30:54<00:00,  1.49s/it, v_num=0fbc, train/loss_step=0.151, train/loss_pose_step=0.127, train/loss_grip_step=0.0214, train/loss_phase_step=0.00389, val/loss=0.145, val/pos_error_m=0.021, train/loss_epoch=0.116, train/loss_pose_epoch=0.0851, train/loss_grip_epoch=0.024, train/loss_phase_epoch=0.0144, train/phase_acc=0.995] [2025-12-01 17:50:11,634][train_semantic_planner][INFO] - End of epoch 25: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-01 17:51:46,484][train_semantic_planner][INFO] - Failsafe backup for epoch 25 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/backup_epoch_025.ckpt.
[2025-12-01 17:51:46,489][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_023.ckpt

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:09<00:42,  1.04it/s]
Validation DataLoader 0:  37% 20/54 [00:22<00:38,  1.14s/it]
Validation DataLoader 0:  56% 30/54 [00:31<00:25,  1.05s/it]
Validation DataLoader 0:  74% 40/54 [00:41<00:14,  1.05s/it]
Validation DataLoader 0:  93% 50/54 [00:52<00:04,  1.04s/it]
Validation DataLoader 0: 100% 54/54 [00:55<00:00,  1.02s/it]
Epoch 26: 100% 1248/1248 [30:52<00:00,  1.48s/it, v_num=0fbc, train/loss_step=0.0647, train/loss_pose_step=0.0455, train/loss_grip_step=0.0156, train/loss_phase_step=0.00731, val/loss=0.136, val/pos_error_m=0.0202, train/loss_epoch=0.121, train/loss_pose_epoch=0.0859, train/loss_grip_epoch=0.0269, train/loss_phase_epoch=0.0172, train/phase_acc=0.994][2025-12-01 18:23:36,427][train_semantic_planner][INFO] - End of epoch 26: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-01 18:25:27,902][train_semantic_planner][INFO] - Failsafe backup for epoch 26 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/backup_epoch_026.ckpt.
[2025-12-01 18:25:27,913][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_024.ckpt
Epoch 27: 100% 1248/1248 [30:54<00:00,  1.49s/it, v_num=0fbc, train/loss_step=0.0568, train/loss_pose_step=0.033, train/loss_grip_step=0.0233, train/loss_phase_step=0.000803, val/loss=0.136, val/pos_error_m=0.0202, train/loss_epoch=0.116, train/loss_pose_epoch=0.0822, train/loss_grip_epoch=0.0267, train/loss_phase_epoch=0.0139, train/phase_acc=0.995][2025-12-01 18:56:22,248][train_semantic_planner][INFO] - End of epoch 27: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-01 18:57:53,820][train_semantic_planner][INFO] - Failsafe backup for epoch 27 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/backup_epoch_027.ckpt.
[2025-12-01 18:57:53,830][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_025.ckpt

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:09<00:43,  1.02it/s]
Validation DataLoader 0:  37% 20/54 [00:22<00:38,  1.12s/it]
Validation DataLoader 0:  56% 30/54 [00:31<00:24,  1.04s/it]
Validation DataLoader 0:  74% 40/54 [00:41<00:14,  1.03s/it]
Validation DataLoader 0:  93% 50/54 [00:52<00:04,  1.04s/it]
Validation DataLoader 0: 100% 54/54 [00:55<00:00,  1.02s/it]
Epoch 28: 100% 1248/1248 [30:52<00:00,  1.48s/it, v_num=0fbc, train/loss_step=0.0867, train/loss_pose_step=0.0683, train/loss_grip_step=0.0171, train/loss_phase_step=0.00259, val/loss=0.148, val/pos_error_m=0.0168, train/loss_epoch=0.113, train/loss_pose_epoch=0.0813, train/loss_grip_epoch=0.026, train/loss_phase_epoch=0.0117, train/phase_acc=0.996][2025-12-01 19:31:35,868][train_semantic_planner][INFO] - End of epoch 28: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-01 19:33:16,563][train_semantic_planner][INFO] - Failsafe backup for epoch 28 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/backup_epoch_028.ckpt.
[2025-12-01 19:33:16,569][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_026.ckpt
Epoch 29: 100% 1248/1248 [30:51<00:00,  1.48s/it, v_num=0fbc, train/loss_step=0.0662, train/loss_pose_step=0.0621, train/loss_grip_step=0.00375, train/loss_phase_step=0.000823, val/loss=0.148, val/pos_error_m=0.0168, train/loss_epoch=0.110, train/loss_pose_epoch=0.0805, train/loss_grip_epoch=0.0234, train/loss_phase_epoch=0.0117, train/phase_acc=0.996][2025-12-01 20:04:08,399][train_semantic_planner][INFO] - End of epoch 29: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-01 20:06:23,869][train_semantic_planner][INFO] - Failsafe backup for epoch 29 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/backup_epoch_029.ckpt.
[2025-12-01 20:06:23,876][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_027.ckpt

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:09<00:43,  1.02it/s]
Validation DataLoader 0:  37% 20/54 [00:21<00:35,  1.05s/it]
Validation DataLoader 0:  56% 30/54 [00:29<00:23,  1.02it/s]
Validation DataLoader 0:  74% 40/54 [00:39<00:13,  1.02it/s]
Validation DataLoader 0:  93% 50/54 [00:48<00:03,  1.02it/s]
Validation DataLoader 0: 100% 54/54 [00:52<00:00,  1.03it/s]
Epoch 30:  84% 1050/1248 [25:58<04:53,  1.48s/it, v_num=0fbc, train/loss_step=0.0333, train/loss_pose_step=0.0282, train/loss_grip_step=0.00506, train/loss_phase_step=0.00011, val/loss=0.159, val/pos_error_m=0.0153, train/loss_epoch=0.0997, train/loss_pose_epoch=0.0746, train/loss_grip_epoch=0.0203, train/loss_phase_epoch=0.00954, train/phase_acc=0.997]
alidation: |          | 0/? [00:00<?, ?it/s][2025-12-02 12:04:07,842][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode
[2025-12-02 12:04:08,795][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode

Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:11<00:48,  1.10s/it]
Validation DataLoader 0:  37% 20/54 [00:21<00:36,  1.06s/it]
Validation DataLoader 0:  56% 30/54 [00:30<00:24,  1.01s/it]
Validation DataLoader 0:  74% 40/54 [00:41<00:14,  1.03s/it]
Validation DataLoader 0:  93% 50/54 [00:51<00:04,  1.02s/it]
Validation DataLoader 0: 100% 54/54 [00:54<00:00,  1.01s/it]
Epoch 30: 100% 1248/1248 [30:47<00:00,  1.48s/it, v_num=swhj, train/loss_step=0.220, train/loss_pose_step=0.113, train/loss_grip_step=0.106, train/loss_phase_step=0.00139, val/loss=0.162, val/pos_error_m=0.0173, train/loss_epoch=0.049, train/loss_pose_epoch=0.0478, train/loss_grip_epoch=0.00122, train/loss_phase_epoch=5.15e-5, train/phase_acc=1.000] [2025-12-02 12:36:56,859][train_semantic_planner][INFO] - End of epoch 30: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-02 12:39:18,587][train_semantic_planner][INFO] - Failsafe backup for epoch 30 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_030.ckpt.
Epoch 31: 100% 1248/1248 [30:47<00:00,  1.48s/it, v_num=swhj, train/loss_step=0.207, train/loss_pose_step=0.112, train/loss_grip_step=0.0944, train/loss_phase_step=0.000296, val/loss=0.162, val/pos_error_m=0.0173, train/loss_epoch=0.0986, train/loss_pose_epoch=0.0729, train/loss_grip_epoch=0.021, train/loss_phase_epoch=0.00944, train/phase_acc=0.997][2025-12-02 13:10:06,222][train_semantic_planner][INFO] - End of epoch 31: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-02 13:11:59,498][train_semantic_planner][INFO] - Failsafe backup for epoch 31 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_031.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:09<00:42,  1.03it/s]
Validation DataLoader 0:  37% 20/54 [00:20<00:34,  1.02s/it]
Validation DataLoader 0:  56% 30/54 [00:30<00:24,  1.01s/it]
Validation DataLoader 0:  74% 40/54 [00:39<00:13,  1.01it/s]
Validation DataLoader 0:  93% 50/54 [00:47<00:03,  1.05it/s]
Validation DataLoader 0: 100% 54/54 [00:51<00:00,  1.05it/s]
Epoch 32: 100% 1248/1248 [30:48<00:00,  1.48s/it, v_num=swhj, train/loss_step=0.109, train/loss_pose_step=0.106, train/loss_grip_step=0.00259, train/loss_phase_step=0.00058, val/loss=0.159, val/pos_error_m=0.0163, train/loss_epoch=0.0955, train/loss_pose_epoch=0.0711, train/loss_grip_epoch=0.0205, train/loss_phase_epoch=0.00778, train/phase_acc=0.997] [2025-12-02 13:46:07,018][train_semantic_planner][INFO] - End of epoch 32: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-02 13:48:00,644][train_semantic_planner][INFO] - Failsafe backup for epoch 32 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_032.ckpt.
Epoch 33: 100% 1248/1248 [30:48<00:00,  1.48s/it, v_num=swhj, train/loss_step=0.0571, train/loss_pose_step=0.0361, train/loss_grip_step=0.00986, train/loss_phase_step=0.0223, val/loss=0.159, val/pos_error_m=0.0163, train/loss_epoch=0.0947, train/loss_pose_epoch=0.069, train/loss_grip_epoch=0.0217, train/loss_phase_epoch=0.00788, train/phase_acc=0.997][2025-12-02 14:18:49,233][train_semantic_planner][INFO] - End of epoch 33: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-02 14:21:00,663][train_semantic_planner][INFO] - Failsafe backup for epoch 33 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_033.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:09<00:39,  1.11it/s]
Validation DataLoader 0:  37% 20/54 [00:20<00:35,  1.03s/it]
Validation DataLoader 0:  56% 30/54 [00:29<00:23,  1.03it/s]
Validation DataLoader 0:  74% 40/54 [00:37<00:13,  1.06it/s]
Validation DataLoader 0:  93% 50/54 [00:47<00:03,  1.05it/s]
Validation DataLoader 0: 100% 54/54 [00:50<00:00,  1.07it/s]
Epoch 34: 100% 1248/1248 [30:47<00:00,  1.48s/it, v_num=swhj, train/loss_step=0.103, train/loss_pose_step=0.0856, train/loss_grip_step=0.0165, train/loss_phase_step=0.00243, val/loss=0.159, val/pos_error_m=0.0139, train/loss_epoch=0.0911, train/loss_pose_epoch=0.0681, train/loss_grip_epoch=0.0201, train/loss_phase_epoch=0.00582, train/phase_acc=0.998]  [2025-12-02 14:55:19,004][train_semantic_planner][INFO] - End of epoch 34: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-02 14:57:24,272][train_semantic_planner][INFO] - Failsafe backup for epoch 34 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_034.ckpt.
Epoch 35: 100% 1248/1248 [30:49<00:00,  1.48s/it, v_num=swhj, train/loss_step=0.0739, train/loss_pose_step=0.0409, train/loss_grip_step=0.0141, train/loss_phase_step=0.0377, val/loss=0.159, val/pos_error_m=0.0139, train/loss_epoch=0.0855, train/loss_pose_epoch=0.0659, train/loss_grip_epoch=0.0174, train/loss_phase_epoch=0.00445, train/phase_acc=0.998][2025-12-02 15:28:13,845][train_semantic_planner][INFO] - End of epoch 35: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-02 15:30:16,323][train_semantic_planner][INFO] - Failsafe backup for epoch 35 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_035.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:09<00:43,  1.00it/s]
Validation DataLoader 0:  37% 20/54 [00:20<00:34,  1.03s/it]
Validation DataLoader 0:  56% 30/54 [00:31<00:25,  1.05s/it]
Validation DataLoader 0:  74% 40/54 [00:41<00:14,  1.03s/it]
Validation DataLoader 0:  93% 50/54 [00:49<00:03,  1.01it/s]
Validation DataLoader 0: 100% 54/54 [00:52<00:00,  1.03it/s]
Epoch 36:   0% 0/1248 [00:00<?, ?it/s, v_num=swhj, train/loss_step=0.0739, train/loss_pose_step=0.0409, train/loss_grip_step=0.0141, train/loss_phase_step=0.0377, val/loss=0.179, val/pos_error_m=0.0163, train/loss_epoch=0.087, train/loss_pose_epoch=0.0658, train/loss_grip_epoch=0.018, train/loss_phase_epoch=0.00628, train/phase_acc=0.998]


First run with model epoch 22:
[2025-11-28 19:52:15,046][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode
/usr/local/lib/python3.12/dist-packages/torch/utils/data/_utils/pin_memory.py:57: DeprecationWarning: The argument 'device' of Tensor.pin_memory() is deprecated. Please do not pass this argument. (Triggered internally at /pytorch/aten/src/ATen/native/Memory.cpp:46.)
  return data.pin_memory(device)
/usr/local/lib/python3.12/dist-packages/torch/utils/data/_utils/pin_memory.py:57: DeprecationWarning: The argument 'device' of Tensor.is_pinned() is deprecated. Please do not pass this argument. (Triggered internally at /pytorch/aten/src/ATen/native/Memory.cpp:31.)
  return data.pin_memory(device)
Validating: 100% 108/108 [02:50<00:00,  1.58s/it]
[2025-11-28 19:55:03,868][__main__][INFO] - Calculating Aggregate Statistics...


lidation: |          | 0/? [00:00<?, ?it/s][2025-12-02 17:49:26,937][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode
[2025-12-02 17:49:26,938][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode

Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:09<00:42,  1.03it/s]
Validation DataLoader 0:  37% 20/54 [00:18<00:31,  1.09it/s]
Validation DataLoader 0:  56% 30/54 [00:29<00:23,  1.02it/s]
Validation DataLoader 0:  74% 40/54 [00:39<00:13,  1.02it/s]
Validation DataLoader 0:  93% 50/54 [00:47<00:03,  1.05it/s]
Validation DataLoader 0: 100% 54/54 [00:51<00:00,  1.05it/s]
Epoch 36: 100% 1248/1248 [29:06<00:00,  1.40s/it, v_num=oxut, train/loss_step=0.0401, train/loss_pose_step=0.0368, train/loss_grip_step=0.00322, train/loss_phase_step=3.28e-5, val/loss=0.178, val/pos_error_m=0.015, train/loss_epoch=0.0404, train/loss_pose_epoch=0.0377, train/loss_grip_epoch=0.0027, train/loss_phase_epoch=1.28e-5, train/phase_acc=1.000][2025-12-02 18:20:59,366][train_semantic_planner][INFO] - End of epoch 36: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-02 18:22:40,639][train_semantic_planner][INFO] - Failsafe backup for epoch 36 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_036.ckpt.
Epoch 37: 100% 1248/1248 [29:07<00:00,  1.40s/it, v_num=oxut, train/loss_step=0.0782, train/loss_pose_step=0.0638, train/loss_grip_step=0.0133, train/loss_phase_step=0.00222, val/loss=0.178, val/pos_error_m=0.015, train/loss_epoch=0.0873, train/loss_pose_epoch=0.0646, train/loss_grip_epoch=0.0193, train/loss_phase_epoch=0.00668, train/phase_acc=0.998][2025-12-02 18:51:48,173][train_semantic_planner][INFO] - End of epoch 37: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-02 18:53:37,856][train_semantic_planner][INFO] - Failsafe backup for epoch 37 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_037.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:07<00:34,  1.28it/s]
Validation DataLoader 0:  37% 20/54 [00:19<00:32,  1.04it/s]
Validation DataLoader 0:  56% 30/54 [00:30<00:24,  1.00s/it]
Validation DataLoader 0:  74% 40/54 [00:38<00:13,  1.03it/s]
Validation DataLoader 0:  93% 50/54 [00:48<00:03,  1.04it/s]
Validation DataLoader 0: 100% 54/54 [00:51<00:00,  1.06it/s]
Epoch 38: 100% 1248/1248 [29:07<00:00,  1.40s/it, v_num=oxut, train/loss_step=0.235, train/loss_pose_step=0.216, train/loss_grip_step=0.0181, train/loss_phase_step=0.000409, val/loss=0.182, val/pos_error_m=0.013, train/loss_epoch=0.0842, train/loss_pose_epoch=0.0624, train/loss_grip_epoch=0.0185, train/loss_phase_epoch=0.00647, train/phase_acc=0.998] [2025-12-02 19:25:45,407][train_semantic_planner][INFO] - End of epoch 38: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-02 19:28:04,127][train_semantic_planner][INFO] - Failsafe backup for epoch 38 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_038.ckpt.
Epoch 39: 100% 1248/1248 [29:08<00:00,  1.40s/it, v_num=oxut, train/loss_step=0.0733, train/loss_pose_step=0.0661, train/loss_grip_step=0.00691, train/loss_phase_step=0.000718, val/loss=0.182, val/pos_error_m=0.013, train/loss_epoch=0.0767, train/loss_pose_epoch=0.0601, train/loss_grip_epoch=0.015, train/loss_phase_epoch=0.00316, train/phase_acc=0.999][2025-12-02 19:57:12,199][train_semantic_planner][INFO] - End of epoch 39: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-02 19:59:05,859][train_semantic_planner][INFO] - Failsafe backup for epoch 39 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_039.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:09<00:41,  1.05it/s]
Validation DataLoader 0:  37% 20/54 [00:18<00:31,  1.08it/s]
Validation DataLoader 0:  56% 30/54 [00:28<00:23,  1.04it/s]
Validation DataLoader 0:  74% 40/54 [00:37<00:13,  1.06it/s]
Validation DataLoader 0:  93% 50/54 [00:46<00:03,  1.08it/s]
Validation DataLoader 0: 100% 54/54 [00:49<00:00,  1.09it/s]
Epoch 40: 100% 1248/1248 [29:07<00:00,  1.40s/it, v_num=oxut, train/loss_step=0.115, train/loss_pose_step=0.0715, train/loss_grip_step=0.0436, train/loss_phase_step=2.4e-6, val/loss=0.191, val/pos_error_m=0.0146, train/loss_epoch=0.0785, train/loss_pose_epoch=0.0593, train/loss_grip_epoch=0.0173, train/loss_phase_epoch=0.00381, train/phase_acc=0.999]   [2025-12-02 20:31:31,583][train_semantic_planner][INFO] - End of epoch 40: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-02 20:33:21,885][train_semantic_planner][INFO] - Failsafe backup for epoch 40 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_040.ckpt.
Epoch 41: 100% 1248/1248 [29:08<00:00,  1.40s/it, v_num=oxut, train/loss_step=0.0914, train/loss_pose_step=0.0437, train/loss_grip_step=0.0477, train/loss_phase_step=2.69e-6, val/loss=0.191, val/pos_error_m=0.0146, train/loss_epoch=0.0804, train/loss_pose_epoch=0.0599, train/loss_grip_epoch=0.0182, train/loss_phase_epoch=0.00462, train/phase_acc=0.998]  [2025-12-02 21:02:30,469][train_semantic_planner][INFO] - End of epoch 41: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-02 21:05:05,083][train_semantic_planner][INFO] - Failsafe backup for epoch 41 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_041.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:08<00:37,  1.19it/s]
Validation DataLoader 0:  37% 20/54 [00:19<00:32,  1.03it/s]
Validation DataLoader 0:  56% 30/54 [00:27<00:22,  1.07it/s]
Validation DataLoader 0:  74% 40/54 [00:37<00:13,  1.07it/s]
Validation DataLoader 0:  93% 50/54 [00:46<00:03,  1.08it/s]
Validation DataLoader 0: 100% 54/54 [00:49<00:00,  1.09it/s]
Epoch 42: 100% 1248/1248 [29:09<00:00,  1.40s/it, v_num=oxut, train/loss_step=0.179, train/loss_pose_step=0.166, train/loss_grip_step=0.00927, train/loss_phase_step=0.00688, val/loss=0.208, val/pos_error_m=0.0129, train/loss_epoch=0.0723, train/loss_pose_epoch=0.0565, train/loss_grip_epoch=0.0143, train/loss_phase_epoch=0.00284, train/phase_acc=0.999] [2025-12-02 21:38:29,848][train_semantic_planner][INFO] - End of epoch 42: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-02 21:40:36,834][train_semantic_planner][INFO] - Failsafe backup for epoch 42 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_042.ckpt.
Epoch 43:   3% 40/1248 [00:59<29:46,  1.48s/it, v_num=oxut, train/loss_step=0.105, train/loss_pose_step=0.0952, train/loss_grip_step=0.00939, train/loss_phase_step=0.000697, val/loss=0.208, val/pos_error_m=0.0129, train/loss_epoch=0.0672, train/loss_pose_epoch=0.0539, train/loss_grip_epoch=0.0121, train/loss_phase_epoch=0.00239, train/phase_acc=0.999] 

25-12-03 12:03:01,631][train_semantic_planner][INFO] - Failsafe backup for epoch 42 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_042.ckpt.
Epoch 43: 100% 1248/1248 [30:04<00:00,  1.45s/it, v_num=kh57, train/loss_step=0.0927, train/loss_pose_step=0.0366, train/loss_grip_step=0.0561, train/loss_phase_step=0.000128, train/loss_epoch=0.0275, train/loss_pose_epoch=0.0264, train/loss_grip_epoch=0.000831, train/loss_phase_epoch=0.000471, train/phase_acc=1.000][2025-12-03 12:33:05,770][train_semantic_planner][INFO] - End of epoch 43: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-03 12:33:24,716][train_semantic_planner][INFO] - Failsafe backup for epoch 43 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_043.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s][2025-12-03 12:33:25,996][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode
[2025-12-03 12:33:25,996][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode

Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:10<00:46,  1.05s/it]
Validation DataLoader 0:  37% 20/54 [00:22<00:37,  1.11s/it]
Validation DataLoader 0:  56% 30/54 [00:32<00:25,  1.07s/it]
Validation DataLoader 0:  74% 40/54 [00:41<00:14,  1.03s/it]
Validation DataLoader 0:  93% 50/54 [00:51<00:04,  1.03s/it]
Validation DataLoader 0: 100% 54/54 [00:54<00:00,  1.01s/it]
Epoch 44: 100% 1248/1248 [30:07<00:00,  1.45s/it, v_num=kh57, train/loss_step=0.0241, train/loss_pose_step=0.0191, train/loss_grip_step=0.00449, train/loss_phase_step=0.001, train/loss_epoch=0.0656, train/loss_pose_epoch=0.0536, train/loss_grip_epoch=0.0105, train/loss_phase_epoch=0.00311, train/phase_acc=0.999, val/loss=0.210, val/pos_error_m=0.013][2025-12-03 13:05:55,694][train_semantic_planner][INFO] - End of epoch 44: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-03 13:07:34,738][train_semantic_planner][INFO] - Failsafe backup for epoch 44 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_044.ckpt.
Epoch 45: 100% 1248/1248 [30:07<00:00,  1.45s/it, v_num=kh57, train/loss_step=0.0319, train/loss_pose_step=0.0249, train/loss_grip_step=0.00706, train/loss_phase_step=5.76e-6, train/loss_epoch=0.064, train/loss_pose_epoch=0.0528, train/loss_grip_epoch=0.0102, train/loss_phase_epoch=0.00211, train/phase_acc=0.999, val/loss=0.210, val/pos_error_m=0.013][2025-12-03 13:37:42,223][train_semantic_planner][INFO] - End of epoch 45: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-03 13:39:23,790][train_semantic_planner][INFO] - Failsafe backup for epoch 45 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_045.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:08<00:35,  1.23it/s]
Validation DataLoader 0:  37% 20/54 [00:20<00:34,  1.01s/it]
Validation DataLoader 0:  56% 30/54 [00:29<00:23,  1.01it/s]
Validation DataLoader 0:  74% 40/54 [00:37<00:13,  1.05it/s]
Validation DataLoader 0:  93% 50/54 [00:47<00:03,  1.06it/s]
Validation DataLoader 0: 100% 54/54 [00:50<00:00,  1.08it/s]
Epoch 46: 100% 1248/1248 [30:06<00:00,  1.45s/it, v_num=kh57, train/loss_step=0.0845, train/loss_pose_step=0.0777, train/loss_grip_step=0.00275, train/loss_phase_step=0.00811, train/loss_epoch=0.064, train/loss_pose_epoch=0.052, train/loss_grip_epoch=0.0105, train/loss_phase_epoch=0.00307, train/phase_acc=0.999, val/loss=0.219, val/pos_error_m=0.015][2025-12-03 14:10:21,900][train_semantic_planner][INFO] - End of epoch 46: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-03 14:12:05,666][train_semantic_planner][INFO] - Failsafe backup for epoch 46 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_046.ckpt.
Epoch 47: 100% 1248/1248 [30:07<00:00,  1.45s/it, v_num=kh57, train/loss_step=0.0809, train/loss_pose_step=0.0768, train/loss_grip_step=0.00404, train/loss_phase_step=0.00011, train/loss_epoch=0.0625, train/loss_pose_epoch=0.0506, train/loss_grip_epoch=0.0105, train/loss_phase_epoch=0.00269, train/phase_acc=0.999, val/loss=0.219, val/pos_error_m=0.015][2025-12-03 14:42:13,429][train_semantic_planner][INFO] - End of epoch 47: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-03 14:44:09,081][train_semantic_planner][INFO] - Failsafe backup for epoch 47 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_047.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:08<00:37,  1.18it/s]
Validation DataLoader 0:  37% 20/54 [00:20<00:34,  1.00s/it]
Validation DataLoader 0:  56% 30/54 [00:28<00:23,  1.04it/s]
Validation DataLoader 0:  74% 40/54 [00:37<00:13,  1.05it/s]
Validation DataLoader 0:  93% 50/54 [00:47<00:03,  1.06it/s]
Validation DataLoader 0: 100% 54/54 [00:50<00:00,  1.08it/s]
Epoch 48: 100% 1248/1248 [30:06<00:00,  1.45s/it, v_num=kh57, train/loss_step=0.0593, train/loss_pose_step=0.0479, train/loss_grip_step=0.0114, train/loss_phase_step=6.12e-5, train/loss_epoch=0.0596, train/loss_pose_epoch=0.0495, train/loss_grip_epoch=0.00924, train/loss_phase_epoch=0.00165, train/phase_acc=0.999, val/loss=0.226, val/pos_error_m=0.0131][2025-12-03 15:15:07,515][train_semantic_planner][INFO] - End of epoch 48: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-03 15:17:04,715][train_semantic_planner][INFO] - Failsafe backup for epoch 48 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_048.ckpt.
Epoch 49:   7% 90/1248 [02:12<28:25,  1.47s/it, v_num=kh57, train/loss_step=0.0462, train/loss_pose_step=0.0432, train/loss_grip_step=0.00295, train/loss_phase_step=2.04e-6, train/loss_epoch=0.0564, train/loss_pose_epoch=0.048, train/loss_grip_epoch=0.00785, train/loss_phase_epoch=0.00127, train/phase_acc=1.000, val/loss=0.226, val/pos_error_m=0.0131]

025-12-03 16:51:02,784][train_semantic_planner][INFO] - End of epoch 48: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-03 16:51:34,394][train_semantic_planner][INFO] - Failsafe backup for epoch 48 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_048.ckpt.
Epoch 49: 100% 1248/1248 [29:57<00:00,  1.44s/it, v_num=fz5r, train/loss_step=0.031, train/loss_pose_step=0.026, train/loss_grip_step=0.00494, train/loss_phase_step=3.77e-6, train/loss_epoch=0.028, train/loss_pose_epoch=0.0272, train/loss_grip_epoch=0.00071, train/loss_phase_epoch=1.15e-5, train/phase_acc=1.000] [2025-12-03 17:21:32,309][train_semantic_planner][INFO] - End of epoch 49: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-03 17:22:29,761][train_semantic_planner][INFO] - Failsafe backup for epoch 49 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_049.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s][2025-12-03 17:22:30,495][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode
[2025-12-03 17:22:30,494][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode

Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:12<00:55,  1.26s/it]
Validation DataLoader 0:  37% 20/54 [00:21<00:35,  1.06s/it]
Validation DataLoader 0:  56% 30/54 [00:31<00:25,  1.05s/it]
Validation DataLoader 0:  74% 40/54 [00:41<00:14,  1.05s/it]
Validation DataLoader 0:  93% 50/54 [00:50<00:04,  1.01s/it]
Validation DataLoader 0: 100% 54/54 [00:54<00:00,  1.01s/it]
Epoch 50: 100% 1248/1248 [30:03<00:00,  1.45s/it, v_num=fz5r, train/loss_step=0.0708, train/loss_pose_step=0.0687, train/loss_grip_step=0.00172, train/loss_phase_step=0.00077, train/loss_epoch=0.0571, train/loss_pose_epoch=0.0485, train/loss_grip_epoch=0.00802, train/loss_phase_epoch=0.000971, train/phase_acc=1.000, val/loss=0.251, val/pos_error_m=0.0124][2025-12-03 17:54:53,381][train_semantic_planner][INFO] - End of epoch 50: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-03 17:55:51,191][train_semantic_planner][INFO] - Failsafe backup for epoch 50 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_050.ckpt.
Epoch 51: 100% 1248/1248 [30:08<00:00,  1.45s/it, v_num=fz5r, train/loss_step=0.148, train/loss_pose_step=0.116, train/loss_grip_step=0.0317, train/loss_phase_step=0.000439, train/loss_epoch=0.0557, train/loss_pose_epoch=0.0466, train/loss_grip_epoch=0.00865, train/loss_phase_epoch=0.000881, train/phase_acc=1.000, val/loss=0.251, val/pos_error_m=0.0124]   [2025-12-03 18:25:59,787][train_semantic_planner][INFO] - End of epoch 51: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-03 18:27:36,796][train_semantic_planner][INFO] - Failsafe backup for epoch 51 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_051.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:10<00:45,  1.04s/it]
Validation DataLoader 0:  37% 20/54 [00:21<00:36,  1.08s/it]
Validation DataLoader 0:  56% 30/54 [00:29<00:23,  1.03it/s]
Validation DataLoader 0:  74% 40/54 [00:38<00:13,  1.04it/s]
Validation DataLoader 0:  93% 50/54 [00:46<00:03,  1.08it/s]
Validation DataLoader 0: 100% 54/54 [00:50<00:00,  1.07it/s]
Epoch 52: 100% 1248/1248 [30:04<00:00,  1.45s/it, v_num=fz5r, train/loss_step=0.00915, train/loss_pose_step=0.00914, train/loss_grip_step=8.94e-6, train/loss_phase_step=2.93e-6, train/loss_epoch=0.0528, train/loss_pose_epoch=0.045, train/loss_grip_epoch=0.00734, train/loss_phase_epoch=0.000966, train/phase_acc=1.000, val/loss=0.255, val/pos_error_m=0.0124][2025-12-03 19:00:12,797][train_semantic_planner][INFO] - End of epoch 52: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-03 19:01:42,256][train_semantic_planner][INFO] - Failsafe backup for epoch 52 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_052.ckpt.
Epoch 53: 100% 1248/1248 [30:07<00:00,  1.45s/it, v_num=fz5r, train/loss_step=0.0154, train/loss_pose_step=0.0149, train/loss_grip_step=0.000487, train/loss_phase_step=8.54e-6, train/loss_epoch=0.0518, train/loss_pose_epoch=0.0454, train/loss_grip_epoch=0.00605, train/loss_phase_epoch=0.000716, train/phase_acc=1.000, val/loss=0.255, val/pos_error_m=0.0124][2025-12-03 19:31:50,144][train_semantic_planner][INFO] - End of epoch 53: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-03 19:33:19,373][train_semantic_planner][INFO] - Failsafe backup for epoch 53 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_053.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:08<00:39,  1.13it/s]
Validation DataLoader 0:  37% 20/54 [00:17<00:29,  1.14it/s]
Validation DataLoader 0:  56% 30/54 [00:27<00:22,  1.09it/s]
Validation DataLoader 0:  74% 40/54 [00:38<00:13,  1.03it/s]
Validation DataLoader 0:  93% 50/54 [00:47<00:03,  1.04it/s]
Validation DataLoader 0: 100% 54/54 [00:52<00:00,  1.03it/s]
Epoch 54: 100% 1248/1248 [31:03<00:00,  1.49s/it, v_num=fz5r, train/loss_step=0.0175, train/loss_pose_step=0.0174, train/loss_grip_step=0.000145, train/loss_phase_step=9.91e-6, train/loss_epoch=0.0484, train/loss_pose_epoch=0.0434, train/loss_grip_epoch=0.00473, train/loss_phase_epoch=0.000523, train/phase_acc=1.000, val/loss=0.273, val/pos_error_m=0.0111][2025-12-03 20:06:57,925][train_semantic_planner][INFO] - End of epoch 54: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-03 20:08:59,139][train_semantic_planner][INFO] - Failsafe backup for epoch 54 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_054.ckpt.
Epoch 55: 100% 1248/1248 [31:17<00:00,  1.50s/it, v_num=fz5r, train/loss_step=0.0221, train/loss_pose_step=0.022, train/loss_grip_step=0.000165, train/loss_phase_step=2.38e-5, train/loss_epoch=0.0477, train/loss_pose_epoch=0.0433, train/loss_grip_epoch=0.00414, train/loss_phase_epoch=0.000437, train/phase_acc=1.000, val/loss=0.273, val/pos_error_m=0.0111][2025-12-03 20:40:16,580][train_semantic_planner][INFO] - End of epoch 55: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-03 20:41:55,210][train_semantic_planner][INFO] - Failsafe backup for epoch 55 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_055.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:10<00:45,  1.03s/it]
Validation DataLoader 0:  37% 20/54 [00:21<00:36,  1.08s/it]
Validation DataLoader 0:  56% 30/54 [00:30<00:24,  1.02s/it]
Validation DataLoader 0:  74% 40/54 [00:40<00:14,  1.00s/it]
Validation DataLoader 0:  93% 50/54 [00:48<00:03,  1.03it/s]
Validation DataLoader 0: 100% 54/54 [00:52<00:00,  1.03it/s]
Epoch 56:   0% 0/1248 [00:00<?, ?it/s, v_num=fz5r, train/loss_step=0.0221, train/loss_pose_step=0.022, train/loss_grip_step=0.000165, train/loss_phase_step=2.38e-5, train/loss_epoch=0.0473, train/loss_pose_epoch=0.0427, train/loss_grip_epoch=0.00442, train/loss_phase_epoch=0.000497, train/phase_acc=1.000, val/loss=0.283, val/pos_error_m=0.0115][rank: 0] Received SIGTERM: 15
sr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/fit_loop.py:534: Found 160 module(s) in eval mode at the start of training. This may lead to unexpected behavior during training. If this is intentional, you can ignore this warning.
Training: |          | 0/? [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/training_epoch_loop.py:224: You're resuming from a checkpoint that ended before the epoch ended and your dataloader is not resumable. This can cause unreliable results if further training is done. Consider using an end-of-epoch checkpoint or make your dataloader resumable by implementing the `state_dict` / `load_state_dict` interface.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/_utils/pin_memory.py:57: DeprecationWarning: The argument 'device' of Tensor.pin_memory() is deprecated. Please do not pass this argument. (Triggered internally at /pytorch/aten/src/ATen/native/Memory.cpp:46.)
  return data.pin_memory(device)
/usr/local/lib/python3.12/dist-packages/torch/utils/data/_utils/pin_memory.py:57: DeprecationWarning: The argument 'device' of Tensor.is_pinned() is deprecated. Please do not pass this argument. (Triggered internally at /pytorch/aten/src/ATen/native/Memory.cpp:31.)
  return data.pin_memory(device)
[2025-12-04 20:56:17,486][train_semantic_planner][INFO] - End of epoch 62: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-04 20:57:47,631][train_semantic_planner][INFO] - Failsafe backup for epoch 62 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_062.ckpt.
Epoch 63: 100% 1248/1248 [29:10<00:00,  1.40s/it, v_num=0x8c, train/loss_step=0.0156, train/loss_pose_step=0.0149, train/loss_grip_step=0.000706, train/loss_phase_step=3.73e-8, train/loss_epoch=0.0227, train/loss_pose_epoch=0.0218, train/loss_grip_epoch=0.000827, train/loss_phase_epoch=1.38e-7, train/phase_acc=1.000][2025-12-04 21:26:58,451][train_semantic_planner][INFO] - End of epoch 63: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-04 21:27:23,697][train_semantic_planner][INFO] - Failsafe backup for epoch 63 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_063.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s][2025-12-04 21:27:25,442][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode
[2025-12-04 21:27:25,442][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode

Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:10<00:44,  1.00s/it]
Validation DataLoader 0:  37% 20/54 [00:21<00:36,  1.09s/it]
Validation DataLoader 0:  56% 30/54 [00:33<00:26,  1.12s/it]
Validation DataLoader 0:  74% 40/54 [00:43<00:15,  1.10s/it]
Validation DataLoader 0:  93% 50/54 [00:52<00:04,  1.05s/it]
Validation DataLoader 0: 100% 54/54 [00:56<00:00,  1.05s/it]
Epoch 64: 100% 1248/1248 [29:12<00:00,  1.40s/it, v_num=0x8c, train/loss_step=0.00395, train/loss_pose_step=0.00326, train/loss_grip_step=0.000653, train/loss_phase_step=6.58e-5, train/loss_epoch=0.0377, train/loss_pose_epoch=0.0359, train/loss_grip_epoch=0.00168, train/loss_phase_epoch=0.000346, train/phase_acc=1.000, val/loss=0.347, val/pos_error_m=0.0109][2025-12-04 21:58:49,422][train_semantic_planner][INFO] - End of epoch 64: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-04 22:00:16,088][train_semantic_planner][INFO] - Failsafe backup for epoch 64 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_064.ckpt.
Epoch 65: 100% 1248/1248 [29:14<00:00,  1.41s/it, v_num=0x8c, train/loss_step=0.0419, train/loss_pose_step=0.0416, train/loss_grip_step=0.000262, train/loss_phase_step=1.68e-7, train/loss_epoch=0.036, train/loss_pose_epoch=0.0348, train/loss_grip_epoch=0.00105, train/loss_phase_epoch=0.0003, train/phase_acc=1.000, val/loss=0.347, val/pos_error_m=0.0109][2025-12-04 22:29:30,141][train_semantic_planner][INFO] - End of epoch 65: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-04 22:31:07,502][train_semantic_planner][INFO] - Failsafe backup for epoch 65 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_065.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:09<00:41,  1.06it/s]
Validation DataLoader 0:  37% 20/54 [00:18<00:31,  1.10it/s]
Validation DataLoader 0:  56% 30/54 [00:28<00:22,  1.06it/s]
Validation DataLoader 0:  74% 40/54 [00:38<00:13,  1.04it/s]
Validation DataLoader 0:  93% 50/54 [00:46<00:03,  1.07it/s]
Validation DataLoader 0: 100% 54/54 [00:49<00:00,  1.08it/s]
Epoch 66: 100% 1248/1248 [29:13<00:00,  1.40s/it, v_num=0x8c, train/loss_step=0.0306, train/loss_pose_step=0.0243, train/loss_grip_step=0.00616, train/loss_phase_step=0.000201, train/loss_epoch=0.0357, train/loss_pose_epoch=0.0345, train/loss_grip_epoch=0.0011, train/loss_phase_epoch=0.000229, train/phase_acc=1.000, val/loss=0.367, val/pos_error_m=0.0118]  [2025-12-04 23:01:12,792][train_semantic_planner][INFO] - End of epoch 66: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-04 23:02:52,056][train_semantic_planner][INFO] - Failsafe backup for epoch 66 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_066.ckpt.
Epoch 67: 100% 1248/1248 [29:15<00:00,  1.41s/it, v_num=0x8c, train/loss_step=0.0151, train/loss_pose_step=0.010, train/loss_grip_step=0.00508, train/loss_phase_step=2.58e-6, train/loss_epoch=0.0356, train/loss_pose_epoch=0.0343, train/loss_grip_epoch=0.00115, train/loss_phase_epoch=0.000318, train/phase_acc=1.000, val/loss=0.367, val/pos_error_m=0.0118]   [2025-12-04 23:32:07,138][train_semantic_planner][INFO] - End of epoch 67: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-04 23:33:59,624][train_semantic_planner][INFO] - Failsafe backup for epoch 67 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_067.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:07<00:34,  1.28it/s]
Validation DataLoader 0:  37% 20/54 [00:19<00:32,  1.04it/s]
Validation DataLoader 0:  56% 30/54 [00:29<00:23,  1.02it/s]
Validation DataLoader 0:  74% 40/54 [00:37<00:13,  1.07it/s]
Validation DataLoader 0:  93% 50/54 [00:47<00:03,  1.06it/s]
Validation DataLoader 0: 100% 54/54 [00:49<00:00,  1.08it/s]
Epoch 68: 100% 1248/1248 [29:16<00:00,  1.41s/it, v_num=0x8c, train/loss_step=0.0125, train/loss_pose_step=0.0124, train/loss_grip_step=6.05e-5, train/loss_phase_step=0.000162, train/loss_epoch=0.0343, train/loss_pose_epoch=0.0333, train/loss_grip_epoch=0.000922, train/loss_phase_epoch=0.000104, train/phase_acc=1.000, val/loss=0.388, val/pos_error_m=0.0109][2025-12-05 00:06:51,682][train_semantic_planner][INFO] - End of epoch 68: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 00:07:57,705][train_semantic_planner][INFO] - Failsafe backup for epoch 68 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_068.ckpt.
Epoch 69: 100% 1248/1248 [29:15<00:00,  1.41s/it, v_num=0x8c, train/loss_step=0.024, train/loss_pose_step=0.0239, train/loss_grip_step=7e-5, train/loss_phase_step=3.45e-7, train/loss_epoch=0.0338, train/loss_pose_epoch=0.0329, train/loss_grip_epoch=0.000801, train/loss_phase_epoch=0.000271, train/phase_acc=1.000, val/loss=0.388, val/pos_error_m=0.0109]    [2025-12-05 00:37:13,304][train_semantic_planner][INFO] - End of epoch 69: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 00:38:47,515][train_semantic_planner][INFO] - Failsafe backup for epoch 69 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_069.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:08<00:38,  1.15it/s]
Validation DataLoader 0:  37% 20/54 [00:19<00:32,  1.03it/s]
Validation DataLoader 0:  56% 30/54 [00:28<00:22,  1.07it/s]
Validation DataLoader 0:  74% 40/54 [00:38<00:13,  1.05it/s]
Validation DataLoader 0:  93% 50/54 [00:47<00:03,  1.06it/s]
Validation DataLoader 0: 100% 54/54 [00:50<00:00,  1.07it/s]
Epoch 70: 100% 1248/1248 [29:13<00:00,  1.40s/it, v_num=0x8c, train/loss_step=0.034, train/loss_pose_step=0.0339, train/loss_grip_step=9.27e-5, train/loss_phase_step=3.28e-7, train/loss_epoch=0.0329, train/loss_pose_epoch=0.0322, train/loss_grip_epoch=0.000641, train/loss_phase_epoch=0.000113, train/phase_acc=1.000, val/loss=0.394, val/pos_error_m=0.0106] [2025-12-05 01:10:56,854][train_semantic_planner][INFO] - End of epoch 70: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 01:12:09,042][train_semantic_planner][INFO] - Failsafe backup for epoch 70 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_070.ckpt.
Epoch 71:  20% 250/1248 [05:53<23:30,  1.41s/it, v_num=0x8c, train/loss_step=0.018, train/loss_pose_step=0.0176, train/loss_grip_step=0.00041, train/loss_phase_step=9.31e-9, train/loss_epoch=0.0324, train/loss_pose_epoch=0.0317, train/loss_grip_epoch=0.000688, train/loss_phase_epoch=5.83e-5, train/phase_acc=1.000, val/loss=0.394, val/pos_error_m=0.0106]

ning: |          | 0/? [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/training_epoch_loop.py:224: You're resuming from a checkpoint that ended before the epoch ended and your dataloader is not resumable. This can cause unreliable results if further training is done. Consider using an end-of-epoch checkpoint or make your dataloader resumable by implementing the `state_dict` / `load_state_dict` interface.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/_utils/pin_memory.py:57: DeprecationWarning: The argument 'device' of Tensor.pin_memory() is deprecated. Please do not pass this argument. (Triggered internally at /pytorch/aten/src/ATen/native/Memory.cpp:46.)
  return data.pin_memory(device)
/usr/local/lib/python3.12/dist-packages/torch/utils/data/_utils/pin_memory.py:57: DeprecationWarning: The argument 'device' of Tensor.is_pinned() is deprecated. Please do not pass this argument. (Triggered internally at /pytorch/aten/src/ATen/native/Memory.cpp:31.)
  return data.pin_memory(device)
[2025-12-05 10:35:06,937][train_semantic_planner][INFO] - End of epoch 70: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 10:36:23,605][train_semantic_planner][INFO] - Failsafe backup for epoch 70 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_070.ckpt.
Epoch 71: 100% 1248/1248 [29:51<00:00,  1.44s/it, v_num=9jsm, train/loss_step=0.00733, train/loss_pose_step=0.00712, train/loss_grip_step=0.000211, train/loss_phase_step=9.22e-7, train/loss_epoch=0.0188, train/loss_pose_epoch=0.0187, train/loss_grip_epoch=3.23e-5, train/loss_phase_epoch=4.66e-7, train/phase_acc=1.000][2025-12-05 11:06:14,712][train_semantic_planner][INFO] - End of epoch 71: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 11:08:14,152][train_semantic_planner][INFO] - Failsafe backup for epoch 71 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_071.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s][2025-12-05 11:08:15,394][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode
[2025-12-05 11:08:15,405][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode

Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:10<00:47,  1.09s/it]
Validation DataLoader 0:  37% 20/54 [00:19<00:33,  1.00it/s]
Validation DataLoader 0:  56% 30/54 [00:30<00:24,  1.01s/it]
Validation DataLoader 0:  74% 40/54 [00:38<00:13,  1.04it/s]
Validation DataLoader 0:  93% 50/54 [00:48<00:03,  1.03it/s]
Validation DataLoader 0: 100% 54/54 [00:51<00:00,  1.05it/s]
Epoch 72: 100% 1248/1248 [29:55<00:00,  1.44s/it, v_num=9jsm, train/loss_step=0.0172, train/loss_pose_step=0.0165, train/loss_grip_step=0.000699, train/loss_phase_step=1.14e-7, train/loss_epoch=0.0321, train/loss_pose_epoch=0.0314, train/loss_grip_epoch=0.000718, train/loss_phase_epoch=5.36e-5, train/phase_acc=1.000, val/loss=0.409, val/pos_error_m=0.0106][2025-12-05 11:42:33,280][train_semantic_planner][INFO] - End of epoch 72: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 11:44:24,849][train_semantic_planner][INFO] - Failsafe backup for epoch 72 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_072.ckpt.
Epoch 73: 100% 1248/1248 [29:55<00:00,  1.44s/it, v_num=9jsm, train/loss_step=0.0204, train/loss_pose_step=0.0202, train/loss_grip_step=0.00021, train/loss_phase_step=1.49e-8, train/loss_epoch=0.0316, train/loss_pose_epoch=0.0309, train/loss_grip_epoch=0.000669, train/loss_phase_epoch=6.25e-5, train/phase_acc=1.000, val/loss=0.409, val/pos_error_m=0.0106][2025-12-05 12:14:20,092][train_semantic_planner][INFO] - End of epoch 73: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 12:16:00,868][train_semantic_planner][INFO] - Failsafe backup for epoch 73 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_073.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:07<00:33,  1.31it/s]
Validation DataLoader 0:  37% 20/54 [00:18<00:31,  1.07it/s]
Validation DataLoader 0:  56% 30/54 [00:28<00:22,  1.06it/s]
Validation DataLoader 0:  74% 40/54 [00:36<00:12,  1.08it/s]
Validation DataLoader 0:  93% 50/54 [00:46<00:03,  1.09it/s]
Validation DataLoader 0: 100% 54/54 [00:48<00:00,  1.11it/s]
Epoch 74: 100% 1248/1248 [30:12<00:00,  1.45s/it, v_num=9jsm, train/loss_step=0.0156, train/loss_pose_step=0.0156, train/loss_grip_step=1.35e-6, train/loss_phase_step=1.75e-7, train/loss_epoch=0.031, train/loss_pose_epoch=0.0304, train/loss_grip_epoch=0.00053, train/loss_phase_epoch=0.00011, train/phase_acc=1.000, val/loss=0.418, val/pos_error_m=0.0104][2025-12-05 12:49:06,512][train_semantic_planner][INFO] - End of epoch 74: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 12:51:02,694][train_semantic_planner][INFO] - Failsafe backup for epoch 74 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_074.ckpt.
Epoch 75: 100% 1248/1248 [30:43<00:00,  1.48s/it, v_num=9jsm, train/loss_step=0.010, train/loss_pose_step=0.0098, train/loss_grip_step=0.000212, train/loss_phase_step=2.42e-8, train/loss_epoch=0.0305, train/loss_pose_epoch=0.0299, train/loss_grip_epoch=0.00052, train/loss_phase_epoch=0.000125, train/phase_acc=1.000, val/loss=0.418, val/pos_error_m=0.0104][2025-12-05 13:21:46,508][train_semantic_planner][INFO] - End of epoch 75: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 13:23:32,278][train_semantic_planner][INFO] - Failsafe backup for epoch 75 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_075.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:08<00:39,  1.13it/s]
Validation DataLoader 0:  37% 20/54 [00:19<00:32,  1.04it/s]
Validation DataLoader 0:  56% 30/54 [00:29<00:23,  1.01it/s]
Validation DataLoader 0:  74% 40/54 [00:39<00:13,  1.03it/s]
Validation DataLoader 0:  93% 50/54 [00:47<00:03,  1.06it/s]
Validation DataLoader 0: 100% 54/54 [00:50<00:00,  1.08it/s]
Epoch 76: 100% 1248/1248 [30:41<00:00,  1.48s/it, v_num=9jsm, train/loss_step=0.0174, train/loss_pose_step=0.0173, train/loss_grip_step=7.97e-5, train/loss_phase_step=7.73e-6, train/loss_epoch=0.030, train/loss_pose_epoch=0.0296, train/loss_grip_epoch=0.000387, train/loss_phase_epoch=4.87e-5, train/phase_acc=1.000, val/loss=0.426, val/pos_error_m=0.0103] [2025-12-05 13:56:59,697][train_semantic_planner][INFO] - End of epoch 76: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 13:58:52,275][train_semantic_planner][INFO] - Failsafe backup for epoch 76 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_076.ckpt.
Epoch 77:   2% 20/1248 [00:32<33:34,  1.64s/it, v_num=9jsm, train/loss_step=0.0257, train/loss_pose_step=0.0257, train/loss_grip_step=1.32e-5, train/loss_phase_step=5.49e-7, train/loss_epoch=0.0297, train/loss_pose_epoch=0.0293, train/loss_grip_epoch=0.000399, train/loss_phase_epoch=5.77e-5, train/phase_acc=1.000, val/loss=0.426, val/pos_error_m=0.0103] Process Process-3:
Process Process-1:
025-12-05 15:29:45,879][train_semantic_planner][INFO] - End of epoch 76: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 15:30:06,537][train_semantic_planner][INFO] - Failsafe backup for epoch 76 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_076.ckpt.
[2025-12-05 15:30:06,545][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_020.ckpt
[2025-12-05 15:30:06,547][train_semantic_planner][INFO] - Cleaned up old failsafe backup: backup_epoch_021.ckpt
Epoch 77: 100% 1248/1248 [31:34<00:00,  1.52s/it, v_num=fdz6, train/loss_step=0.0245, train/loss_pose_step=0.0228, train/loss_grip_step=0.00167, train/loss_phase_step=2.6e-6, train/loss_epoch=0.0176, train/loss_pose_epoch=0.0176, train/loss_grip_epoch=5.2e-6, train/loss_phase_epoch=3.35e-7, train/phase_acc=1.000]  [2025-12-05 16:01:41,463][train_semantic_planner][INFO] - End of epoch 77: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 16:02:28,660][train_semantic_planner][INFO] - Failsafe backup for epoch 77 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_077.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s][2025-12-05 16:02:29,785][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode
[2025-12-05 16:02:29,799][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode

Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:10<00:47,  1.08s/it]
Validation DataLoader 0:  37% 20/54 [00:23<00:39,  1.16s/it]
Validation DataLoader 0:  56% 30/54 [00:34<00:27,  1.16s/it]
Validation DataLoader 0:  74% 40/54 [00:44<00:15,  1.11s/it]
Validation DataLoader 0:  93% 50/54 [00:54<00:04,  1.10s/it]
Validation DataLoader 0: 100% 54/54 [00:58<00:00,  1.08s/it]
Epoch 78: 100% 1248/1248 [31:38<00:00,  1.52s/it, v_num=fdz6, train/loss_step=0.0859, train/loss_pose_step=0.0837, train/loss_grip_step=0.00215, train/loss_phase_step=5.68e-6, train/loss_epoch=0.0293, train/loss_pose_epoch=0.0289, train/loss_grip_epoch=0.000382, train/loss_phase_epoch=7.37e-5, train/phase_acc=1.000, val/loss=0.437, val/pos_error_m=0.0103][2025-12-05 16:36:45,930][train_semantic_planner][INFO] - End of epoch 78: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 16:38:21,786][train_semantic_planner][INFO] - Failsafe backup for epoch 78 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_078.ckpt.
Epoch 79: 100% 1248/1248 [31:45<00:00,  1.53s/it, v_num=fdz6, train/loss_step=0.0107, train/loss_pose_step=0.0106, train/loss_grip_step=7.13e-5, train/loss_phase_step=2.98e-8, train/loss_epoch=0.029, train/loss_pose_epoch=0.0286, train/loss_grip_epoch=0.000334, train/loss_phase_epoch=3.46e-5, train/phase_acc=1.000, val/loss=0.437, val/pos_error_m=0.0103][2025-12-05 17:10:07,489][train_semantic_planner][INFO] - End of epoch 79: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 17:11:34,112][train_semantic_planner][INFO] - Failsafe backup for epoch 79 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_079.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:11<00:51,  1.17s/it]
Validation DataLoader 0:  37% 20/54 [00:24<00:41,  1.23s/it]
Validation DataLoader 0:  56% 30/54 [00:34<00:27,  1.13s/it]
Validation DataLoader 0:  74% 40/54 [00:44<00:15,  1.11s/it]
Validation DataLoader 0:  93% 50/54 [00:54<00:04,  1.09s/it]
Validation DataLoader 0: 100% 54/54 [00:57<00:00,  1.07s/it]
Epoch 80: 100% 1248/1248 [31:58<00:00,  1.54s/it, v_num=fdz6, train/loss_step=0.0423, train/loss_pose_step=0.0422, train/loss_grip_step=0.000102, train/loss_phase_step=2.6e-6, train/loss_epoch=0.0288, train/loss_pose_epoch=0.0284, train/loss_grip_epoch=0.000319, train/loss_phase_epoch=4.69e-5, train/phase_acc=1.000, val/loss=0.444, val/pos_error_m=0.0102] [2025-12-05 17:45:57,805][train_semantic_planner][INFO] - End of epoch 80: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 17:47:51,515][train_semantic_planner][INFO] - Failsafe backup for epoch 80 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_080.ckpt.
Epoch 81: 100% 1248/1248 [32:02<00:00,  1.54s/it, v_num=fdz6, train/loss_step=0.0322, train/loss_pose_step=0.0321, train/loss_grip_step=7.24e-5, train/loss_phase_step=1.9e-6, train/loss_epoch=0.0286, train/loss_pose_epoch=0.0283, train/loss_grip_epoch=0.000321, train/loss_phase_epoch=3.37e-5, train/phase_acc=1.000, val/loss=0.444, val/pos_error_m=0.0102] [2025-12-05 18:19:54,277][train_semantic_planner][INFO] - End of epoch 81: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 18:21:12,740][train_semantic_planner][INFO] - Failsafe backup for epoch 81 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_081.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:09<00:40,  1.08it/s]
Validation DataLoader 0:  37% 20/54 [00:20<00:34,  1.01s/it]
Validation DataLoader 0:  56% 30/54 [00:29<00:23,  1.02it/s]
Validation DataLoader 0:  74% 40/54 [00:40<00:14,  1.02s/it]
Validation DataLoader 0:  93% 50/54 [00:52<00:04,  1.05s/it]
Validation DataLoader 0: 100% 54/54 [00:55<00:00,  1.03s/it]
Epoch 82: 100% 1248/1248 [32:05<00:00,  1.54s/it, v_num=fdz6, train/loss_step=0.0347, train/loss_pose_step=0.0345, train/loss_grip_step=0.000194, train/loss_phase_step=5.33e-7, train/loss_epoch=0.0283, train/loss_pose_epoch=0.0281, train/loss_grip_epoch=0.000264, train/loss_phase_epoch=3.25e-5, train/phase_acc=1.000, val/loss=0.450, val/pos_error_m=0.0102][2025-12-05 18:55:52,012][train_semantic_planner][INFO] - End of epoch 82: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 18:57:15,869][train_semantic_planner][INFO] - Failsafe backup for epoch 82 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_082.ckpt.
Epoch 83:   2% 20/1248 [00:33<34:12,  1.67s/it, v_num=fdz6, train/loss_step=0.0299, train/loss_pose_step=0.0297, train/loss_grip_step=0.000263, train/loss_phase_step=3.61e-6, train/loss_epoch=0.0282, train/loss_pose_epoch=0.0279, train/loss_grip_epoch=0.000249, train/loss_phase_epoch=2.91e-5, train/phase_acc=1.000, val/loss=0.450, val/pos_error_m=0.0102] 

025-12-05 20:12:06,403][train_semantic_planner][INFO] - End of epoch 82: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 20:12:24,730][train_semantic_planner][INFO] - Failsafe backup for epoch 82 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_082.ckpt.
Epoch 83: 100% 1248/1248 [29:11<00:00,  1.40s/it, v_num=kiz0, train/loss_step=0.00471, train/loss_pose_step=0.0047, train/loss_grip_step=1.04e-5, train/loss_phase_step=6.03e-7, train/loss_epoch=0.0186, train/loss_pose_epoch=0.0185, train/loss_grip_epoch=4.01e-5, train/loss_phase_epoch=1.42e-7, train/phase_acc=1.000][2025-12-05 20:41:36,115][train_semantic_planner][INFO] - End of epoch 83: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 20:42:18,712][train_semantic_planner][INFO] - Failsafe backup for epoch 83 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_083.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s][2025-12-05 20:42:20,016][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode
[2025-12-05 20:42:20,016][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode

Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:13<00:57,  1.31s/it]
Validation DataLoader 0:  37% 20/54 [00:25<00:43,  1.27s/it]
Validation DataLoader 0:  56% 30/54 [00:36<00:29,  1.22s/it]
Validation DataLoader 0:  74% 40/54 [00:45<00:15,  1.14s/it]
Validation DataLoader 0:  93% 50/54 [00:56<00:04,  1.13s/it]
Validation DataLoader 0: 100% 54/54 [00:59<00:00,  1.10s/it]
Epoch 84: 100% 1248/1248 [29:11<00:00,  1.40s/it, v_num=kiz0, train/loss_step=0.0225, train/loss_pose_step=0.0222, train/loss_grip_step=0.00025, train/loss_phase_step=3.26e-7, train/loss_epoch=0.028, train/loss_pose_epoch=0.0277, train/loss_grip_epoch=0.000276, train/loss_phase_epoch=3.99e-5, train/phase_acc=1.000, val/loss=0.453, val/pos_error_m=0.0103][2025-12-05 21:12:35,416][train_semantic_planner][INFO] - End of epoch 84: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 21:13:19,446][train_semantic_planner][INFO] - Failsafe backup for epoch 84 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_084.ckpt.
Epoch 85: 100% 1248/1248 [29:13<00:00,  1.40s/it, v_num=kiz0, train/loss_step=0.0124, train/loss_pose_step=0.0118, train/loss_grip_step=0.000615, train/loss_phase_step=1.1e-5, train/loss_epoch=0.0277, train/loss_pose_epoch=0.0275, train/loss_grip_epoch=0.00023, train/loss_phase_epoch=4.07e-5, train/phase_acc=1.000, val/loss=0.453, val/pos_error_m=0.0103]   [2025-12-05 21:42:32,539][train_semantic_planner][INFO] - End of epoch 85: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 21:43:51,101][train_semantic_planner][INFO] - Failsafe backup for epoch 85 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_085.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:10<00:45,  1.03s/it]
Validation DataLoader 0:  37% 20/54 [00:23<00:40,  1.18s/it]
Validation DataLoader 0:  56% 30/54 [00:31<00:25,  1.04s/it]
Validation DataLoader 0:  74% 40/54 [00:40<00:14,  1.02s/it]
Validation DataLoader 0:  93% 50/54 [00:49<00:03,  1.01it/s]
Validation DataLoader 0: 100% 54/54 [00:53<00:00,  1.01it/s]
Epoch 86: 100% 1248/1248 [29:12<00:00,  1.40s/it, v_num=kiz0, train/loss_step=0.0128, train/loss_pose_step=0.0127, train/loss_grip_step=0.000107, train/loss_phase_step=4.37e-6, train/loss_epoch=0.0276, train/loss_pose_epoch=0.0274, train/loss_grip_epoch=0.000227, train/loss_phase_epoch=3.45e-5, train/phase_acc=1.000, val/loss=0.455, val/pos_error_m=0.0101][2025-12-05 22:16:07,341][train_semantic_planner][INFO] - End of epoch 86: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 22:17:42,671][train_semantic_planner][INFO] - Failsafe backup for epoch 86 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_086.ckpt.
Epoch 87: 100% 1248/1248 [29:12<00:00,  1.40s/it, v_num=kiz0, train/loss_step=0.0495, train/loss_pose_step=0.0493, train/loss_grip_step=0.000198, train/loss_phase_step=3.08e-5, train/loss_epoch=0.0276, train/loss_pose_epoch=0.0274, train/loss_grip_epoch=0.00023, train/loss_phase_epoch=4.17e-5, train/phase_acc=1.000, val/loss=0.455, val/pos_error_m=0.0101][2025-12-05 22:46:54,963][train_semantic_planner][INFO] - End of epoch 87: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 22:48:42,926][train_semantic_planner][INFO] - Failsafe backup for epoch 87 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_087.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:07<00:34,  1.28it/s]
Validation DataLoader 0:  37% 20/54 [00:20<00:35,  1.03s/it]
Validation DataLoader 0:  56% 30/54 [00:30<00:24,  1.02s/it]
Validation DataLoader 0:  74% 40/54 [00:38<00:13,  1.03it/s]
Validation DataLoader 0:  93% 50/54 [00:49<00:03,  1.02it/s]
Validation DataLoader 0: 100% 54/54 [00:51<00:00,  1.04it/s]
Epoch 88: 100% 1248/1248 [29:12<00:00,  1.40s/it, v_num=kiz0, train/loss_step=0.0194, train/loss_pose_step=0.0193, train/loss_grip_step=0.000108, train/loss_phase_step=1.92e-7, train/loss_epoch=0.0275, train/loss_pose_epoch=0.0273, train/loss_grip_epoch=0.000215, train/loss_phase_epoch=5.01e-5, train/phase_acc=1.000, val/loss=0.457, val/pos_error_m=0.0102][2025-12-05 23:20:40,792][train_semantic_planner][INFO] - End of epoch 88: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 23:22:09,535][train_semantic_planner][INFO] - Failsafe backup for epoch 88 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_088.ckpt.
Epoch 89: 100% 1248/1248 [29:12<00:00,  1.40s/it, v_num=kiz0, train/loss_step=0.0283, train/loss_pose_step=0.0282, train/loss_grip_step=3.57e-5, train/loss_phase_step=7.26e-5, train/loss_epoch=0.0275, train/loss_pose_epoch=0.0273, train/loss_grip_epoch=0.000216, train/loss_phase_epoch=3.52e-5, train/phase_acc=1.000, val/loss=0.457, val/pos_error_m=0.0102][2025-12-05 23:51:22,155][train_semantic_planner][INFO] - End of epoch 89: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-05 23:52:31,993][train_semantic_planner][INFO] - Failsafe backup for epoch 89 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_089.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:08<00:37,  1.17it/s]
Validation DataLoader 0:  37% 20/54 [00:19<00:32,  1.03it/s]
Validation DataLoader 0:  56% 30/54 [00:28<00:23,  1.04it/s]
Validation DataLoader 0:  74% 40/54 [00:37<00:13,  1.06it/s]
Validation DataLoader 0:  93% 50/54 [00:45<00:03,  1.09it/s]
Validation DataLoader 0: 100% 54/54 [00:48<00:00,  1.11it/s]
Epoch 90:  29% 

ining: |          | 0/? [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/training_epoch_loop.py:224: You're resuming from a checkpoint that ended before the epoch ended and your dataloader is not resumable. This can cause unreliable results if further training is done. Consider using an end-of-epoch checkpoint or make your dataloader resumable by implementing the `state_dict` / `load_state_dict` interface.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/_utils/pin_memory.py:57: DeprecationWarning: The argument 'device' of Tensor.pin_memory() is deprecated. Please do not pass this argument. (Triggered internally at /pytorch/aten/src/ATen/native/Memory.cpp:46.)
  return data.pin_memory(device)
/usr/local/lib/python3.12/dist-packages/torch/utils/data/_utils/pin_memory.py:57: DeprecationWarning: The argument 'device' of Tensor.is_pinned() is deprecated. Please do not pass this argument. (Triggered internally at /pytorch/aten/src/ATen/native/Memory.cpp:31.)
  return data.pin_memory(device)
Training: |          | 1930/? [15:19<00:00,  2.10it/s, v_num=d6w3, train/loss_step=0.110, train/loss_pose_step=0.0361, train/loss_grip_step=0.0384, train/loss_phase_step=0.0715][2025-12-09 13:28:45,934][train_semantic_planner][INFO] - End of epoch 96: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-09 13:29:19,880][train_semantic_planner][INFO] - Failsafe backup for epoch 96 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_096.ckpt.
Epoch 97: 100% 1939/1939 [46:13<00:00,  1.43s/it, v_num=d6w3, train/loss_step=0.116, train/loss_pose_step=0.0587, train/loss_grip_step=0.0358, train/loss_phase_step=0.0437, train/loss_epoch=0.433, train/loss_pose_epoch=0.0711, train/loss_grip_epoch=0.298, train/loss_phase_epoch=0.129, train/phase_acc=0.969][2025-12-09 14:15:33,088][train_semantic_planner][INFO] - End of epoch 97: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-09 14:16:31,559][train_semantic_planner][INFO] - Failsafe backup for epoch 97 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_097.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s][2025-12-09 14:16:32,788][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode
[2025-12-09 14:16:32,789][lmdb_utils][INFO] -  Opened LMDB at /content/drive/MyDrive/pda/data/validation/training_set.lmdb | mode=RO | file-mode

Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:11<00:50,  1.15s/it]
Validation DataLoader 0:  37% 20/54 [00:20<00:34,  1.02s/it]
Validation DataLoader 0:  56% 30/54 [00:30<00:24,  1.03s/it]
Validation DataLoader 0:  74% 40/54 [00:40<00:14,  1.01s/it]
Validation DataLoader 0:  93% 50/54 [00:49<00:03,  1.01it/s]
Validation DataLoader 0: 100% 54/54 [00:52<00:00,  1.03it/s]
Epoch 98: 100% 1939/1939 [45:56<00:00,  1.42s/it, v_num=d6w3, train/loss_step=0.0386, train/loss_pose_step=0.0132, train/loss_grip_step=0.0204, train/loss_phase_step=0.0098, train/loss_epoch=0.236, train/loss_pose_epoch=0.0614, train/loss_grip_epoch=0.139, train/loss_phase_epoch=0.0725, train/phase_acc=0.975, val/loss=0.0987, val/pos_error_m=0.0106][2025-12-09 15:03:44,881][train_semantic_planner][INFO] - End of epoch 98: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-09 15:04:20,270][train_semantic_planner][INFO] - Failsafe backup for epoch 98 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_098.ckpt.
Epoch 99: 100% 1939/1939 [46:16<00:00,  1.43s/it, v_num=d6w3, train/loss_step=0.0298, train/loss_pose_step=0.0214, train/loss_grip_step=0.00423, train/loss_phase_step=0.00837, train/loss_epoch=0.192, train/loss_pose_epoch=0.0563, train/loss_grip_epoch=0.110, train/loss_phase_epoch=0.0514, train/phase_acc=0.981, val/loss=0.0987, val/pos_error_m=0.0106][2025-12-09 15:50:36,786][train_semantic_planner][INFO] - End of epoch 99: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-09 15:52:12,017][train_semantic_planner][INFO] - Failsafe backup for epoch 99 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_099.ckpt.

Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/54 [00:00<?, ?it/s]
Validation DataLoader 0:  19% 10/54 [00:08<00:35,  1.23it/s]
Validation DataLoader 0:  37% 20/54 [00:18<00:31,  1.07it/s]
Validation DataLoader 0:  56% 30/54 [00:29<00:23,  1.02it/s]
Validation DataLoader 0:  74% 40/54 [00:37<00:13,  1.07it/s]
Validation DataLoader 0:  93% 50/54 [00:46<00:03,  1.07it/s]
Validation DataLoader 0: 100% 54/54 [00:49<00:00,  1.08it/s]
Epoch 100: 100% 1939/1939 [46:42<00:00,  1.45s/it, v_num=d6w3, train/loss_step=0.550, train/loss_pose_step=0.0956, train/loss_grip_step=0.445, train/loss_phase_step=0.0192, train/loss_epoch=0.168, train/loss_pose_epoch=0.0551, train/loss_grip_epoch=0.0932, train/loss_phase_epoch=0.0392, train/phase_acc=0.986, val/loss=0.0921, val/pos_error_m=0.0108] [2025-12-09 16:39:45,517][train_semantic_planner][INFO] - End of epoch 100: Triggering atomic failsafe backup...
`weights_only` was not set, defaulting to `False`.
[2025-12-09 16:41:27,338][train_semantic_planner][INFO] - Failsafe backup for epoch 100 saved successfully to /content/drive/MyDrive/pda/models/backups/awr/awr_backup_epoch_100.ckpt.
Epoch 101:   3% 50/1939 [01:14<46:58,  1.49s/it, v_num=d6w3, train/loss_step=0.294, train/loss_pose_step=0.0716, train/loss_grip_step=0.204, train/loss_phase_step=0.0365, train/loss_epoch=0.141, train/loss_pose_epoch=0.0521, train/loss_grip_epoch=0.0736, train/loss_phase_epoch=0.0297, train/phase_acc=0.989, val/loss=0.0921, val/pos_error_m=0.0108]   
Detected KeyboardInterrupt, attempting graceful shutdown ...



============================================================
   AWSP v9.0 VALIDATION REPORT CARD   
============================================================
                    Metric      Value
     Traj Pos Error (Mean)    2.42 cm
Next-Step Pos Error (Mean)    2.52 cm
 Next-Step Pos Error (99%)    5.87 cm
     Traj Rot Error (Mean)   3.25 deg
Next-Step Rot Error (Mean)   3.36 deg
                Gripper F1     0.9708
         Gripper Precision     0.9607
            Gripper Recall     0.9811
          Avg Latency (ms) 1132.96 ms
============================================================

>>> AUTOMATED DIAGNOSIS:
❌ CAUTION. Precision (2.52cm) exceeds 2cm threshold.
   Recommendation: Train longer or check dataset quality.

The second run for epoch 22 model:::
============================================================
   AWSP v9.0 VALIDATION REPORT CARD   
============================================================
                    Metric      Value
     Traj Pos Error (Mean)    2.42 cm
Next-Step Pos Error (Mean)    2.52 cm
 Next-Step Pos Error (99%)    5.87 cm
     Traj Rot Error (Mean)   3.25 deg
Next-Step Rot Error (Mean)   3.36 deg
                Gripper F1     0.9708
         Gripper Precision     0.9607
            Gripper Recall     0.9811
          Avg Latency (ms) 1143.44 ms
============================================================

>>> AUTOMATED DIAGNOSIS:
❌ CAUTION. Precision (2.52cm) exceeds 2cm threshold.
   Recommendation: Train longer or check dataset quality.



===============
===============

First eval run for epoch 23 model:

============================================================
   AWSP v9.0 VALIDATION REPORT CARD   
============================================================
                    Metric      Value
     Traj Pos Error (Mean)    2.23 cm
Next-Step Pos Error (Mean)    2.05 cm
 Next-Step Pos Error (99%)    5.64 cm
     Traj Rot Error (Mean)   2.96 deg
Next-Step Rot Error (Mean)   3.01 deg
                Gripper F1     0.9722
         Gripper Precision     0.9661
            Gripper Recall     0.9783
          Avg Latency (ms) 1135.54 ms
============================================================

>>> AUTOMATED DIAGNOSIS:
❌ CAUTION. Precision (2.05cm) exceeds 2cm threshold.
   Recommendation: Train longer or check dataset quality.
-===========
=============

for model 21::::::::::::::
============================================================
   AWSP v9.0 VALIDATION REPORT CARD   
============================================================
                    Metric      Value
     Traj Pos Error (Mean)    1.79 cm
Next-Step Pos Error (Mean)    1.66 cm
 Next-Step Pos Error (99%)    4.72 cm
     Traj Rot Error (Mean)   3.26 deg
Next-Step Rot Error (Mean)   3.28 deg
                Gripper F1     0.9699
         Gripper Precision     0.9549
            Gripper Recall     0.9853
          Avg Latency (ms) 1127.56 ms
============================================================

>>> AUTOMATED DIAGNOSIS:
✅ PASS. Precision (1.66cm) is within 2cm tolerance.
   Recommendation: Proceed to Simulation / Real Robot deployment.
================
=========================

For mode 20::::::

return data.pin_memory(device)
/usr/local/lib/python3.12/dist-packages/torch/utils/data/_utils/pin_memory.py:57: DeprecationWarning: The argument 'device' of Tensor.is_pinned() is deprecated. Please do not pass this argument. (Triggered internally at /pytorch/aten/src/ATen/native/Memory.cpp:31.)
  return data.pin_memory(device)
Validating: 100% 108/108 [02:46<00:00,  1.55s/it]
[2025-11-28 20:10:23,988][__main__][INFO] - Calculating Aggregate Statistics...

============================================================
   AWSP v9.0 VALIDATION REPORT CARD   
============================================================
                    Metric      Value
     Traj Pos Error (Mean)    1.82 cm
Next-Step Pos Error (Mean)    1.72 cm
 Next-Step Pos Error (99%)    4.60 cm
     Traj Rot Error (Mean)   3.41 deg
Next-Step Rot Error (Mean)   3.44 deg
                Gripper F1     0.9722
         Gripper Precision     0.9625
            Gripper Recall     0.9821
          Avg Latency (ms) 1130.76 ms
============================================================

>>> AUTOMATED DIAGNOSIS:
✅ PASS. Precision (1.72cm) is within 2cm tolerance.
   Recommendation: Proceed to Simulation / Real Robot deployment.




============
=============

First eval run for epoch 28 model:::::::
=================================================
PHASE 1: ACTION CHUNKING SANITY REPORT
==================================================
                     Metric      Value
Trajectory Pos Error (Mean)    2.77 cm
Trajectory Rot Error (Mean)   2.97 deg
 Next-Step Pos Error (Mean)    2.30 cm
 Next-Step Rot Error (Mean)   3.03 deg
Next-Step Pos Error (99th%)    6.18 cm
           Gripper F1-Score     0.9097
          Gripper Precision     0.8954
             Gripper Recall     0.9244
          Inference Latency 1123.64 ms
==================================================
Gripper Confusion Matrix:
[[13765   895]
 [  627  7663]]
==================================================

>>> AUTOMATED DIAGNOSIS:
[FAIL] Model is undertrained. DO NOT run simulation yet.
 - Position Error too high (2.30cm). Need < 2.0cm for reliable grasping.



 for epoch 29
 ==================================================
PHASE 1: ACTION CHUNKING SANITY REPORT
==================================================
                     Metric      Value
Trajectory Pos Error (Mean)    2.79 cm
Trajectory Rot Error (Mean)   3.06 deg
 Next-Step Pos Error (Mean)    2.25 cm
 Next-Step Rot Error (Mean)   2.97 deg
Next-Step Pos Error (99th%)    6.44 cm
           Gripper F1-Score     0.9077
          Gripper Precision     0.8940
             Gripper Recall     0.9218
          Inference Latency 1121.73 ms
==================================================
Gripper Confusion Matrix:
[[13754   906]
 [  648  7642]]
==================================================

>>> AUTOMATED DIAGNOSIS:
[FAIL] Model is undertrained. DO NOT run simulation yet.
 - Position Error too high (2.25cm). Need < 2.0cm for reliable grasping.


for epoch 34::
alidating: 100% 108/108 [02:54<00:00,  1.61s/it]
[2025-12-02 15:37:23,913][__main__][INFO] - Calculating Aggregate Statistics...

============================================================
   AWSP v9.0 VALIDATION REPORT CARD   
============================================================
                    Metric      Value
     Traj Pos Error (Mean)    1.53 cm
Next-Step Pos Error (Mean)    1.31 cm
 Next-Step Pos Error (99%)    3.86 cm
     Traj Rot Error (Mean)   2.16 deg
Next-Step Rot Error (Mean)   2.29 deg
                Gripper F1     0.9716
         Gripper Precision     0.9667
            Gripper Recall     0.9767
          Avg Latency (ms) 1197.92 ms
============================================================

>>> AUTOMATED DIAGNOSIS:
✅ PASS. Precision (1.31cm) is within 2cm tolerance.
   Recommendation: Proceed to Simulation / Real Robot deployment.


for epoch 33
============================================================
   AWSP v9.0 VALIDATION REPORT CARD   
============================================================
                    Metric      Value
     Traj Pos Error (Mean)    1.56 cm
Next-Step Pos Error (Mean)    1.38 cm
 Next-Step Pos Error (99%)    4.08 cm
     Traj Rot Error (Mean)   2.08 deg
Next-Step Rot Error (Mean)   2.15 deg
                Gripper F1     0.9708
         Gripper Precision     0.9639
            Gripper Recall     0.9778
          Avg Latency (ms) 1181.32 ms
============================================================

>>> AUTOMATED DIAGNOSIS:
✅ PASS. Precision (1.38cm) is within 2cm tolerance.
   Recommendation: Proceed to Simulation / Real Robot deployment.


   for epoch 30:

  ============================================================
   AWSP v9.0 VALIDATION REPORT CARD   
============================================================
                    Metric      Value
     Traj Pos Error (Mean)    1.62 cm
Next-Step Pos Error (Mean)    1.51 cm
 Next-Step Pos Error (99%)    3.92 cm
     Traj Rot Error (Mean)   2.43 deg
Next-Step Rot Error (Mean)   2.53 deg
                Gripper F1     0.9719
         Gripper Precision     0.9622
            Gripper Recall     0.9819
          Avg Latency (ms) 1181.88 ms
============================================================

>>> AUTOMATED DIAGNOSIS:
✅ PASS. Precision (1.51cm) is within 2cm tolerance.
   Recommendation: Proceed to Simulation / Real Robot deployment.
