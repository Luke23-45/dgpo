ation DataLoader 0:  65% 40/62 [01:18<00:43,  1.96s/it]
Validation DataLoader 0:  81% 50/62 [01:37<00:23,  1.95s/it]
Validation DataLoader 0:  97% 60/62 [01:57<00:03,  1.96s/it]
Validation DataLoader 0: 100% 62/62 [02:01<00:00,  1.96s/it]
Epoch 0: 100% 638/638 [24:00<00:00,  2.26s/it, v_num=skpv, train/loss_step=0.971, train/pose_step=0.534, train/grip_step=0.437, train/reg_step=0.00252, val/loss=0.554, val/pos_error_m=0.151, train/loss_epoch=1.870, train/pose_epoch=0.765, train/grip_epoch=1.100, train/reg_epoch=0.00167, train/router_phase_acc=0.912][2025-12-24 03:53:36,587][train_moe_post][INFO] - ========================================
[2025-12-24 03:53:36,591][train_moe_post][INFO] - Epoch 0 Expert Statistics:
[2025-12-24 03:53:36,592][train_moe_post][INFO] -   Expert 0 (Approach): Avg Loss = 0.1261 (637 samples)
[2025-12-24 03:53:36,594][train_moe_post][INFO] -   Expert 1 (Grasp): Avg Loss = 0.1299 (602 samples)
[2025-12-24 03:53:36,596][train_moe_post][INFO] -   Expert 2 (Lift): Avg Loss = 0.1250 (638 samples)
[2025-12-24 03:53:36,597][train_moe_post][INFO] -   Expert 3 (Place): Avg Loss = 0.1291 (636 samples)
[2025-12-24 03:53:36,599][train_moe_post][INFO] -   Expert 4 (Retract): Avg Loss = 0.1307 (595 samples)
[2025-12-24 03:53:54,022][train_moe_post][INFO] - [Backup] New Best Model Copied (Val Loss: 0.5537)
[2025-12-24 03:53:54,030][train_moe_post][INFO] -   Best Val Loss so far: 0.5537
[2025-12-24 03:53:54,032][train_moe_post][INFO] - ========================================
[2025-12-24 03:53:55,295][train_moe_post][INFO] - [Metrics] Saved training logs to /content/drive/MyDrive/pda/logs/moe_post/training_metrics_moe_post_v1_20251224_032823.csv
Epoch 1: 100% 638/638 [22:06<00:00,  2.08s/it, v_num=skpv, train/loss_step=0.657, train/pose_step=0.456, train/grip_step=0.200, train/reg_step=0.00205, val/loss=0.554, val/pos_error_m=0.151, train/loss_epoch=1.870, train/pose_epoch=0.765, train/grip_epoch=1.100, train/reg_epoch=0.00167, train/router_phase_acc=0.912]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/62 [00:00<?, ?it/s]
Validation DataLoader 0:  16% 10/62 [00:19<01:41,  1.96s/it]
Validation DataLoader 0:  32% 20/62 [00:38<01:21,  1.95s/it]
Validation DataLoader 0:  48% 30/62 [00:59<01:03,  1.97s/it]
Validation DataLoader 0:  65% 40/62 [01:18<00:42,  1.95s/it]
Validation DataLoader 0:  81% 50/62 [01:36<00:23,  1.93s/it]
Validation DataLoader 0:  97% 60/62 [01:56<00:03,  1.95s/it]
Validation DataLoader 0: 100% 62/62 [01:59<00:00,  1.93s/it]
Epoch 1: 100% 638/638 [24:09<00:00,  2.27s/it, v_num=skpv, train/loss_step=0.657, train/pose_step=0.456, train/grip_step=0.200, train/reg_step=0.00205, val/loss=0.340, val/pos_error_m=0.0948, train/loss_epoch=0.868, train/pose_epoch=0.545, train/grip_epoch=0.323, train/reg_epoch=0.00208, train/router_phase_acc=0.912][2025-12-24 04:21:12,872][train_moe_post][INFO] - ========================================
[2025-12-24 04:21:12,877][train_moe_post][INFO] - Epoch 1 Expert Statistics:
[2025-12-24 04:21:12,878][train_moe_post][INFO] -   Expert 0 (Approach): Avg Loss = 0.0900 (637 samples)
[2025-12-24 04:21:12,879][train_moe_post][INFO] -   Expert 1 (Grasp): Avg Loss = 0.0926 (601 samples)
[2025-12-24 04:21:12,881][train_moe_post][INFO] -   Expert 2 (Lift): Avg Loss = 0.0872 (638 samples)
[2025-12-24 04:21:12,882][train_moe_post][INFO] -   Expert 3 (Place): Avg Loss = 0.0934 (633 samples)
[2025-12-24 04:21:12,883][train_moe_post][INFO] -   Expert 4 (Retract): Avg Loss = 0.0926 (596 samples)
[2025-12-24 04:22:55,028][train_moe_post][INFO] - [Backup] New Best Model Copied (Val Loss: 0.3396)
[2025-12-24 04:22:55,035][train_moe_post][INFO] -   Best Val Loss so far: 0.3396
[2025-12-24 04:22:55,035][train_moe_post][INFO] - ========================================
[2025-12-24 04:22:55,845][train_moe_post][INFO] - [Metrics] Saved training logs to /content/drive/MyDrive/pda/logs/moe_post/training_metrics_moe_post_v1_20251224_032823.csv
Epoch 2: 100% 638/638 [21:51<00:00,  2.06s/it, v_num=skpv, train/loss_step=0.778, train/pose_step=0.549, train/grip_step=0.229, train/reg_step=0.00148, val/loss=0.340, val/pos_error_m=0.0948, train/loss_epoch=0.868, train/pose_epoch=0.545, train/grip_epoch=0.323, train/reg_epoch=0.00208, train/router_phase_acc=0.912]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/62 [00:00<?, ?it/s]
Validation DataLoader 0:  16% 10/62 [00:18<01:37,  1.88s/it]
Validation DataLoader 0:  32% 20/62 [00:38<01:20,  1.91s/it]
Validation DataLoader 0:  48% 30/62 [00:57<01:00,  1.90s/it]
Validation DataLoader 0:  65% 40/62 [01:16<00:41,  1.91s/it]
Validation DataLoader 0:  81% 50/62 [01:35<00:22,  1.91s/it]
Validation DataLoader 0:  97% 60/62 [01:54<00:03,  1.92s/it]
Validation DataLoader 0: 100% 62/62 [01:58<00:00,  1.90s/it]
Epoch 2: 100% 638/638 [23:53<00:00,  2.25s/it, v_num=skpv, train/loss_step=0.778, train/pose_step=0.549, train/grip_step=0.229, train/reg_step=0.00148, val/loss=0.293, val/pos_error_m=0.0849, train/loss_epoch=0.802, train/pose_epoch=0.529, train/grip_epoch=0.273, train/reg_epoch=0.00218, train/router_phase_acc=0.912][2025-12-24 04:54:04,832][train_moe_post][INFO] - ========================================
[2025-12-24 04:54:04,838][train_moe_post][INFO] - Epoch 2 Expert Statistics:
[2025-12-24 04:54:04,838][train_moe_post][INFO] -   Expert 0 (Approach): Avg Loss = 0.0873 (638 samples)
[2025-12-24 04:54:04,840][train_moe_post][INFO] -   Expert 1 (Grasp): Avg Loss = 0.0884 (607 samples)
[2025-12-24 04:54:04,841][train_moe_post][INFO] -   Expert 2 (Lift): Avg Loss = 0.0847 (636 samples)
[2025-12-24 04:54:04,842][train_moe_post][INFO] -   Expert 3 (Place): Avg Loss = 0.0913 (635 samples)
[2025-12-24 04:54:04,844][train_moe_post][INFO] -   Expert 4 (Retract): Avg Loss = 0.0908 (594 samples)
[2025-12-24 04:55:46,630][train_moe_post][INFO] - [Backup] New Best Model Copied (Val Loss: 0.2928)
[2025-12-24 04:55:46,646][train_moe_post][INFO] -   Best Val Loss so far: 0.2928
[2025-12-24 04:55:46,646][train_moe_post][INFO] - ========================================
[2025-12-24 04:55:47,482][train_moe_post][INFO] - [Metrics] Saved training logs to /content/drive/MyDrive/pda/logs/moe_post/training_metrics_moe_post_v1_20251224_032823.csv
Epoch 3: 100% 638/638 [21:22<00:00,  2.01s/it, v_num=skpv, train/loss_step=0.772, train/pose_step=0.510, train/grip_step=0.262, train/reg_step=0.000942, val/loss=0.293, val/pos_error_m=0.0849, train/loss_epoch=0.802, train/pose_epoch=0.529, train/grip_epoch=0.273, train/reg_epoch=0.00218, train/router_phase_acc=0.912]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/62 [00:00<?, ?it/s]
Validation DataLoader 0:  16% 10/62 [00:19<01:40,  1.94s/it]
Validation DataLoader 0:  32% 20/62 [00:39<01:21,  1.95s/it]
Validation DataLoader 0:  48% 30/62 [00:58<01:02,  1.95s/it]
Validation DataLoader 0:  65% 40/62 [01:19<00:43,  1.98s/it]
Validation DataLoader 0:  81% 50/62 [01:38<00:23,  1.97s/it]
Validation DataLoader 0:  97% 60/62 [01:57<00:03,  1.95s/it]
Validation DataLoader 0: 100% 62/62 [02:00<00:00,  1.94s/it]
Epoch 3: 100% 638/638 [23:26<00:00,  2.20s/it, v_num=skpv, train/loss_step=0.772, train/pose_step=0.510, train/grip_step=0.262, train/reg_step=0.000942, val/loss=0.275, val/pos_error_m=0.0811, train/loss_epoch=0.757, train/pose_epoch=0.518, train/grip_epoch=0.239, train/reg_epoch=0.00188, train/router_phase_acc=0.912][2025-12-24 05:26:11,669][train_moe_post][INFO] - ========================================
[2025-12-24 05:26:11,675][train_moe_post][INFO] - Epoch 3 Expert Statistics:
[2025-12-24 05:26:11,676][train_moe_post][INFO] -   Expert 0 (Approach): Avg Loss = 0.0858 (638 samples)
[2025-12-24 05:26:11,678][train_moe_post][INFO] -   Expert 1 (Grasp): Avg Loss = 0.0858 (599 samples)
[2025-12-24 05:26:11,680][train_moe_post][INFO] -   Expert 2 (Lift): Avg Loss = 0.0834 (637 samples)
[2025-12-24 05:26:11,681][train_moe_post][INFO] -   Expert 3 (Place): Avg Loss = 0.0892 (636 samples)
[2025-12-24 05:26:11,683][train_moe_post][INFO] -   Expert 4 (Retract): Avg Loss = 0.0883 (593 samples)
[2025-12-24 05:28:03,377][train_moe_post][INFO] - [Backup] New Best Model Copied (Val Loss: 0.2749)
[2025-12-24 05:28:03,380][train_moe_post][INFO] -   Best Val Loss so far: 0.2749
[2025-12-24 05:28:03,381][train_moe_post][INFO] - ========================================
[2025-12-24 05:28:04,260][train_moe_post][INFO] - [Metrics] Saved training logs to /content/drive/MyDrive/pda/logs/moe_post/training_metrics_moe_post_v1_20251224_032823.csv
Epoch 4: 100% 638/638 [21:26<00:00,  2.02s/it, v_num=skpv, train/loss_step=0.673, train/pose_step=0.518, train/grip_step=0.155, train/reg_step=0.0015, val/loss=0.275, val/pos_error_m=0.0811, train/loss_epoch=0.757, train/pose_epoch=0.518, train/grip_epoch=0.239, train/reg_epoch=0.00188, train/router_phase_acc=0.912]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/62 [00:00<?, ?it/s]
Validation DataLoader 0:  16% 10/62 [00:19<01:42,  1.98s/it]
Validation DataLoader 0:  32% 20/62 [00:39<01:23,  2.00s/it]
Validation DataLoader 0:  48% 30/62 [00:58<01:02,  1.96s/it]
Validation DataLoader 0:  65% 40/62 [01:17<00:42,  1.95s/it]
Validation DataLoader 0:  81% 50/62 [01:36<00:23,  1.93s/it]
Validation DataLoader 0:  97% 60/62 [01:56<00:03,  1.94s/it]
Validation DataLoader 0: 100% 62/62 [02:00<00:00,  1.94s/it]
Epoch 4: 100% 638/638 [23:31<00:00,  2.21s/it, v_num=skpv, train/loss_step=0.673, train/pose_step=0.518, train/grip_step=0.155, train/reg_step=0.0015, val/loss=0.267, val/pos_error_m=0.0798, train/loss_epoch=0.738, train/pose_epoch=0.509, train/grip_epoch=0.229, train/reg_epoch=0.00188, train/router_phase_acc=0.912][2025-12-24 06:00:41,287][train_moe_post][INFO] - ========================================
[2025-12-24 06:00:41,289][train_moe_post][INFO] - Epoch 4 Expert Statistics:
[2025-12-24 06:00:41,289][train_moe_post][INFO] -   Expert 0 (Approach): Avg Loss = 0.0849 (638 samples)
[2025-12-24 06:00:41,291][train_moe_post][INFO] -   Expert 1 (Grasp): Avg Loss = 0.0854 (603 samples)
[2025-12-24 06:00:41,293][train_moe_post][INFO] -   Expert 2 (Lift): Avg Loss = 0.0825 (638 samples)
[2025-12-24 06:00:41,294][train_moe_post][INFO] -   Expert 3 (Place): Avg Loss = 0.0871 (635 samples)
[2025-12-24 06:00:41,295][train_moe_post][INFO] -   Expert 4 (Retract): Avg Loss = 0.0859 (591 samples)
[2025-12-24 06:01:52,323][train_moe_post][INFO] - [Backup] Copied periodic checkpoint: /content/drive/MyDrive/pda/models/moe_post/backups/moe_post_v1_20251224_032823_epoch_4.pt
[2025-12-24 06:03:00,038][train_moe_post][INFO] - [Backup] New Best Model Copied (Val Loss: 0.2673)
[2025-12-24 06:03:00,041][train_moe_post][INFO] -   Best Val Loss so far: 0.2673
[2025-12-24 06:03:00,041][train_moe_post][INFO] - ========================================
[2025-12-24 06:03:01,882][train_moe_post][INFO] - [Metrics] Saved training logs to /content/drive/MyDrive/pda/logs/moe_post/training_metrics_moe_post_v1_20251224_032823.csv
Epoch 5: 100% 638/638 [21:20<00:00,  2.01s/it, v_num=skpv, train/loss_step=0.720, train/pose_step=0.519, train/grip_step=0.200, train/reg_step=0.00114, val/loss=0.267, val/pos_error_m=0.0798, train/loss_epoch=0.738, train/pose_epoch=0.509, train/grip_epoch=0.229, train/reg_epoch=0.00188, train/router_phase_acc=0.912]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/62 [00:00<?, ?it/s]
Validation DataLoader 0:  16% 10/62 [00:18<01:34,  1.82s/it]
Validation DataLoader 0:  32% 20/62 [00:38<01:19,  1.90s/it]
Validation DataLoader 0:  48% 30/62 [00:57<01:01,  1.91s/it]
Validation DataLoader 0:  65% 40/62 [01:16<00:41,  1.90s/it]
Validation DataLoader 0:  81% 50/62 [01:37<00:23,  1.94s/it]
Validation DataLoader 0:  97% 60/62 [01:56<00:03,  1.95s/it]
Validation DataLoader 0: 100% 62/62 [02:00<00:00,  1.94s/it]
Epoch 5: 100% 638/638 [23:24<00:00,  2.20s/it, v_num=skpv, train/loss_step=0.720, train/pose_step=0.519, train/grip_step=0.200, train/reg_step=0.00114, val/loss=0.263, val/pos_error_m=0.0772, train/loss_epoch=0.720, train/pose_epoch=0.501, train/grip_epoch=0.219, train/reg_epoch=0.00166, train/router_phase_acc=0.912][2025-12-24 06:34:16,197][train_moe_post][INFO] - ========================================
[2025-12-24 06:34:16,205][train_moe_post][INFO] - Epoch 5 Expert Statistics:
[2025-12-24 06:34:16,205][train_moe_post][INFO] -   Expert 0 (Approach): Avg Loss = 0.0834 (638 samples)
[2025-12-24 06:34:16,207][train_moe_post][INFO] -   Expert 1 (Grasp): Avg Loss = 0.0834 (595 samples)
[2025-12-24 06:34:16,208][train_moe_post][INFO] -   Expert 2 (Lift): Avg Loss = 0.0819 (637 samples)
[2025-12-24 06:34:16,209][train_moe_post][INFO] -   Expert 3 (Place): Avg Loss = 0.0858 (638 samples)
[2025-12-24 06:34:16,211][train_moe_post][INFO] -   Expert 4 (Retract): Avg Loss = 0.0846 (593 samples)
[2025-12-24 06:35:48,414][train_moe_post][INFO] - [Backup] New Best Model Copied (Val Loss: 0.2634)
[2025-12-24 06:35:48,420][train_moe_post][INFO] -   Best Val Loss so far: 0.2634
[2025-12-24 06:35:48,420][train_moe_post][INFO] - ========================================
[2025-12-24 06:35:56,830][train_moe_post][INFO] - [Metrics] Saved training logs to /content/drive/MyDrive/pda/logs/moe_post/training_metrics_moe_post_v1_20251224_032823.csv
Epoch 6: 100% 638/638 [21:16<00:00,  2.00s/it, v_num=skpv, train/loss_step=0.770, train/pose_step=0.579, train/grip_step=0.191, train/reg_step=0.00131, val/loss=0.263, val/pos_error_m=0.0772, train/loss_epoch=0.720, train/pose_epoch=0.501, train/grip_epoch=0.219, train/reg_epoch=0.00166, train/router_phase_acc=0.912]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/62 [00:00<?, ?it/s]
Validation DataLoader 0:  16% 10/62 [00:19<01:39,  1.92s/it]
Validation DataLoader 0:  32% 20/62 [00:40<01:24,  2.01s/it]
Validation DataLoader 0:  48% 30/62 [01:00<01:04,  2.02s/it]
Validation DataLoader 0:  65% 40/62 [01:19<00:43,  1.99s/it]
Validation DataLoader 0:  81% 50/62 [01:39<00:23,  2.00s/it]
Validation DataLoader 0:  97% 60/62 [01:58<00:03,  1.98s/it]
Validation DataLoader 0: 100% 62/62 [02:01<00:00,  1.96s/it]
Epoch 6: 100% 638/638 [23:21<00:00,  2.20s/it, v_num=skpv, train/loss_step=0.770, train/pose_step=0.579, train/grip_step=0.191, train/reg_step=0.00131, val/loss=0.261, val/pos_error_m=0.0759, train/loss_epoch=0.709, train/pose_epoch=0.495, train/grip_epoch=0.214, train/reg_epoch=0.00161, train/router_phase_acc=0.912][2025-12-24 07:06:36,435][train_moe_post][INFO] - ========================================
[2025-12-24 07:06:36,440][train_moe_post][INFO] - Epoch 6 Expert Statistics:
[2025-12-24 07:06:36,441][train_moe_post][INFO] -   Expert 0 (Approach): Avg Loss = 0.0825 (638 samples)
[2025-12-24 07:06:36,443][train_moe_post][INFO] -   Expert 1 (Grasp): Avg Loss = 0.0826 (593 samples)
[2025-12-24 07:06:36,444][train_moe_post][INFO] -   Expert 2 (Lift): Avg Loss = 0.0807 (637 samples)
[2025-12-24 07:06:36,446][train_moe_post][INFO] -   Expert 3 (Place): Avg Loss = 0.0848 (638 samples)
[2025-12-24 07:06:36,447][train_moe_post][INFO] -   Expert 4 (Retract): Avg Loss = 0.0833 (586 samples)
[2025-12-24 07:08:28,071][train_moe_post][INFO] - [Backup] New Best Model Copied (Val Loss: 0.2608)
[2025-12-24 07:08:28,075][train_moe_post][INFO] -   Best Val Loss so far: 0.2608
[2025-12-24 07:08:28,076][train_moe_post][INFO] - ========================================
[2025-12-24 07:08:28,654][train_moe_post][INFO] - [Metrics] Saved training logs to /content/drive/MyDrive/pda/logs/moe_post/training_metrics_moe_post_v1_20251224_032823.csv
Epoch 7: 100% 638/638 [21:10<00:00,  1.99s/it, v_num=skpv, train/loss_step=0.762, train/pose_step=0.511, train/grip_step=0.251, train/reg_step=0.000976, val/loss=0.261, val/pos_error_m=0.0759, train/loss_epoch=0.709, train/pose_epoch=0.495, train/grip_epoch=0.214, train/reg_epoch=0.00161, train/router_phase_acc=0.912]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/62 [00:00<?, ?it/s]
Validation DataLoader 0:  16% 10/62 [00:19<01:41,  1.95s/it]
Validation DataLoader 0:  32% 20/62 [00:38<01:21,  1.94s/it]
Validation DataLoader 0:  48% 30/62 [00:58<01:02,  1.97s/it]
Validation DataLoader 0:  65% 40/62 [01:18<00:43,  1.96s/it]
Validation DataLoader 0:  81% 50/62 [01:37<00:23,  1.96s/it]
Validation DataLoader 0:  97% 60/62 [01:56<00:03,  1.95s/it]
Validation DataLoader 0: 100% 62/62 [02:00<00:00,  1.94s/it]
Epoch 7: 100% 638/638 [23:14<00:00,  2.19s/it, v_num=skpv, train/loss_step=0.762, train/pose_step=0.511, train/grip_step=0.251, train/reg_step=0.000976, val/loss=0.259, val/pos_error_m=0.0733, train/loss_epoch=0.698, train/pose_epoch=0.492, train/grip_epoch=0.206, train/reg_epoch=0.00161, train/router_phase_acc=0.912][2025-12-24 07:40:03,506][train_moe_post][INFO] - ========================================
[2025-12-24 07:40:03,516][train_moe_post][INFO] - Epoch 7 Expert Statistics:
[2025-12-24 07:40:03,516][train_moe_post][INFO] -   Expert 0 (Approach): Avg Loss = 0.0825 (637 samples)
[2025-12-24 07:40:03,518][train_moe_post][INFO] -   Expert 1 (Grasp): Avg Loss = 0.0825 (603 samples)
[2025-12-24 07:40:03,519][train_moe_post][INFO] -   Expert 2 (Lift): Avg Loss = 0.0809 (637 samples)
[2025-12-24 07:40:03,521][train_moe_post][INFO] -   Expert 3 (Place): Avg Loss = 0.0838 (637 samples)
[2025-12-24 07:40:03,522][train_moe_post][INFO] -   Expert 4 (Retract): Avg Loss = 0.0824 (590 samples)
[2025-12-24 07:41:41,814][train_moe_post][INFO] - [Backup] New Best Model Copied (Val Loss: 0.2592)
[2025-12-24 07:41:41,821][train_moe_post][INFO] -   Best Val Loss so far: 0.2592
[2025-12-24 07:41:41,821][train_moe_post][INFO] - ========================================
[2025-12-24 07:41:42,483][train_moe_post][INFO] - [Metrics] Saved training logs to /content/drive/MyDrive/pda/logs/moe_post/training_metrics_moe_post_v1_20251224_032823.csv
Epoch 8: 100% 638/638 [21:19<00:00,  2.01s/it, v_num=skpv, train/loss_step=0.834, train/pose_step=0.508, train/grip_step=0.326, train/reg_step=0.0017, val/loss=0.259, val/pos_error_m=0.0733, train/loss_epoch=0.698, train/pose_epoch=0.492, train/grip_epoch=0.206, train/reg_epoch=0.00161, train/router_phase_acc=0.912]  
Validation: |          | 0/? [00:00<?, ?it/s]
Validation: |          | 0/? [00:00<?, ?it/s]
Validation DataLoader 0:   0% 0/62 [00:00<?, ?it/s]
Validation DataLoader 0:  16% 10/62 [00:18<01:37,  1.87s/it]
Validation DataLoader 0:  32% 20/62 [00:37<01:19,  1.89s/it]
Validation DataLoader 0:  48% 30/62 [00:58<01:02,  1.96s/it]
Validation DataLoader 0:  65% 40/62 [01:17<00:42,  1.93s/it]
Validation DataLoader 0:  81% 50/62 [01:36<00:23,  1.92s/it]

