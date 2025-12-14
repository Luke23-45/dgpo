"""
Analysis script to compare hovering behavior between semantic and unified planners.
"""
import pandas as pd
import numpy as np

# Load both CSVs
print("Loading data...")
semantic = pd.read_csv('dgpo_eval_dagger_iter_0011_20251211_190220.csv')
unified = pd.read_csv('unified_eval_unified_planner_backup_epoch_017_20251214_073510.csv')

print('='*60)
print('SEMANTIC PLANNER (DAgger) ANALYSIS')
print('='*60)
print(f'Total steps: {len(semantic)}')
print(f'Episodes: {semantic["episode_id"].nunique()}')
print(f'Steps per episode: {len(semantic) // semantic["episode_id"].nunique()}')
print()

# Grasp analysis
print('GRASP ANALYSIS:')
print(f'  is_grasped mean: {semantic["is_grasped"].mean():.4f}')
print(f'  is_grasped max: {semantic["is_grasped"].max():.4f}')
print(f'  Ever grasped: {(semantic["is_grasped"] > 0.5).any()}')
print()

# Distance analysis
print('DISTANCE ANALYSIS (Episode 0):')
ep0 = semantic[semantic['episode_id'] == 0]
print(f'  dist_ee_obj: {ep0["dist_ee_obj"].min():.4f} to {ep0["dist_ee_obj"].max():.4f} (mean: {ep0["dist_ee_obj"].mean():.4f})')
print(f'  dist_obj_goal start: {ep0["dist_obj_goal"].iloc[0]:.4f}')
print(f'  dist_obj_goal end: {ep0["dist_obj_goal"].iloc[-1]:.4f}')
print(f'  dist_obj_goal changed: {abs(ep0["dist_obj_goal"].iloc[0] - ep0["dist_obj_goal"].iloc[-1]) > 0.01}')
print()

# Gripper command analysis
print('GRIPPER CMD ANALYSIS (Episode 0):')
print(f'  gripper_cmd=-1 (close) count: {(ep0["gripper_cmd"] < 0).sum()}')
print(f'  gripper_cmd=+1 (open) count: {(ep0["gripper_cmd"] > 0).sum()}')
print()

# Z-height analysis (hovering detection)
print('Z-HEIGHT ANALYSIS (Hovering Detection):')
print(f'  ee_z min: {ep0["ee_z"].min():.4f}')
print(f'  ee_z max: {ep0["ee_z"].max():.4f}')
print(f'  ee_z mean: {ep0["ee_z"].mean():.4f}')
print(f'  obj_z (table height): {ep0["obj_z"].mean():.4f}')
print(f'  Min clearance above object: {(ep0["ee_z"] - ep0["obj_z"]).min():.4f}')

print()
print('='*60)
print('UNIFIED PLANNER ANALYSIS')
print('='*60)
print(f'Total steps: {len(unified)}')
print(f'Episodes: {unified["episode_id"].nunique()}')
print(f'Steps per episode: {len(unified) // unified["episode_id"].nunique()}')
print()

# Grasp analysis
print('GRASP ANALYSIS:')
print(f'  is_grasped mean: {unified["is_grasped"].mean():.4f}')
print(f'  is_grasped max: {unified["is_grasped"].max():.4f}')
print(f'  Ever grasped: {(unified["is_grasped"] > 0.5).any()}')
print()

# Distance analysis per episode
print('DISTANCE ANALYSIS (per episode):')
for ep_id in unified['episode_id'].unique():
    ep = unified[unified['episode_id'] == ep_id]
    print(f'  Episode {ep_id}:')
    print(f'    dist_ee_obj: {ep["dist_ee_obj"].min():.4f} to {ep["dist_ee_obj"].max():.4f}')
    print(f'    dist_obj_goal change: {ep["dist_obj_goal"].iloc[0]:.4f} -> {ep["dist_obj_goal"].iloc[-1]:.4f}')
    print(f'    Closest approach to object: {ep["dist_ee_obj"].min():.4f}')
print()

# Policy delta analysis
print('POLICY DELTA ANALYSIS (Episode 0):')
ep0_u = unified[unified['episode_id'] == 0]
print(f'  policy_dx: mean={ep0_u["policy_dx"].mean():.6f}, std={ep0_u["policy_dx"].std():.6f}')
print(f'  policy_dy: mean={ep0_u["policy_dy"].mean():.6f}, std={ep0_u["policy_dy"].std():.6f}')
print(f'  policy_dz: mean={ep0_u["policy_dz"].mean():.6f}, std={ep0_u["policy_dz"].std():.6f}')
print(f'  policy_grip: mean={ep0_u["policy_grip"].mean():.4f}, std={ep0_u["policy_grip"].std():.4f}')
print()

# Gripper command analysis
print('GRIPPER CMD ANALYSIS:')
print(f'  gripper_cmd min: {unified["gripper_cmd"].min():.4f}')
print(f'  gripper_cmd max: {unified["gripper_cmd"].max():.4f}')
print(f'  gripper_cmd mean: {unified["gripper_cmd"].mean():.4f}')
print()

# Z-height analysis (hovering detection)
print('Z-HEIGHT ANALYSIS (Hovering Detection):')
print(f'  ee_z min: {ep0_u["ee_z"].min():.4f}')
print(f'  ee_z max: {ep0_u["ee_z"].max():.4f}')
print(f'  ee_z mean: {ep0_u["ee_z"].mean():.4f}')
print(f'  obj_z (table height): {ep0_u["obj_z"].mean():.4f}')
print(f'  Min clearance above object: {(ep0_u["ee_z"] - ep0_u["obj_z"]).min():.4f}')

# Success analysis
print()
print('SUCCESS ANALYSIS:')
print(f'  Semantic success_flag sum: {semantic["success_flag"].sum()}')
print(f'  Unified success_flag sum: {unified["success_flag"].sum()}')

# Phase analysis - where does the robot spend most time?
print()
print('='*60)
print('PHASE ANALYSIS - WHERE IS THE ROBOT HOVERING?')
print('='*60)

# For unified planner
print('\nUNIFIED PLANNER - Trajectory phases:')
for ep_id in unified['episode_id'].unique():
    ep = unified[unified['episode_id'] == ep_id]
    
    # Divide into quartiles by time
    q1 = ep.iloc[:len(ep)//4]
    q2 = ep.iloc[len(ep)//4:len(ep)//2]
    q3 = ep.iloc[len(ep)//2:3*len(ep)//4]
    q4 = ep.iloc[3*len(ep)//4:]
    
    print(f'\nEpisode {ep_id} phases:')
    print(f'  Q1 (0-25%): dist_ee_obj={q1["dist_ee_obj"].mean():.4f}, ee_z={q1["ee_z"].mean():.4f}')
    print(f'  Q2 (25-50%): dist_ee_obj={q2["dist_ee_obj"].mean():.4f}, ee_z={q2["ee_z"].mean():.4f}')
    print(f'  Q3 (50-75%): dist_ee_obj={q3["dist_ee_obj"].mean():.4f}, ee_z={q3["ee_z"].mean():.4f}')
    print(f'  Q4 (75-100%): dist_ee_obj={q4["dist_ee_obj"].mean():.4f}, ee_z={q4["ee_z"].mean():.4f}')

# Analyze when robot is close but not grasping
print()
print('='*60)
print('CRITICAL: CLOSE BUT NOT GRASPING ANALYSIS')
print('='*60)
CLOSE_THRESHOLD = 0.05  # 5cm

for ep_id in unified['episode_id'].unique():
    ep = unified[unified['episode_id'] == ep_id]
    close_steps = ep[ep['dist_ee_obj'] < CLOSE_THRESHOLD]
    
    print(f'\nEpisode {ep_id}:')
    print(f'  Steps within {CLOSE_THRESHOLD*100:.0f}cm of object: {len(close_steps)} / {len(ep)} ({100*len(close_steps)/len(ep):.1f}%)')
    
    if len(close_steps) > 0:
        print(f'  During close steps:')
        print(f'    is_grasped mean: {close_steps["is_grasped"].mean():.4f}')
        print(f'    gripper_cmd mean: {close_steps["gripper_cmd"].mean():.4f}')
        print(f'    ee_z mean: {close_steps["ee_z"].mean():.4f}')
        print(f'    obj_z mean: {close_steps["obj_z"].mean():.4f}')
