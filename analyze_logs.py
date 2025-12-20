
import pandas as pd
import numpy as np
import sys
import os

def analyze_logs(file_path):
    print(f"🔍 Analyzing Log File: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print("Please check the path. If running on Colab, it might be in '/content/drive/MyDrive/pda/logs/training_metrics.csv'")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return

    # Check required columns
    required_cols = ['iteration', 'blend_alpha', 'exec_pos_div', 'success_rate']
    for col in required_cols:
        if col not in df.columns:
            print(f"⚠️ Warning: Column '{col}' not found in CSV. Available: {df.columns.tolist()}")
    
    # 1. Overview
    print("\n📊 Data Overview:")
    print(f"   Total Iterations: {len(df)}")
    print(f"   Alpha Range: {df['blend_alpha'].min()} - {df['blend_alpha'].max()}")
    print(f"   Exec Div Range: {df['exec_pos_div'].min()} - {df['exec_pos_div'].max()}")
    
    # 2. Check Blending Correlation
    print("\n🕵️‍♀️ Blending Integrity Check:")
    
    # Filter for active blending
    blending_active = df[df['blend_alpha'] > 0.0]
    
    if len(blending_active) == 0:
        print("   ⚠️ No iterations with Alpha > 0 found.")
    else:
        # Check Exec Div during blending
        zero_exec = blending_active[blending_active['exec_pos_div'] < 1e-6] # Effectively 0
        non_zero_exec = blending_active[blending_active['exec_pos_div'] >= 1e-6]
        
        print(f"   Iterations with Alpha > 0: {len(blending_active)}")
        print(f"   Iterations with Exec=0.00cm despite Alpha>0: {len(zero_exec)} (❌ SUSPICIOUS)")
        print(f"   Iterations with Exec>0.00cm (Working): {len(non_zero_exec)}")
        
        if len(zero_exec) > 0:
            print("\n❌ CRITICAL FINDING: Blending was IGNORED in the following iterations:")
            print(zero_exec[['iteration', 'blend_alpha', 'exec_pos_div']].to_string(index=False))
            print("\nThis confirms that despite Alpha > 0, the environment executed the EXPERT action exactly.")
        else:
            print("\n✅ OK: Execution Divergence is present when Alpha > 0.")
            
        # 3. Correlation
        if len(non_zero_exec) > 0:
            corr = non_zero_exec['blend_alpha'].corr(non_zero_exec['exec_pos_div'])
            print(f"\n📈 Correlation (Alpha vs ExecDiv): {corr:.4f}")
            if corr > 0.5:
                print("   (Strong positive correlation - As Alpha increases, Policy deviation increases. This is EXPECTED.)")
            else:
                print("   (Weak or negative correlation - Unusual.)")

    # 4. Success Analysis
    print("\n🏆 Success Rate Analysis:")
    if 'success_rate' in df.columns:
        print(f"   Max Success: {df['success_rate'].max()}%")
        failed_iters = df[df['success_rate'] == 0.0]
        print(f"   Iterations with 0% Success: {len(failed_iters)}")
        if len(failed_iters) == len(df):
            print("❌ CRITICAL: Success rate never went above 0%.")
            print("   Possible Causes: Expert Logic broken, Physics instability, or Goal Unreachable.")

if __name__ == "__main__":
    # Default path or argument
    path = "logs/training_metrics.csv"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    
    analyze_logs(path)
