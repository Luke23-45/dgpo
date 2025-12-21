import torch

def test_kl_gradient_flow():
    print("Alignment Test: Checking KL Penalty Gradient Flow")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Mock Data (Log Probs)
    B = 4
    log_probs_old = torch.randn(B).to(device).detach() # Fixed old policy
    
    # 2. Mock New Log Probs (The output of our current policy)
    # We set requires_grad=True to simulate that these come from the Policy Network
    log_probs_new = torch.randn(B, requires_grad=True, device=device)
    
    # 3. Simulate the Buggy Logic (from train_dgpo_robust.py)
    beta = 10.0
    
    print("\n--- TEST 1: The 'Proposed' Buggy Logic ---")
    print("Code: with torch.no_grad(): kl = (old - new).mean()")
    print("      penalty = beta * kl")
    
    # --- BUG SIMULATION ---
    with torch.no_grad():
        kl_div_detached = (log_probs_old - log_probs_new).mean()
        
    penalty_buggy = beta * kl_div_detached
    
    # Verify: Does 'penalty_buggy' have a grad_fn?
    print(f"Penalty Grad Fn: {penalty_buggy.grad_fn}")
    
    # Try Backward
    try:
        loss_1 = penalty_buggy * 1.0
        # If the tensor is a scalar with no graph, backward() might fail or do nothing
        if penalty_buggy.requires_grad:
            loss_1.backward()
            print(f"Gradients on LogProb: {log_probs_new.grad}")
        else:
            print(">>> CRITICAL FAILURE: 'penalty_buggy' has requires_grad=False.")
            print(">>> The optimizer updates NOTHING.")
            
    except Exception as e:
        print(f"Backward Failed: {e}")

    # Reset
    log_probs_new.grad = None
    
    print("\n--- TEST 2: The Correct Logic ---")
    print("Code: kl = (old - new).mean()  # No torch.no_grad()!")
    
    # --- CORRECT LOGIC ---
    kl_div_correct = (log_probs_old - log_probs_new).mean() 
    penalty_correct = beta * kl_div_correct
    
    print(f"Penalty Grad Fn: {penalty_correct.grad_fn}")
    
    loss_2 = penalty_correct * 1.0
    loss_2.backward()
    
    if log_probs_new.grad is not None:
        grad_sum = log_probs_new.grad.abs().sum().item()
        print(f"Gradients on LogProb (Sum): {grad_sum:.4f}")
        if grad_sum > 0:
            print(">>> SUCCESS: Gradients are flowing correctly.")

if __name__ == "__main__":
    test_kl_gradient_flow()
