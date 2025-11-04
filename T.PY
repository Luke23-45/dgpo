# In FILE: models/vip_c.py
# In CLASS: ViPC

    @torch.no_grad()
    def plan(self,
             current_image: torch.Tensor,
             goal_image: torch.Tensor,
             task_phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference-only method to run the Planner and extract the subgoals.
        Returns: predicted_heatmap, predicted_implicit_subgoal, predicted_subgoal_coord
        """
        self.eval()
        
        # --- [MODIFY THIS] ---
        predicted_heatmap, predicted_implicit_subgoal = self.planner(current_image, goal_image, task_phase)
        predicted_subgoal_coord = self.soft_argmax_2d(predicted_heatmap)
        return predicted_heatmap, predicted_implicit_subgoal, predicted_subgoal_coord
        # --- [END MODIFY] ---

# In FILE: train/train_vip_c.py
# In CLASS: ViPCLightningModule._log_qualitative_validation

    def _log_qualitative_validation(self, batch: Dict[str, Any]):
        try:
            with torch.no_grad():
                # --- [MODIFY THIS] ---
                # plan now returns three items
                predicted_heatmap, _, _ = self.ema.ema_model.plan(
                    current_image=batch['planner_current_image'],
                    goal_image=batch['planner_goal_image'],
                    task_phase=batch['planner_task_phase']
                )
                # --- [END MODIFY] ---
            # ... (rest of the validation method is correct) ...