# tests/test_transfer_bc_weights.py
import unittest
import torch
import torch.nn as nn
from collections import OrderedDict
from unittest.mock import MagicMock
import logging

# The function we are testing
from utils.transfer_bc_to_ppo import transfer_bc_weights

# Suppress the logger output during tests for cleaner results
logging.disable(logging.CRITICAL)


# --- Mock Models for Realistic Testing ---

class MockBCNet(nn.Module):
    """A fake BCNet with parameter names matching the transfer script's expectations."""
    def __init__(self):
        super().__init__()
        self.proprio_mlp = nn.Sequential(OrderedDict([
            ('0', nn.Linear(10, 32)), ('1', nn.ReLU()), ('2', nn.Linear(32, 64))
        ]))
        self.head = nn.Sequential(OrderedDict([
            ('0', nn.Linear(64, 128)), ('1', nn.ReLU()),
            ('2', nn.Linear(128, 128)), ('3', nn.ReLU()),
            ('4', nn.Linear(128, 8))
        ]))
        self.unmatched_param = nn.Parameter(torch.randn(5, 5))

class MockPPOPolicy(nn.Module):
    """A fake PPO Policy with parameter names matching the transfer script's targets."""
    def __init__(self):
        super().__init__()
        self.features_extractor = nn.ModuleDict({
            "mlp": nn.Sequential(OrderedDict([
                ('0', nn.Linear(10, 32)), ('1', nn.ReLU()), ('2', nn.Linear(32, 64))
            ]))
        })
        self.mlp_extractor = nn.ModuleDict({
            "policy_net": nn.Sequential(OrderedDict([
                ('0', nn.Linear(64, 128)), ('1', nn.ReLU()), ('2', nn.Linear(128, 128))
            ]))
        })
        self.action_net = nn.Linear(128, 8)
        self.value_net = nn.Linear(128, 1)
        # --- FIX: Add a layer to create a true ambiguity for the suffix-based search ---
        self.another_ambiguous_net = nn.Linear(128, 8)


class TestWeightTransfer(unittest.TestCase):
    """A test suite for the robust weight transfer utility."""

    def setUp(self):
        self.bc_model = MockBCNet()
        self.ppo_policy = MockPPOPolicy()
        self.mock_ppo_agent = MagicMock()
        self.mock_ppo_agent.policy = self.ppo_policy

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def test_A_prefix_mapping_success(self):
        """Test that the prefix-based mapping works correctly."""
        known_weight = torch.full_like(self.bc_model.proprio_mlp[0].weight, 3.14)
        self.bc_model.proprio_mlp[0].weight.data.copy_(known_weight)
        report = transfer_bc_weights(self.bc_model, self.mock_ppo_agent, verbose=False)
        self.assertIn(('proprio_mlp.0.weight', 'features_extractor.mlp.0.weight'), report['transferred'])
        ppo_weight = self.ppo_policy.features_extractor.mlp[0].weight
        self.assertTrue(torch.equal(ppo_weight, known_weight))
        print("\n✅ [TestWeightTransfer] Prefix mapping is correct.")

    def test_B_explicit_head_mapping_success(self):
        """Test that the explicit head mapping for the action layer works correctly."""
        known_weight = torch.full_like(self.bc_model.head[4].weight, 7.77)
        self.bc_model.head[4].weight.data.copy_(known_weight)
        report = transfer_bc_weights(self.bc_model, self.mock_ppo_agent, verbose=False)
        self.assertIn(('head.4.weight', 'action_net.weight'), report['transferred'])
        ppo_weight = self.ppo_policy.action_net.weight
        self.assertTrue(torch.equal(ppo_weight, known_weight))
        print("✅ [TestWeightTransfer] Explicit head mapping is correct.")

    def test_C_shape_mismatch_skips_transfer(self):
        """Test that parameters with mismatched shapes are correctly skipped."""
        self.bc_model.head[0] = nn.Linear(64, 999)
        original_ppo_weight = self.ppo_policy.mlp_extractor.policy_net[0].weight.clone()
        report = transfer_bc_weights(self.bc_model, self.mock_ppo_agent, verbose=False)
        self.assertEqual(len(report['skipped_shape_mismatch']), 2)
        skipped_item = report['skipped_shape_mismatch'][0]
        self.assertEqual(skipped_item[0], 'head.0.weight')
        self.assertIn('policy_net.0.weight', skipped_item[1])
        self.assertTrue(torch.equal(self.ppo_policy.mlp_extractor.policy_net[0].weight, original_ppo_weight))
        print("✅ [TestWeightTransfer] Shape mismatch check is correct.")

    def test_D_unmatched_and_ambiguous_reporting(self):
        """Test the reporting of unfound and ambiguous parameters."""
        # Add a layer to the BC model that will cause an ambiguous match
        self.bc_model.ambiguous_layer = nn.Linear(128, 8)
        
        report = transfer_bc_weights(self.bc_model, self.mock_ppo_agent, verbose=False)

        # 1. Check for the parameter that has no defined match
        self.assertIn('unmatched_param', report['not_found'])
        
        # 2. Check for the ambiguous match
        # It's ambiguous because 'ambiguous_layer.weight' (shape 8,128) could map to
        # 'action_net.weight' OR 'another_ambiguous_net.weight' via suffix matching.
        ambiguous_found = False
        for bc_key, ppo_candidates in report['ambiguous']:
            if bc_key == 'ambiguous_layer.weight':
                self.assertIn('action_net.weight', ppo_candidates)
                self.assertIn('another_ambiguous_net.weight', ppo_candidates)
                self.assertEqual(len(ppo_candidates), 2)
                ambiguous_found = True
                break
        
        self.assertTrue(ambiguous_found, "Ambiguous match was not reported correctly.")
        print("✅ [TestWeightTransfer] Reporting for unmatched/ambiguous params is correct.")


if __name__ == '__main__':
    unittest.main(verbosity=2)