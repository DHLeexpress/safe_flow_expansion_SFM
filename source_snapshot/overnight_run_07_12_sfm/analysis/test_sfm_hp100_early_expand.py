from argparse import Namespace

import torch

import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_early_expand as EARLY


def _args(**updates):
    values = dict(
        rounds=3, verifier_workers=32, batch_size=64,
        learning_rate=3.0e-7, negative_alpha=1.0e-3, seed=2,
    )
    values.update(updates)
    return Namespace(**values)


def test_early_protocol_is_the_declared_executed_window_arm():
    config = EARLY.protocol_config(_args(), lengthscale=0.5)
    assert (config.rounds, config.max_steps, config.max_steps_status) == (
        3, 30, "EARLY_CUTOFF",
    )
    assert (config.K, config.B, config.parallel_episodes) == (64, 16, 16)
    assert config.max_retry_batches == 1
    assert config.successful_trajectories_per_gamma == 0
    assert config.archive_rule == "executed_plus_nvp_negative"
    assert config.replay_acceptance == "execution_eligible"
    assert config.execution_rule == "min_cost"
    assert config.execution_step_margin_weight == 50_000.0
    assert config.flow_base_std == 3.0
    assert config.gp_reference_mode == (
        "sliding_executed_positive_per_gamma_frozen_phi"
    )
    assert config.gp_reference_rounds == config.replay_rounds == 2
    assert config.gp_buffer_cap == 2_688
    assert config.gp_buffer_cap // len(config.gammas) == 384
    assert config.gp_sliding_row_selector == "trajectory_uniform"
    assert config.optimizer_scope == "last_block_and_head"
    assert config.head_only_update is False
    assert config.inner_steps is None and config.microbatch_repeats == 1
    assert config.learning_rate == 3.0e-7
    assert config.negative_alpha == 1.0e-3
    assert config.adaptive_beta and config.ess_target == 0.1


def test_last_block_and_head_scope_is_exact_and_leaves_context_encoder_frozen():
    policy = GPS.GridSFMHP100FlowPolicy()
    adapter = PORT.HP100ExpansionPolicy(policy)
    parameters = adapter.expansion_optimizer_parameters("last_block_and_head")
    trainable = {
        name for name, value in adapter.named_parameters() if value.requires_grad
    }
    expected = {
        *(f"policy.trunk.blocks.1.{name}" for name, _ in policy.trunk.blocks[1].named_parameters()),
        *(f"policy.head.{name}" for name, _ in policy.head.named_parameters()),
    }
    assert trainable == expected
    assert sum(parameter.numel() for parameter in parameters) == 137_236
    assert all(not value.requires_grad for value in policy.grid_conv.parameters())
    assert all(not value.requires_grad for value in policy.grid_projection.parameters())
    assert all(not value.requires_grad for value in policy.gru.parameters())
    assert all(not value.requires_grad for value in policy.enc_low.parameters())
    assert all(not value.requires_grad for value in policy.trunk.blocks[0].parameters())


def test_last_block_and_head_can_move_without_changing_frozen_modules():
    policy = GPS.GridSFMHP100FlowPolicy().eval()
    adapter = PORT.HP100ExpansionPolicy(policy)
    parameters = adapter.expansion_optimizer_parameters("last_block_and_head")
    frozen_before = {
        name: value.detach().clone()
        for name, value in adapter.named_parameters()
        if not value.requires_grad
    }
    optimizer = torch.optim.Adam(parameters, lr=1.0e-4)
    loss = sum(parameter.square().sum() for parameter in parameters)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert all(
        torch.equal(value, dict(adapter.named_parameters())[name])
        for name, value in frozen_before.items()
    )
