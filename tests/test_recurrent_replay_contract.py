from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def _compute_prio_kernel_source() -> str:
    source = (ROOT / "src" / "pufferlib.cu").read_text()
    match = re.search(
        r"__global__ void compute_prio_adv_reduction\(.*?\n}\n",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(0)


def test_recurrent_replay_keeps_prefix_before_first_internal_terminal():
    kernel = _compute_prio_kernel_source()

    # A terminal at t=0 is a valid new-episode action when replay starts from
    # zero state. A later terminal begins an unrepresentable second sequence,
    # so priority uses only the prefix before it.
    assert "valid_steps = stride" in kernel
    assert "for (int t = 1; t < stride; t++)" in kernel
    assert "t < valid_steps" in kernel


def test_selected_recurrent_state_zeros_terminal_at_chunk_start():
    source = (ROOT / "src" / "pufferlib.cu").read_text()
    match = re.search(
        r"__global__ void select_recurrent_states\(.*?\n}\n",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    kernel = match.group(0)

    assert "reset_at_start" in kernel
    assert "source_agent * horizon" in kernel
    assert "? from_float(0.0f)" in kernel


def test_recurrent_prefix_mask_starts_after_terminal_boundary():
    source = (ROOT / "src" / "pufferlib.cu").read_text()
    match = re.search(
        r"__device__ __forceinline__ void copy_recurrent_prefix_mask\(.*?\n}\n",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    helper = match.group(0)

    assert "t > 0" in helper
    assert "valid = false" in helper
    assert "dst_valid[drh + t]" in helper


def test_ppo_masks_suffix_gradients_and_preserves_replay_state():
    source = (ROOT / "src" / "pufferlib.cu").read_text()
    start = source.index("__global__ void ppo_loss_compute")
    end = source.index("__global__ void ppo_loss_reduce", start)
    kernel = source[start:end]

    assert "g.valid[nt]" in kernel
    assert "g.out_ratio[nt] = from_float(1.0f)" in kernel
    assert "g.out_newvalue[nt] = g.values[nt]" in kernel
    assert "a.grad_values_pred[nt] = 0.0f" in kernel
    assert "a.grad_logits[grad_logits_base + h] = 0.0f" in kernel


def test_advantage_normalization_counts_only_valid_prefix_tokens():
    source = (ROOT / "src" / "pufferlib.cu").read_text()
    start = source.index("__global__ void ppo_masked_var_mean")
    end = source.index("void ppo_loss_fwd_bwd", start)
    kernel = source[start:end]

    assert "to_float(valid[i]) > 0.5f" in kernel
    assert "*count_out = valid_count" in kernel
    assert "valid_count > 1.0f" in kernel


def test_rollout_start_state_is_captured_before_terminal_masking():
    source = (ROOT / "src" / "pufferlib.cu").read_text()
    callback_start = source.index('extern "C" void net_callback_wrapper')
    callback_end = source.index(
        "__device__ __forceinline__ void ppo_discrete_head", callback_start
    )
    callback = source[callback_start:callback_end]

    capture = callback.index("capture_rollout_initial_states<<<")
    reset = callback.index("reset_terminal_recurrent_states<<<")
    forward = callback.index("policy_forward(")

    assert "if (t == 0 && !hypers.reset_state)" in callback
    assert capture < reset < forward


def test_cuda_graph_warmup_rezeros_persistent_recurrent_state():
    source = (ROOT / "src" / "pufferlib.cu").read_text()
    warmup_restore = source.index("// Re-init RNG states corrupted by warmup")
    warmup_end = source.index("pufferl->epoch = 0;", warmup_restore)
    restored = source[warmup_restore:warmup_end]

    assert "puf_zero(&pufferl->rollout_initial_states" in restored
    assert "puf_zero(&pufferl->buffer_states[i]" in restored
