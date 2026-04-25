#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def _align8(idx):
    return (idx + 7) & ~7


def _take_aligned(weights, idx, count):
    chunk = weights[idx:idx + count]
    if len(chunk) != count:
        raise ValueError(f"weight file ended early: requested {count}, got {len(chunk)}")
    return chunk, _align8(idx + count)


def _quantize_per_output(weight_matrix):
    max_abs = np.max(np.abs(weight_matrix), axis=1)
    scales = np.where(max_abs > 1e-8, max_abs / 127.0, 1.0 / 127.0).astype(np.float32)
    q = np.round(weight_matrix / scales[:, None]).clip(-127, 127).astype(np.int8)
    return q, scales


def main():
    parser = argparse.ArgumentParser(description="Export PufferNet float weights to Q8 per-channel arrays")
    parser.add_argument("input", type=Path, help="Float32 PufferNet .bin weight file")
    parser.add_argument("output", type=Path, help="Output .npz path")
    parser.add_argument("--input-dim", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-actions", type=int, default=4)
    args = parser.parse_args()

    weights = np.fromfile(args.input, dtype=np.float32)
    idx = 0
    arrays = {}
    manifest = {
        "format": "puffernet_q8_npz_v1",
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_actions": args.num_actions,
        "notes": "Per-output-channel int8 weights with float32 dequant scales. Recurrent state remains mixed precision.",
    }

    encoder, idx = _take_aligned(weights, idx, args.hidden_dim * args.input_dim)
    encoder = encoder.reshape(args.hidden_dim, args.input_dim)
    arrays["encoder_q"], arrays["encoder_scale"] = _quantize_per_output(encoder)

    decoder_dim = args.num_actions + 1
    decoder, idx = _take_aligned(weights, idx, decoder_dim * args.hidden_dim)
    decoder = decoder.reshape(decoder_dim, args.hidden_dim)
    arrays["decoder_q"], arrays["decoder_scale"] = _quantize_per_output(decoder)

    log_std, idx = _take_aligned(weights, idx, args.num_actions)
    arrays["log_std"] = log_std.astype(np.float32)

    for layer_idx in range(args.num_layers):
        proj, idx = _take_aligned(weights, idx, 3 * args.hidden_dim * args.hidden_dim)
        proj = proj.reshape(3 * args.hidden_dim, args.hidden_dim)
        arrays[f"mingru_{layer_idx}_proj_q"], arrays[f"mingru_{layer_idx}_proj_scale"] = _quantize_per_output(proj)

    manifest["float_weights_consumed"] = int(idx)
    manifest["float_weights_available"] = int(len(weights))
    arrays["manifest_json"] = np.array(json.dumps(manifest, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
