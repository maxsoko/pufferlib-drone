#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os

import numpy as np
import torch

import pufferlib
import pufferlib.pytorch
from pufferlib.environments.drone_hover.environment import DroneHoverEnv
import pufferlib.emulation
from pufferlib.environments.drone_hover.torch import Policy, SquashedPolicy


def load_policy(env, model_path=None, device='cpu', policy_name='SquashedPolicy', hidden_size=256):
    policy_cls = SquashedPolicy if policy_name == 'SquashedPolicy' else Policy
    policy = policy_cls(env, hidden_size=hidden_size)
    policy.to(device)
    policy.eval()

    if model_path:
        state_dict = torch.load(model_path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)
    return policy


def eval_once(env, policy, seed, device, deterministic=False):
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    steps = 0
    pos_error_sum = 0.0
    angle_error_sum = 0.0
    hover_time_max = 0.0
    success = 0.0

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = policy.forward_eval(obs_t)
            if deterministic and hasattr(logits, "loc"):
                action = torch.sigmoid(logits.loc)
            elif deterministic and isinstance(logits, torch.Tensor):
                action = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                action, _, _ = pufferlib.pytorch.sample_logits(logits)

        action = action.cpu().numpy().reshape(env.action_space.shape)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        steps += 1
        pos_error_sum += float(info.get('pos_error', 0.0))
        angle_error_sum += float(info.get('angle_error', 0.0))
        hover_time_max = max(hover_time_max, float(info.get('hover_time', 0.0)))
        success = max(success, float(info.get('success', 0.0)))

    return {
        'total_reward': total_reward,
        'steps': steps,
        'success': success,
        'hover_time_max': hover_time_max,
        'pos_error_sum': pos_error_sum,
        'angle_error_sum': angle_error_sum,
    }


def run_eval(episodes, seed, model_path, csv_path, device, policy_name, env_kwargs, deterministic, hidden_size):
    raw_env = DroneHoverEnv(**env_kwargs)
    puffer_env = pufferlib.emulation.GymnasiumPufferEnv(env=raw_env)
    policy = load_policy(puffer_env, model_path=model_path, device=device, policy_name=policy_name, hidden_size=hidden_size)

    rows = []
    successes = 0

    for ep in range(episodes):
        metrics = eval_once(raw_env, policy, seed + ep, device, deterministic=deterministic)
        successes += int(metrics['success'] > 0.0)
        rows.append({
            'episode': ep,
            'seed': seed + ep,
            **metrics,
        })

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    success_rate = successes / episodes
    print(f"Success rate: {success_rate:.2%} ({successes}/{episodes})")
    print(f"CSV saved to: {csv_path}")

    return success_rate


def run_eval_multi(episodes, seed, model_paths, csv_path, device, policy_name, env_kwargs, deterministic, hidden_size):
    raw_env = DroneHoverEnv(**env_kwargs)
    puffer_env = pufferlib.emulation.GymnasiumPufferEnv(env=raw_env)
    summary_rows = []

    for model_path in model_paths:
        policy = load_policy(puffer_env, model_path=model_path, device=device, policy_name=policy_name, hidden_size=hidden_size)
        successes = 0
        rewards = []
        hover_times = []

        for ep in range(episodes):
            metrics = eval_once(raw_env, policy, seed + ep, device, deterministic=deterministic)
            successes += int(metrics['success'] > 0.0)
            rewards.append(metrics['total_reward'])
            hover_times.append(metrics['hover_time_max'])

        success_rate = successes / episodes
        summary_rows.append({
            'model_path': model_path,
            'success_rate': success_rate,
            'mean_reward': float(np.mean(rewards)),
            'mean_hover_time': float(np.mean(hover_times)),
        })

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Summary CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate drone hover policy')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--model-glob', type=str, default=None)
    parser.add_argument('--csv-path', type=str, default='experiments/drone_hover_eval.csv')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--policy-name', type=str, default='SquashedPolicy', choices=['Policy', 'SquashedPolicy'])
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--env-kwargs', type=str, default='{}', help='JSON dict of env kwargs')
    parser.add_argument('--deterministic', action='store_true', help='Use mean/argmax actions instead of sampling')
    args = parser.parse_args()

    try:
        env_kwargs = json.loads(args.env_kwargs)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --env-kwargs JSON: {exc}") from exc

    if args.model_glob:
        model_paths = sorted(glob.glob(args.model_glob))
        if not model_paths:
            raise SystemExit(f"No models found for glob: {args.model_glob}")
        run_eval_multi(
            episodes=args.episodes,
            seed=args.seed,
            model_paths=model_paths,
            csv_path=args.csv_path,
            device=args.device,
            policy_name=args.policy_name,
            env_kwargs=env_kwargs,
            deterministic=args.deterministic,
            hidden_size=args.hidden_size,
        )
    else:
        run_eval(
            episodes=args.episodes,
            seed=args.seed,
            model_path=args.model_path,
            csv_path=args.csv_path,
            device=args.device,
            policy_name=args.policy_name,
            env_kwargs=env_kwargs,
            deterministic=args.deterministic,
            hidden_size=args.hidden_size,
        )


if __name__ == '__main__':
    main()
