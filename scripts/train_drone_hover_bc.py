#!/usr/bin/env python3
import argparse
import os
import time

import numpy as np
import torch

import pufferlib
import pufferlib.emulation
import pufferlib.pytorch
from pufferlib.environments.drone_hover.environment import DroneHoverEnv
from pufferlib.environments.drone_hover.torch import SquashedPolicy


def collect_dataset(env, steps, seed, policy, device, deterministic, label_mode, pd_assist_scale):
    obs_list = []
    act_list = []
    obs, _ = env.reset(seed=seed)
    for t in range(steps):
        if policy is None:
            action = np.full(env.action_space.shape, 0.5, dtype=np.float32)
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = policy.forward_eval(obs_t)
                if deterministic and hasattr(logits, "loc"):
                    action = torch.sigmoid(logits.loc)
                else:
                    action, _, _ = pufferlib.pytorch.sample_logits(logits)
            action = action.cpu().numpy().reshape(env.action_space.shape)
            action = np.clip(action, env.action_space.low, env.action_space.high)

        next_obs, _, terminated, truncated, info = env.step(action)
        pd_action = info.get("pd_base_action")
        if label_mode in ("pd_base", "applied") and pd_action is None:
            raise RuntimeError("pd_base_action missing; enable pd_label or pd_assist in env")
        if label_mode == "pd_base":
            label_action = pd_action
        elif label_mode == "applied":
            residual = (action - 0.5) * 2.0 * float(pd_assist_scale)
            label_action = np.clip(pd_action + residual, 0.0, 1.0)
        else:
            label_action = action

        obs_list.append(obs)
        act_list.append(label_action)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset(seed=seed + t + 1)
    return np.asarray(obs_list, dtype=np.float32), np.asarray(act_list, dtype=np.float32)


def train_bc(policy, obs, actions, device, batch_size, epochs, lr, loss_mode):
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.tensor(actions, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        perm = torch.randperm(obs_t.shape[0], device=device)
        total_loss = 0.0
        count = 0
        for i in range(0, obs_t.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            batch_obs = obs_t[idx]
            batch_act = act_t[idx]
            logits, _ = policy.forward_eval(batch_obs)
            mean = logits.loc
            if loss_mode == "logit":
                eps = 1e-6
                target = torch.clamp(batch_act, eps, 1.0 - eps)
                target_logit = torch.log(target) - torch.log1p(-target)
                loss = torch.mean((mean - target_logit) ** 2)
            else:
                pred = torch.sigmoid(mean)
                loss = torch.mean((pred - batch_act) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_obs.shape[0]
            count += batch_obs.shape[0]
        avg_loss = total_loss / max(count, 1)
        print(f"epoch {epoch + 1}/{epochs} loss {avg_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Behavior cloning for drone hover")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, default="action", choices=["action", "logit"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--out-dir", type=str, default="experiments")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--teacher-model", type=str, default=None)
    parser.add_argument("--teacher-pd-assist", action="store_true")
    parser.add_argument("--teacher-pd-assist-scale", type=float, default=0.0)
    parser.add_argument("--label-mode", type=str, default="pd_base",
                        choices=["pd_base", "teacher", "applied"])
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--mass-range", type=float, default=0.05)
    parser.add_argument("--inertia-range", type=float, default=0.05)
    parser.add_argument("--wind-std", type=float, default=0.0)
    parser.add_argument("--sensor-noise-std", type=float, default=0.01)
    parser.add_argument("--pd-kp-xy", type=float, default=0.6)
    parser.add_argument("--pd-kd-xy", type=float, default=0.4)
    parser.add_argument("--pd-kp-z", type=float, default=5.0)
    parser.add_argument("--pd-kd-z", type=float, default=2.5)
    parser.add_argument("--pd-kp-ang", type=float, default=8.0)
    parser.add_argument("--pd-kd-ang", type=float, default=2.0)
    args = parser.parse_args()

    env = DroneHoverEnv(
        randomize=args.randomize,
        curriculum=args.curriculum,
        hover_curriculum=False,
        pd_label=True,
        pd_assist=args.teacher_pd_assist,
        pd_assist_scale=args.teacher_pd_assist_scale,
        pd_assist_scale_final=args.teacher_pd_assist_scale,
        pd_assist_anneal_episodes=0,
        mass_range=args.mass_range,
        inertia_range=args.inertia_range,
        wind_std=args.wind_std,
        sensor_noise_std=args.sensor_noise_std,
        pd_kp_xy=args.pd_kp_xy,
        pd_kd_xy=args.pd_kd_xy,
        pd_kp_z=args.pd_kp_z,
        pd_kd_z=args.pd_kd_z,
        pd_kp_ang=args.pd_kp_ang,
        pd_kd_ang=args.pd_kd_ang,
    )

    policy = None
    if args.teacher_model:
        puffer_env = pufferlib.emulation.GymnasiumPufferEnv(env=env)
        policy = SquashedPolicy(puffer_env, hidden_size=args.hidden_size)
        policy.to(args.device)
        state = torch.load(args.teacher_model, map_location=args.device)
        state = {k.replace('module.', ''): v for k, v in state.items()}
        policy.load_state_dict(state)
        policy.eval()

    obs, actions = collect_dataset(
        env,
        args.steps,
        args.seed,
        policy=policy,
        device=args.device,
        deterministic=args.deterministic,
        label_mode=args.label_mode,
        pd_assist_scale=args.teacher_pd_assist_scale,
    )

    puffer_env = pufferlib.emulation.GymnasiumPufferEnv(env=env)
    policy = SquashedPolicy(puffer_env, hidden_size=args.hidden_size)
    policy.to(args.device)

    train_bc(policy, obs, actions, args.device, args.batch_size, args.epochs, args.lr, args.loss)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"drone_hover_bc_{int(time.time())}.pt")
    torch.save(policy.state_dict(), out_path)
    print(f"Saved model to: {out_path}")


if __name__ == "__main__":
    main()
