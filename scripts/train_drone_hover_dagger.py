#!/usr/bin/env python3
import argparse
import os
import time

import numpy as np
import torch

import pufferlib
import pufferlib.emulation
from pufferlib.environments.drone_hover.environment import DroneHoverEnv
from pufferlib.environments.drone_hover.torch import SquashedPolicy


def policy_action(policy, obs, device, deterministic):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = policy.forward_eval(obs_t)
        if deterministic and hasattr(logits, "loc"):
            action = torch.sigmoid(logits.loc)
        else:
            action, _, _ = pufferlib.pytorch.sample_logits(logits)
    return action.squeeze(0).cpu().numpy()


def collect_rollout(env, policy, steps, seed, device, deterministic, use_policy):
    obs_list = []
    act_list = []
    obs, _ = env.reset(seed=seed)
    for t in range(steps):
        if use_policy:
            action = policy_action(policy, obs, device, deterministic)
        else:
            action = np.full(env.action_space.shape, 0.5, dtype=np.float32)
        next_obs, _, terminated, truncated, info = env.step(action)
        pd_action = info.get("pd_base_action")
        if pd_action is None:
            raise RuntimeError("pd_base_action missing; enable pd_label or pd_assist in env")
        obs_list.append(obs)
        act_list.append(pd_action)
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
    parser = argparse.ArgumentParser(description="DAgger training for drone hover")
    parser.add_argument("--bootstrap-steps", type=int, default=50000)
    parser.add_argument("--rollout-steps", type=int, default=20000)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, default="action", choices=["action", "logit"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--out-dir", type=str, default="experiments")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--init-model", type=str, default=None)
    parser.add_argument("--rollout-pd-assist", action="store_true")
    parser.add_argument("--rollout-pd-assist-scale", type=float, default=0.0)
    parser.add_argument("--pd-kp-xy", type=float, default=0.6)
    parser.add_argument("--pd-kd-xy", type=float, default=0.4)
    parser.add_argument("--pd-kp-z", type=float, default=5.0)
    parser.add_argument("--pd-kd-z", type=float, default=2.5)
    parser.add_argument("--pd-kp-ang", type=float, default=8.0)
    parser.add_argument("--pd-kd-ang", type=float, default=2.0)
    args = parser.parse_args()

    pd_params = dict(
        pd_kp_xy=args.pd_kp_xy,
        pd_kd_xy=args.pd_kd_xy,
        pd_kp_z=args.pd_kp_z,
        pd_kd_z=args.pd_kd_z,
        pd_kp_ang=args.pd_kp_ang,
        pd_kd_ang=args.pd_kd_ang,
    )

    bootstrap_env = DroneHoverEnv(
        randomize=False,
        curriculum=False,
        hover_curriculum=False,
        pd_assist=True,
        pd_assist_scale=0.0,
        **pd_params,
    )
    obs_data, act_data = collect_rollout(
        bootstrap_env,
        policy=None,
        steps=args.bootstrap_steps,
        seed=args.seed,
        device=args.device,
        deterministic=args.deterministic,
        use_policy=False,
    )

    puffer_env = pufferlib.emulation.GymnasiumPufferEnv(env=bootstrap_env)
    policy = SquashedPolicy(puffer_env, hidden_size=args.hidden_size)
    policy.to(args.device)
    if args.init_model:
        state = torch.load(args.init_model, map_location=args.device)
        state = {k.replace('module.', ''): v for k, v in state.items()}
        policy.load_state_dict(state)

    train_bc(policy, obs_data, act_data, args.device, args.batch_size, args.epochs, args.lr, args.loss)

    for iteration in range(args.iters):
        rollout_env = DroneHoverEnv(
            randomize=False,
            curriculum=False,
            hover_curriculum=False,
            pd_label=True,
            pd_assist=args.rollout_pd_assist,
            pd_assist_scale=args.rollout_pd_assist_scale,
            **pd_params,
        )
        new_obs, new_act = collect_rollout(
            rollout_env,
            policy=policy,
            steps=args.rollout_steps,
            seed=args.seed + (iteration + 1) * 1000,
            device=args.device,
            deterministic=args.deterministic,
            use_policy=True,
        )
        obs_data = np.concatenate([obs_data, new_obs], axis=0)
        act_data = np.concatenate([act_data, new_act], axis=0)
        print(f"DAgger iter {iteration + 1}: dataset size {obs_data.shape[0]}")
        train_bc(policy, obs_data, act_data, args.device, args.batch_size, args.epochs, args.lr, args.loss)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"drone_hover_dagger_{int(time.time())}.pt")
    torch.save(policy.state_dict(), out_path)
    print(f"Saved model to: {out_path}")


if __name__ == "__main__":
    main()
