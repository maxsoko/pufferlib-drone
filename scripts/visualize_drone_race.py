#!/usr/bin/env python3
"""Interactive 3D visualizer for PufferLib drone_race environment.

Examples:
    python scripts/visualize_drone_race.py --mode scripted
    python scripts/visualize_drone_race.py --mode policy --model-path experiments/<ckpt>.pt --deterministic
"""

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch

import pufferlib
import pufferlib.emulation
import pufferlib.pytorch
from pufferlib.environments.drone_race.environment import DroneRaceEnv
from pufferlib.environments.drone_race.torch import Policy

HAVE_PYGAME = True
try:
    import pygame
    from pygame.locals import (
        DOUBLEBUF,
        OPENGL,
        QUIT,
        KEYDOWN,
        MOUSEBUTTONDOWN,
        MOUSEBUTTONUP,
        MOUSEMOTION,
        MOUSEWHEEL,
        K_ESCAPE,
        K_q,
        K_SPACE,
        K_r,
        K_t,
        K_w,
        K_a,
        K_s,
        K_d,
        K_PLUS,
        K_MINUS,
        K_EQUALS,
    )
except ImportError:
    HAVE_PYGAME = False
    pygame = None
    DOUBLEBUF = OPENGL = QUIT = KEYDOWN = 0
    MOUSEBUTTONDOWN = MOUSEBUTTONUP = MOUSEMOTION = MOUSEWHEEL = 0
    K_ESCAPE = K_q = K_SPACE = K_r = K_t = 0
    K_w = K_a = K_s = K_d = K_PLUS = K_MINUS = K_EQUALS = 0

HAVE_OPENGL = True
try:
    from OpenGL.GL import (
        glBegin,
        glBlendFunc,
        glClear,
        glClearColor,
        glColor3f,
        glColor4f,
        glDepthFunc,
        glDisable,
        glDrawPixels,
        glEnable,
        glEnd,
        glLineWidth,
        glLoadIdentity,
        glMatrixMode,
        glOrtho,
        glPointSize,
        glPopMatrix,
        glPushMatrix,
        glRasterPos2f,
        glRotatef,
        glTranslatef,
        glVertex3f,
        glViewport,
        GL_BLEND,
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        GL_LEQUAL,
        GL_LINE_LOOP,
        GL_LINE_STRIP,
        GL_LINES,
        GL_MODELVIEW,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_POINTS,
        GL_PROJECTION,
        GL_RGBA,
        GL_SRC_ALPHA,
        GL_UNSIGNED_BYTE,
    )
    from OpenGL.GLU import gluLookAt, gluPerspective
except ImportError:
    HAVE_OPENGL = False


COLOR_BG = (0.05, 0.05, 0.08)
COLOR_GRID = (0.18, 0.18, 0.22)
COLOR_GATE_NEXT = (0.2, 1.0, 0.4)
COLOR_GATE_FUTURE = (0.3, 0.5, 0.8)
COLOR_GATE_PASSED = (0.3, 0.3, 0.3)
COLOR_DRONE = (1.0, 0.35, 0.35)


def draw_ground(size=40.0, spacing=2.0):
    glColor3f(*COLOR_GRID)
    glLineWidth(1.0)
    glBegin(GL_LINES)
    n = int(size / spacing)
    for i in range(-n, n + 1):
        x = i * spacing
        glVertex3f(x, -size, 0.0)
        glVertex3f(x, size, 0.0)
        glVertex3f(-size, x, 0.0)
        glVertex3f(size, x, 0.0)
    glEnd()


def draw_gate(center, normal, gate_radius, color):
    yaw = math.degrees(math.atan2(float(normal[1]), float(normal[0])))

    glPushMatrix()
    glTranslatef(float(center[0]), float(center[1]), float(center[2]))
    glRotatef(yaw, 0.0, 0.0, 1.0)

    glColor3f(*color)
    glLineWidth(3.0)

    # Draw gate as circular aperture in YZ plane (x=0).
    glBegin(GL_LINE_LOOP)
    for deg in range(0, 360, 12):
        rad = math.radians(deg)
        glVertex3f(0.0, gate_radius * math.cos(rad), gate_radius * math.sin(rad))
    glEnd()

    # Direction arrow through gate.
    glColor4f(*color, 0.45)
    glLineWidth(1.5)
    glBegin(GL_LINES)
    glVertex3f(-0.3, 0.0, 0.0)
    glVertex3f(0.55, 0.0, 0.0)
    glVertex3f(0.55, 0.0, 0.0)
    glVertex3f(0.45, 0.06, 0.0)
    glVertex3f(0.55, 0.0, 0.0)
    glVertex3f(0.45, -0.06, 0.0)
    glEnd()

    glPopMatrix()


def draw_drone(position, yaw, color=COLOR_DRONE):
    arm = 0.18
    prop = 0.06

    glPushMatrix()
    glTranslatef(float(position[0]), float(position[1]), float(position[2]))
    glRotatef(math.degrees(float(yaw)), 0.0, 0.0, 1.0)

    glColor3f(*color)
    glLineWidth(3.0)
    glBegin(GL_LINES)
    glVertex3f(-arm, -arm, 0.0)
    glVertex3f(arm, arm, 0.0)
    glVertex3f(-arm, arm, 0.0)
    glVertex3f(arm, -arm, 0.0)
    glEnd()

    glPointSize(6.0)
    glBegin(GL_POINTS)
    glVertex3f(0.0, 0.0, 0.0)
    glEnd()

    glColor3f(1.0, 1.0, 1.0)
    glLineWidth(2.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(arm * 0.7, 0.0, 0.0)
    glEnd()

    glColor4f(*color, 0.6)
    for dx, dy in [(-arm, -arm), (-arm, arm), (arm, -arm), (arm, arm)]:
        glBegin(GL_LINE_LOOP)
        for deg in range(0, 360, 20):
            rad = math.radians(deg)
            glVertex3f(dx + prop * math.cos(rad), dy + prop * math.sin(rad), 0.0)
        glEnd()

    glPopMatrix()


def draw_trail(points, color=COLOR_DRONE):
    if len(points) < 2:
        return

    glLineWidth(1.4)
    glBegin(GL_LINE_STRIP)
    n = len(points)
    for i, p in enumerate(points):
        alpha = 0.8 * (i / max(n - 1, 1))
        glColor4f(color[0], color[1], color[2], alpha)
        glVertex3f(float(p[0]), float(p[1]), float(p[2]))
    glEnd()


class DroneRaceVisualizer:
    def __init__(
        self,
        mode,
        model_path,
        hidden_size,
        deterministic,
        device,
        env_kwargs,
        seed,
        width,
        height,
        fps,
        speed,
        trail_len,
    ):
        self.mode = mode
        self.model_path = model_path
        self.hidden_size = hidden_size
        self.deterministic = deterministic
        self.device = device
        self.seed = seed
        self.width = width
        self.height = height
        self.fps_limit = fps
        self.speed = max(0.05, float(speed))
        self._step_budget = 0.0

        self.env = DroneRaceEnv(**env_kwargs)
        self.obs, self.info = self.env.reset(seed=seed)
        self.episode = 0
        self.step = 0
        self.last_reward = 0.0

        self.policy = None
        if self.mode == "policy":
            self.policy = self._load_policy()

        self.show_trails = True
        self.paused = False

        self.cam_distance = 16.0
        self.cam_azimuth = 25.0
        self.cam_elevation = 45.0
        self.cam_target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.mouse_dragging = False
        self.mouse_last = (0, 0)

        self.trail = deque(maxlen=int(trail_len))

        self.frame_count = 0
        self.render_fps = 0.0
        self.last_fps_time = time.time()
        self._font_cache = {}

    def _load_policy(self):
        puffer_env = pufferlib.emulation.GymnasiumPufferEnv(env=self.env)
        policy = Policy(puffer_env, hidden_size=self.hidden_size)
        policy.to(self.device)
        policy.eval()

        state_dict = torch.load(self.model_path, map_location=self.device)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)
        return policy

    def _select_action(self, obs):
        if self.mode == "scripted":
            return self.env.scripted_action()

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.policy.forward_eval(obs_t)
            if self.deterministic and hasattr(logits, "loc"):
                action = logits.loc
            elif self.deterministic and isinstance(logits, torch.Tensor):
                action = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                action, _, _ = pufferlib.pytorch.sample_logits(logits)

        action = action.detach().cpu().numpy().reshape(self.env.action_space.shape)
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)

    def _init_gl(self):
        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("PufferLib DroneRace Visualizer")

        glClearColor(*COLOR_BG, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glViewport(0, 0, self.width, self.height)

    def _setup_camera(self):
        drone_pos = self.env.position.astype(np.float32)
        self.cam_target = 0.9 * self.cam_target + 0.1 * np.array(
            [drone_pos[0], drone_pos[1], max(0.2, drone_pos[2])], dtype=np.float32
        )

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, self.width / self.height, 0.1, 120.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        az = math.radians(self.cam_azimuth)
        el = math.radians(self.cam_elevation)
        cx = self.cam_target[0] + self.cam_distance * math.cos(el) * math.cos(az)
        cy = self.cam_target[1] + self.cam_distance * math.cos(el) * math.sin(az)
        cz = self.cam_target[2] + self.cam_distance * math.sin(el)

        gluLookAt(
            cx,
            cy,
            cz,
            self.cam_target[0],
            self.cam_target[1],
            self.cam_target[2],
            0.0,
            0.0,
            1.0,
        )

    def _draw_text_2d(self, text, x, y, color=(220, 220, 220), size=16):
        if size not in self._font_cache:
            self._font_cache[size] = pygame.font.SysFont("menlo", size, bold=True)

        font = self._font_cache[size]
        surface = font.render(text, True, color)
        text_data = pygame.image.tobytes(surface, "RGBA", True)
        w, h = surface.get_size()

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glRasterPos2f(float(x), float(y))
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                return False

            if event.type == KEYDOWN:
                if event.key in (K_ESCAPE, K_q):
                    return False
                if event.key == K_SPACE:
                    self.paused = not self.paused
                if event.key == K_t:
                    self.show_trails = not self.show_trails
                if event.key == K_r:
                    self.obs, self.info = self.env.reset(seed=self.seed + self.episode + 1)
                    self.trail.clear()
                if event.key in (K_PLUS, K_EQUALS):
                    self.speed = min(16.0, self.speed * 2.0)
                if event.key == K_MINUS:
                    self.speed = max(0.05, self.speed / 2.0)

            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                self.mouse_dragging = True
                self.mouse_last = event.pos

            if event.type == MOUSEBUTTONUP and event.button == 1:
                self.mouse_dragging = False

            if event.type == MOUSEMOTION and self.mouse_dragging:
                dx = event.pos[0] - self.mouse_last[0]
                dy = event.pos[1] - self.mouse_last[1]
                self.cam_azimuth -= dx * 0.3
                self.cam_elevation = float(np.clip(self.cam_elevation + dy * 0.3, 5.0, 85.0))
                self.mouse_last = event.pos

            if event.type == MOUSEWHEEL:
                self.cam_distance = float(np.clip(self.cam_distance - event.y * 0.8, 2.0, 70.0))

        keys = pygame.key.get_pressed()
        pan_speed = 0.1
        az = math.radians(self.cam_azimuth)
        if keys[K_w]:
            self.cam_target[0] += pan_speed * math.cos(az)
            self.cam_target[1] += pan_speed * math.sin(az)
        if keys[K_s]:
            self.cam_target[0] -= pan_speed * math.cos(az)
            self.cam_target[1] -= pan_speed * math.sin(az)
        if keys[K_a]:
            self.cam_target[0] += pan_speed * math.sin(az)
            self.cam_target[1] -= pan_speed * math.cos(az)
        if keys[K_d]:
            self.cam_target[0] -= pan_speed * math.sin(az)
            self.cam_target[1] += pan_speed * math.cos(az)

        return True

    def _step_once(self):
        action = self._select_action(self.obs)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs
        self.last_reward = float(reward)
        self.info = info
        self.step += 1
        self.trail.append(self.env.position.copy())

        if terminated or truncated:
            status = "success" if info.get("valid_run", 0) == 1 and info.get("gates_passed", 0) == info.get("gates_total", 1) else "end"
            print(
                f"episode={self.episode} status={status} valid={info.get('valid_run', 0)} "
                f"gates={info.get('gates_passed', 0)}/{info.get('gates_total', 0)} "
                f"time={info.get('elapsed_time', 0.0):.2f}s"
            )
            self.episode += 1
            self.step = 0
            self.obs, self.info = self.env.reset(seed=self.seed + self.episode)
            self.trail.clear()

    def _step_sim(self):
        if self.paused:
            return

        self._step_budget += self.speed
        step_count = 0
        while self._step_budget >= 1.0 and step_count < 32:
            self._step_once()
            self._step_budget -= 1.0
            step_count += 1

    def _render_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._setup_camera()

        draw_ground()

        target_gate = int(self.env.current_gate_index)
        for i in range(self.env.gates_total):
            if i < target_gate:
                color = COLOR_GATE_PASSED
            elif i == target_gate:
                color = COLOR_GATE_NEXT
            else:
                color = COLOR_GATE_FUTURE

            draw_gate(
                center=self.env.gate_centers[i],
                normal=self.env.gate_normals[i],
                gate_radius=float(self.env.gate_radius),
                color=color,
            )

        if self.show_trails:
            draw_trail(self.trail)

        draw_drone(self.env.position, self.env.angles[2])

    def _render_hud(self):
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time > 1.0:
            self.render_fps = self.frame_count / (now - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = now

        speed = float(np.linalg.norm(self.env.velocity))
        gate_idx = int(self.info.get("gate_index", self.env.current_gate_index))
        gates_passed = int(self.info.get("gates_passed", 0))
        gates_total = int(self.info.get("gates_total", self.env.gates_total))

        y = self.height - 26
        lh = 20
        self._draw_text_2d("PufferLib DroneRace Visualizer", 12, y, color=(110, 255, 170), size=18)
        y -= lh
        self._draw_text_2d(
            f"mode={self.mode} fps={self.render_fps:.0f} sim_speed={self.speed:.2f}x paused={int(self.paused)}",
            12,
            y,
            color=(190, 190, 190),
            size=14,
        )
        y -= lh
        self._draw_text_2d(
            f"episode={self.episode} step={self.step} reward={self.last_reward:.3f} speed={speed:.2f}m/s",
            12,
            y,
            color=(220, 220, 220),
            size=15,
        )
        y -= lh
        self._draw_text_2d(
            f"gate={gate_idx} passed={gates_passed}/{gates_total} valid={int(self.info.get('valid_run', 0))} "
            f"time={float(self.info.get('elapsed_time', 0.0)):.2f}s",
            12,
            y,
            color=(220, 220, 220),
            size=15,
        )
        self._draw_text_2d(
            "Controls: mouse-drag orbit, scroll zoom, WASD pan, T trails, +/- speed, R reset, SPACE pause, Q quit",
            12,
            10,
            color=(110, 110, 120),
            size=12,
        )

    def run(self):
        self._init_gl()
        print("\n=== PufferLib DroneRace Visualizer ===")
        print(f"mode={self.mode} model={self.model_path or 'none'}")
        print("controls: mouse orbit/zoom, WASD pan, T trails, +/- speed, R reset, SPACE pause, Q quit\n")

        clock = pygame.time.Clock()
        running = True
        while running:
            running = self._handle_events()
            self._step_sim()
            self._render_scene()
            self._render_hud()
            pygame.display.flip()
            clock.tick(self.fps_limit)

        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Visualize PufferLib drone_race in real-time")
    parser.add_argument("--mode", choices=["scripted", "policy"], default="scripted")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--env-kwargs", type=str, default="{}", help="JSON dict passed to DroneRaceEnv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation steps per render frame")
    parser.add_argument("--trail-len", type=int, default=240)
    args = parser.parse_args()

    if args.mode == "policy" and not args.model_path:
        raise SystemExit("Policy mode requires --model-path")

    if not HAVE_PYGAME:
        raise SystemExit("pygame not installed. Install with: pip install pygame")
    if not HAVE_OPENGL:
        raise SystemExit("PyOpenGL not installed. Install with: pip install PyOpenGL PyOpenGL_accelerate")

    try:
        env_kwargs = json.loads(args.env_kwargs)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --env-kwargs JSON: {exc}") from exc

    visualizer = DroneRaceVisualizer(
        mode=args.mode,
        model_path=args.model_path,
        hidden_size=args.hidden_size,
        deterministic=args.deterministic,
        device=args.device,
        env_kwargs=env_kwargs,
        seed=args.seed,
        width=args.width,
        height=args.height,
        fps=args.fps,
        speed=args.speed,
        trail_len=args.trail_len,
    )
    visualizer.run()


if __name__ == "__main__":
    main()
