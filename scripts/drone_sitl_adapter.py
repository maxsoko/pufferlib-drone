#!/usr/bin/env python3
import argparse
import dataclasses
import math
import time


@dataclasses.dataclass
class LocalNedSetpoint:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw: float = 0.0
    yaw_rate: float = 0.0


@dataclasses.dataclass
class AttitudeSetpoint:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    thrust: float = 0.5


class MavlinkSitlAdapter:
    """Minimal MAVLink v2 UDP scaffold for the qualifier adapter boundary."""

    def __init__(self, endpoint, source_system=42, source_component=191):
        try:
            from pymavlink import mavutil
        except ImportError as exc:
            raise RuntimeError("Install pymavlink to use the SITL adapter scaffold") from exc

        self.mavutil = mavutil
        self.master = mavutil.mavlink_connection(
            endpoint,
            source_system=source_system,
            source_component=source_component,
            dialect="common",
            autoreconnect=True,
            force_connected=True,
        )

    def send_heartbeat(self):
        self.master.mav.heartbeat_send(
            self.mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
            self.mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            self.mavutil.mavlink.MAV_STATE_ACTIVE,
        )

    def send_local_ned_setpoint(self, target):
        # Ignore position/acceleration for the first scaffold and send bounded velocity + yaw.
        type_mask = (
            self.mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
            | self.mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
            | self.mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
            | self.mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | self.mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | self.mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
        )
        self.master.mav.set_position_target_local_ned_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,
            1,
            1,
            self.mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            target.x,
            target.y,
            target.z,
            target.vx,
            target.vy,
            target.vz,
            0.0,
            0.0,
            0.0,
            target.yaw,
            target.yaw_rate,
        )

    def send_attitude_setpoint(self, target):
        q = euler_to_quaternion(target.roll, target.pitch, target.yaw)
        self.master.mav.set_attitude_target_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,
            1,
            1,
            0,
            q,
            0.0,
            0.0,
            0.0,
            max(0.0, min(1.0, target.thrust)),
        )


def euler_to_quaternion(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def run_dry(args):
    heartbeat_period = 1.0 / args.heartbeat_hz
    command_period = 1.0 / args.command_hz
    print(
        f"dry-run endpoint={args.endpoint} heartbeat_hz={args.heartbeat_hz} "
        f"command_hz={args.command_hz} command_period={command_period:.4f}s "
        f"heartbeat_period={heartbeat_period:.4f}s"
    )


def run_constant_velocity(args):
    adapter = MavlinkSitlAdapter(args.endpoint)
    heartbeat_period = 1.0 / args.heartbeat_hz
    command_period = 1.0 / args.command_hz
    next_heartbeat = 0.0
    deadline = time.monotonic() + args.duration
    target = LocalNedSetpoint(vx=args.vx, vy=args.vy, vz=args.vz, yaw=args.yaw, yaw_rate=args.yaw_rate)

    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_heartbeat:
            adapter.send_heartbeat()
            next_heartbeat = now + heartbeat_period
        adapter.send_local_ned_setpoint(target)
        time.sleep(command_period)


def main():
    parser = argparse.ArgumentParser(description="MAVLink v2 UDP SITL scaffold for native drone policies")
    parser.add_argument("--endpoint", default="udpout:127.0.0.1:14540")
    parser.add_argument("--mode", choices=["dry-run", "constant-velocity"], default="dry-run")
    parser.add_argument("--heartbeat-hz", type=float, default=2.0)
    parser.add_argument("--command-hz", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--vz", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    args = parser.parse_args()

    if args.heartbeat_hz < 2.0:
        raise ValueError("heartbeat_hz must be >= 2.0 per qualifier spec")
    if not 50.0 <= args.command_hz <= 120.0:
        raise ValueError("command_hz must be within the qualifier 50-120 Hz command band")

    if args.mode == "dry-run":
        run_dry(args)
    else:
        run_constant_velocity(args)


if __name__ == "__main__":
    main()
