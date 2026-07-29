#!/usr/bin/env python3
import argparse
import dataclasses
import json
import math
import os
import struct
import time


MAVLINK_CMD_SIM_RESET = 31000
ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID = 2


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
    body_roll_rate: float = 0.0
    body_pitch_rate: float = 0.0
    body_yaw_rate: float = 0.0
    thrust: float = 0.5


@dataclasses.dataclass(frozen=True)
class RaceStatus:
    sim_boot_time_ms: int
    race_start_boot_time_ms: int
    race_finish_time_ns: int
    active_gate_index: int
    last_gate_race_time: int


@dataclasses.dataclass(frozen=True)
class TrackGateInfo:
    gate_id: int
    position_ned_x: float
    position_ned_y: float
    position_ned_z: float
    orientation_ned_w: float
    orientation_ned_x: float
    orientation_ned_y: float
    orientation_ned_z: float
    width_m: float
    height_m: float


@dataclasses.dataclass
class TelemetryState:
    last_message_monotonic_s: float | None = None
    last_heartbeat_monotonic_s: float | None = None
    last_attitude_monotonic_s: float | None = None
    last_imu_monotonic_s: float | None = None
    last_timesync_monotonic_s: float | None = None
    system_status: int | None = None
    base_mode: int | None = None
    custom_mode: int | None = None
    roll: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    rollspeed: float | None = None
    pitchspeed: float | None = None
    yawspeed: float | None = None
    attitude_time_boot_ms: int | None = None
    imu_time_usec: int | None = None
    xacc: float | None = None
    yacc: float | None = None
    zacc: float | None = None
    xgyro: float | None = None
    ygyro: float | None = None
    zgyro: float | None = None
    xmag: float | None = None
    ymag: float | None = None
    zmag: float | None = None
    abs_pressure: float | None = None
    diff_pressure: float | None = None
    pressure_alt: float | None = None
    temperature: float | None = None
    fields_updated: int | None = None
    timesync_tc1: int | None = None
    timesync_ts1: int | None = None
    linear_velocity_m_s: tuple[float, float, float] | None = None
    local_position_ned_m: tuple[float, float, float] | None = None
    local_velocity_ned_m_s: tuple[float, float, float] | None = None
    odometry_position_ned_m: tuple[float, float, float] | None = None
    odometry_quaternion_wxyz: tuple[float, float, float, float] | None = None
    odometry_velocity_ned_m_s: tuple[float, float, float] | None = None
    actuator_outputs: tuple[float, ...] | None = None
    collision_id: int | None = None
    collision_threat_level: int | None = None
    collision_impact: float | None = None
    race_status: RaceStatus | None = None
    track_gates: tuple[TrackGateInfo, ...] = ()


@dataclasses.dataclass
class TelemetryMetrics:
    messages_seen: int = 0
    malformed_messages: int = 0
    unknown_messages: int = 0
    heartbeats: int = 0
    attitudes: int = 0
    highres_imus: int = 0
    timesyncs: int = 0
    local_positions: int = 0
    odometries: int = 0
    actuator_outputs: int = 0
    collisions: int = 0
    race_statuses: int = 0
    track_infos: int = 0
    data_handshakes: int = 0
    encapsulated_data: int = 0
    receive_timeouts: int = 0
    telemetry_dropouts: int = 0
    drain_calls: int = 0
    messages_drained: int = 0
    max_drain_batch: int = 0
    drain_limit_hits: int = 0
    last_message_age_s: float | None = None
    last_attitude_age_s: float | None = None
    last_imu_age_s: float | None = None
    last_timesync_age_s: float | None = None


@dataclasses.dataclass
class SitlRunReport:
    endpoint: str
    mode: str
    duration_s: float
    heartbeat_hz: float
    command_hz: float
    command_kind: str
    heartbeats_sent: int = 0
    commands_sent: int = 0
    command_rate_violations: int = 0
    effective_command_hz: float = 0.0
    command_publication_duration_s: float = 0.0
    telemetry: TelemetryMetrics = dataclasses.field(default_factory=TelemetryMetrics)
    latest_telemetry: TelemetryState = dataclasses.field(default_factory=TelemetryState)

    def to_dict(self):
        return dataclasses.asdict(self)


class MavlinkTelemetryParser:
    """Track the TS-002 telemetry subset without depending on pymavlink internals."""

    SUPPORTED_TYPES = {
        "HEARTBEAT",
        "ATTITUDE",
        "HIGHRES_IMU",
        "TIMESYNC",
        "LOCAL_POSITION_NED",
        "ODOMETRY",
        "ACTUATOR_OUTPUT_STATUS",
        "COLLISION",
        "DATA_TRANSMISSION_HANDSHAKE",
        "ENCAPSULATED_DATA",
    }

    def __init__(self, *, dropout_after_s=1.0):
        if dropout_after_s <= 0.0:
            raise ValueError("dropout_after_s must be positive")
        self.dropout_after_s = dropout_after_s
        self.state = TelemetryState()
        self.metrics = TelemetryMetrics()
        self._dropout_open = False
        self._track_chunks: dict[int, dict[int, bytes]] = {}
        self._expected_track_chunks: dict[int, int] = {}

    def ingest(self, message, *, now_s=None):
        now = time.monotonic() if now_s is None else now_s
        msg_type = message_type(message)
        if msg_type is None:
            self.metrics.malformed_messages += 1
            return None

        self.metrics.messages_seen += 1
        self.state.last_message_monotonic_s = now
        self._dropout_open = False

        if msg_type == "HEARTBEAT":
            self.metrics.heartbeats += 1
            self.state.last_heartbeat_monotonic_s = now
            self.state.system_status = get_message_attr(message, "system_status")
            self.state.base_mode = get_message_attr(message, "base_mode")
            self.state.custom_mode = get_message_attr(message, "custom_mode")
        elif msg_type == "ATTITUDE":
            self.metrics.attitudes += 1
            self.state.last_attitude_monotonic_s = now
            self.state.attitude_time_boot_ms = get_message_attr(message, "time_boot_ms")
            self.state.roll = get_message_attr(message, "roll")
            self.state.pitch = get_message_attr(message, "pitch")
            self.state.yaw = get_message_attr(message, "yaw")
            self.state.rollspeed = get_message_attr(message, "rollspeed")
            self.state.pitchspeed = get_message_attr(message, "pitchspeed")
            self.state.yawspeed = get_message_attr(message, "yawspeed")
        elif msg_type == "HIGHRES_IMU":
            self.metrics.highres_imus += 1
            self.state.last_imu_monotonic_s = now
            self.state.imu_time_usec = get_message_attr(message, "time_usec")
            self.state.xacc = get_message_attr(message, "xacc")
            self.state.yacc = get_message_attr(message, "yacc")
            self.state.zacc = get_message_attr(message, "zacc")
            self.state.xgyro = get_message_attr(message, "xgyro")
            self.state.ygyro = get_message_attr(message, "ygyro")
            self.state.zgyro = get_message_attr(message, "zgyro")
            self.state.xmag = get_message_attr(message, "xmag")
            self.state.ymag = get_message_attr(message, "ymag")
            self.state.zmag = get_message_attr(message, "zmag")
            self.state.abs_pressure = get_message_attr(message, "abs_pressure")
            self.state.diff_pressure = get_message_attr(message, "diff_pressure")
            self.state.pressure_alt = get_message_attr(message, "pressure_alt")
            self.state.temperature = get_message_attr(message, "temperature")
            self.state.fields_updated = get_message_attr(message, "fields_updated")
        elif msg_type == "TIMESYNC":
            self.metrics.timesyncs += 1
            self.state.last_timesync_monotonic_s = now
            self.state.timesync_tc1 = get_message_attr(message, "tc1")
            self.state.timesync_ts1 = get_message_attr(message, "ts1")
        elif msg_type == "LOCAL_POSITION_NED":
            self.metrics.local_positions += 1
            self.state.local_position_ned_m = (
                get_message_attr(message, "x"),
                get_message_attr(message, "y"),
                get_message_attr(message, "z"),
            )
            self.state.local_velocity_ned_m_s = (
                get_message_attr(message, "vx"),
                get_message_attr(message, "vy"),
                get_message_attr(message, "vz"),
            )
            self.state.linear_velocity_m_s = self.state.local_velocity_ned_m_s
        elif msg_type == "ODOMETRY":
            self.metrics.odometries += 1
            q = get_message_attr(message, "q", None)
            self.state.odometry_position_ned_m = (
                get_message_attr(message, "x"),
                get_message_attr(message, "y"),
                get_message_attr(message, "z"),
            )
            if q is not None and len(q) >= 4:
                self.state.odometry_quaternion_wxyz = (q[0], q[1], q[2], q[3])
            self.state.odometry_velocity_ned_m_s = (
                get_message_attr(message, "vx"),
                get_message_attr(message, "vy"),
                get_message_attr(message, "vz"),
            )
            self.state.linear_velocity_m_s = self.state.odometry_velocity_ned_m_s
        elif msg_type == "ACTUATOR_OUTPUT_STATUS":
            self.metrics.actuator_outputs += 1
            actuator = get_message_attr(message, "actuator", None)
            if actuator is not None:
                self.state.actuator_outputs = tuple(float(value) for value in actuator)
        elif msg_type == "COLLISION":
            self.metrics.collisions += 1
            self.state.collision_id = get_message_attr(message, "id")
            self.state.collision_threat_level = get_message_attr(message, "threat_level")
            self.state.collision_impact = get_message_attr(message, "horizontal_minimum_delta")
        elif msg_type == "DATA_TRANSMISSION_HANDSHAKE":
            self.metrics.data_handshakes += 1
            transfer_id = get_message_attr(message, "width")
            packets = get_message_attr(message, "packets")
            if transfer_id is not None and packets is not None:
                self._track_chunks[int(transfer_id)] = {}
                self._expected_track_chunks[int(transfer_id)] = int(packets)
        elif msg_type == "ENCAPSULATED_DATA":
            self.metrics.encapsulated_data += 1
            self._ingest_encapsulated_data(message)
        else:
            self.metrics.unknown_messages += 1

        self.update_ages(now_s=now)
        return msg_type

    def _ingest_encapsulated_data(self, message) -> None:
        raw_payload = bytes(get_message_attr(message, "data", b""))
        if not raw_payload:
            return

        data_type = int(raw_payload[0])
        if data_type == ENCAPSULATED_RACE_STATUS_MSG_ID:
            self._ingest_race_status(raw_payload)
        elif data_type == ENCAPSULATED_TRACK_INFO_MSG_ID:
            self._ingest_track_data_packet(message, raw_payload)

    def _ingest_race_status(self, raw_payload: bytes) -> None:
        fmt = "<BQqqIq"
        if len(raw_payload) < struct.calcsize(fmt):
            self.metrics.malformed_messages += 1
            return
        (
            _data_type,
            sim_boot_time_ms,
            race_start_boot_time_ms,
            race_finish_time_ns,
            active_gate_index,
            last_gate_race_time,
        ) = struct.unpack_from(fmt, raw_payload)
        self.metrics.race_statuses += 1
        self.state.race_status = RaceStatus(
            sim_boot_time_ms=int(sim_boot_time_ms),
            race_start_boot_time_ms=int(race_start_boot_time_ms),
            race_finish_time_ns=int(race_finish_time_ns),
            active_gate_index=int(active_gate_index),
            last_gate_race_time=int(last_gate_race_time),
        )

    def _ingest_track_data_packet(self, message, raw_payload: bytes) -> None:
        header_fmt = "<BH"
        if len(raw_payload) < struct.calcsize(header_fmt):
            self.metrics.malformed_messages += 1
            return
        _data_type, transfer_id = struct.unpack_from(header_fmt, raw_payload)
        transfer_id = int(transfer_id)
        if transfer_id not in self._expected_track_chunks:
            return
        seqnr = get_message_attr(message, "seqnr")
        if seqnr is None:
            self.metrics.malformed_messages += 1
            return
        self._track_chunks[transfer_id][int(seqnr)] = raw_payload[struct.calcsize(header_fmt):]
        expected = self._expected_track_chunks[transfer_id]
        if len(self._track_chunks[transfer_id]) != expected:
            return

        if any(i not in self._track_chunks[transfer_id] for i in range(expected)):
            self.metrics.malformed_messages += 1
            return
        payload = b"".join(self._track_chunks[transfer_id][i] for i in range(expected))
        del self._track_chunks[transfer_id]
        del self._expected_track_chunks[transfer_id]
        self._ingest_track_data(payload)

    def _ingest_track_data(self, payload: bytes) -> None:
        count_fmt = "<H"
        gate_fmt = "<Hfffffffff"
        count_size = struct.calcsize(count_fmt)
        gate_size = struct.calcsize(gate_fmt)
        if len(payload) < count_size:
            self.metrics.malformed_messages += 1
            return
        num_gates, = struct.unpack_from(count_fmt, payload)
        offset = count_size
        gates = []
        for _idx in range(int(num_gates)):
            if len(payload) < offset + gate_size:
                self.metrics.malformed_messages += 1
                return
            values = struct.unpack_from(gate_fmt, payload, offset)
            offset += gate_size
            gates.append(
                TrackGateInfo(
                    gate_id=int(values[0]),
                    position_ned_x=float(values[1]),
                    position_ned_y=float(values[2]),
                    position_ned_z=float(values[3]),
                    orientation_ned_w=float(values[4]),
                    orientation_ned_x=float(values[5]),
                    orientation_ned_y=float(values[6]),
                    orientation_ned_z=float(values[7]),
                    width_m=float(values[8]),
                    height_m=float(values[9]),
                )
            )
        self.metrics.track_infos += 1
        self.state.track_gates = tuple(gates)

    def note_receive_timeout(self, *, now_s=None):
        self.metrics.receive_timeouts += 1
        self.update_ages(now_s=now_s)

    def update_ages(self, *, now_s=None):
        now = time.monotonic() if now_s is None else now_s
        self.metrics.last_message_age_s = age(now, self.state.last_message_monotonic_s)
        self.metrics.last_attitude_age_s = age(now, self.state.last_attitude_monotonic_s)
        self.metrics.last_imu_age_s = age(now, self.state.last_imu_monotonic_s)
        self.metrics.last_timesync_age_s = age(now, self.state.last_timesync_monotonic_s)
        if (
            self.metrics.last_message_age_s is not None
            and self.metrics.last_message_age_s > self.dropout_after_s
            and not self._dropout_open
        ):
            self.metrics.telemetry_dropouts += 1
            self._dropout_open = True


class MavlinkSitlAdapter:
    """Minimal MAVLink v2 UDP scaffold for the qualifier adapter boundary."""

    def __init__(self, endpoint, source_system=42, source_component=191, *, dropout_after_s=1.0):
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
        self.telemetry = MavlinkTelemetryParser(dropout_after_s=dropout_after_s)

    def _target_ids(self) -> tuple[int, int]:
        """Return learned target IDs while preserving valid component zero."""
        target_system = self.master.target_system
        target_component = self.master.target_component
        return (
            1 if target_system is None or int(target_system) == 0 else int(target_system),
            0 if target_component is None else int(target_component),
        )

    def send_heartbeat(self):
        self.master.mav.heartbeat_send(
            self.mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
            self.mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            self.mavutil.mavlink.MAV_STATE_ACTIVE,
        )

    def send_arm_command(self):
        target_system, target_component = self._target_ids()
        self.master.mav.command_long_send(
            target_system,
            target_component,
            self.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def send_disarm_command(self):
        target_system, target_component = self._target_ids()
        self.master.mav.command_long_send(
            target_system,
            target_component,
            self.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def send_sim_reset_command(self):
        target_system, target_component = self._target_ids()
        self.master.mav.command_long_send(
            target_system,
            target_component,
            MAVLINK_CMD_SIM_RESET,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def send_timesync_request(self):
        # MAVLink TIMESYNC requests put the requesting clock in ts1 and use
        # tc1 == 0 to distinguish the packet from a response.
        self.master.mav.timesync_send(0, time.time_ns())

    def close(self):
        close_fn = getattr(self.master, "close", None)
        if close_fn is not None:
            close_fn()

    def send_local_ned_setpoint(self, target, *, frame: str = "local_ned", yaw_mode: str = "yaw_and_rate"):
        # Ignore position/acceleration for the first scaffold and send bounded velocity + yaw.
        if frame == "local_ned":
            mav_frame = self.mavutil.mavlink.MAV_FRAME_LOCAL_NED
        elif frame == "body_ned":
            mav_frame = self.mavutil.mavlink.MAV_FRAME_BODY_NED
        else:
            raise ValueError("frame must be 'local_ned' or 'body_ned'")
        if yaw_mode not in {"yaw_and_rate", "ignore"}:
            raise ValueError("yaw_mode must be 'yaw_and_rate' or 'ignore'")
        type_mask = (
            self.mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
            | self.mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
            | self.mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
            | self.mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | self.mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | self.mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
        )
        if yaw_mode == "ignore":
            type_mask = (
                type_mask
                | self.mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
                | self.mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
            )
        target_system, target_component = self._target_ids()
        self.master.mav.set_position_target_local_ned_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,
            target_system,
            target_component,
            mav_frame,
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

    def _attitude_type_mask(self, mode: str) -> int:
        if mode not in {"body_rates", "attitude", "attitude_and_rates"}:
            raise ValueError("mode must be 'body_rates', 'attitude', or 'attitude_and_rates'")

        mavlink = self.mavutil.mavlink
        if mode == "body_rates":
            return getattr(mavlink, "ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE", 128)
        if mode == "attitude":
            return (
                getattr(mavlink, "ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE", 1)
                | getattr(mavlink, "ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE", 2)
                | getattr(mavlink, "ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE", 4)
            )
        return 0

    def send_attitude_setpoint(self, target, *, mode: str = "body_rates"):
        type_mask = self._attitude_type_mask(mode)
        q = [1.0, 0.0, 0.0, 0.0] if mode == "body_rates" else euler_to_quaternion(
            target.roll,
            target.pitch,
            target.yaw,
        )
        target_system, target_component = self._target_ids()
        self.master.mav.set_attitude_target_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,
            target_system,
            target_component,
            type_mask,
            q,
            target.body_roll_rate,
            target.body_pitch_rate,
            target.body_yaw_rate,
            max(0.0, min(1.0, target.thrust)),
        )

    def _ingest_telemetry_message(self, message, *, reply_timesync=True):
        msg_type = self.telemetry.ingest(message)
        if reply_timesync and msg_type == "TIMESYNC" and get_message_attr(message, "tc1") == 0:
            self.master.mav.timesync_send(time.time_ns(), get_message_attr(message, "ts1") or 0)
        return msg_type

    def poll_telemetry(self, *, timeout_s=0.0, reply_timesync=True):
        message = self.master.recv_match(blocking=timeout_s > 0.0, timeout=timeout_s)
        if message is None:
            self.telemetry.note_receive_timeout()
            return None
        return self._ingest_telemetry_message(message, reply_timesync=reply_timesync)

    def drain_telemetry(self, *, timeout_s=0.0, max_messages=512, reply_timesync=True):
        """Consume the receive backlog and leave state at the newest sample.

        The simulator publishes several hundred telemetry messages per second.
        Reading only one message per controller tick makes attitude feedback
        seconds old even though UDP itself is healthy.  The first receive may
        block for ``timeout_s``; all remaining receives are non-blocking.
        """
        if max_messages <= 0:
            raise ValueError("max_messages must be positive")

        message = self.master.recv_match(blocking=timeout_s > 0.0, timeout=timeout_s)
        if message is None:
            self.telemetry.note_receive_timeout()
            return 0

        consumed = 0
        while message is not None and consumed < max_messages:
            self._ingest_telemetry_message(message, reply_timesync=reply_timesync)
            consumed += 1
            if consumed < max_messages:
                message = self.master.recv_match(blocking=False, timeout=0.0)
        metrics = self.telemetry.metrics
        metrics.drain_calls += 1
        metrics.messages_drained += consumed
        metrics.max_drain_batch = max(metrics.max_drain_batch, consumed)
        if consumed >= max_messages:
            metrics.drain_limit_hits += 1
        return consumed


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


def age(now_s, then_s):
    return None if then_s is None else max(0.0, now_s - then_s)


def get_message_attr(message, key, default=None):
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def message_type(message):
    if isinstance(message, dict):
        return message.get("type")
    get_type = getattr(message, "get_type", None)
    if get_type is None:
        return None
    return get_type()


def validate_rates(heartbeat_hz, command_hz):
    if heartbeat_hz < 2.0:
        raise ValueError("heartbeat_hz must be >= 2.0 per qualifier spec")
    if not 50.0 <= command_hz < 100.0:
        raise ValueError("command_hz must be >=50 Hz and <100 Hz per TS-002")


def write_report(path, report):
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, sort_keys=True)


def run_dry(args):
    heartbeat_period = 1.0 / args.heartbeat_hz
    command_period = 1.0 / args.command_hz
    report = SitlRunReport(
        endpoint=args.endpoint,
        mode=args.mode,
        duration_s=0.0,
        heartbeat_hz=args.heartbeat_hz,
        command_hz=args.command_hz,
        command_kind=args.command_kind,
    )
    print(
        f"dry-run endpoint={args.endpoint} heartbeat_hz={args.heartbeat_hz} "
        f"command_hz={args.command_hz} command_period={command_period:.4f}s "
        f"heartbeat_period={heartbeat_period:.4f}s"
    )
    write_report(args.json_path, report)


def make_report(args, adapter, mode, command_kind, started_s, heartbeats_sent, commands_sent, violations):
    adapter.telemetry.update_ages()
    return SitlRunReport(
        endpoint=args.endpoint,
        mode=mode,
        duration_s=round(time.monotonic() - started_s, 6),
        heartbeat_hz=args.heartbeat_hz,
        command_hz=args.command_hz,
        command_kind=command_kind,
        heartbeats_sent=heartbeats_sent,
        commands_sent=commands_sent,
        command_rate_violations=violations,
        telemetry=adapter.telemetry.metrics,
        latest_telemetry=adapter.telemetry.state,
    )


def run_monitor(args):
    adapter = MavlinkSitlAdapter(args.endpoint, dropout_after_s=args.telemetry_dropout_s)
    heartbeat_period = 1.0 / args.heartbeat_hz
    next_heartbeat = 0.0
    heartbeats_sent = 0
    started = time.monotonic()
    deadline = started + args.duration

    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_heartbeat:
            adapter.send_heartbeat()
            heartbeats_sent += 1
            next_heartbeat = now + heartbeat_period
        adapter.poll_telemetry(timeout_s=args.telemetry_timeout_s)
        if args.telemetry_timeout_s <= 0.0:
            time.sleep(args.idle_sleep_s)

    report = make_report(args, adapter, args.mode, "none", started, heartbeats_sent, 0, 0)
    write_report(args.json_path, report)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


def run_constant_velocity(args):
    adapter = MavlinkSitlAdapter(args.endpoint, dropout_after_s=args.telemetry_dropout_s)
    heartbeat_period = 1.0 / args.heartbeat_hz
    command_period = 1.0 / args.command_hz
    next_heartbeat = 0.0
    next_command = 0.0
    started = time.monotonic()
    deadline = started + args.duration
    heartbeats_sent = 0
    commands_sent = 0
    command_rate_violations = 0
    last_command_sent_s = None
    target = LocalNedSetpoint(vx=args.vx, vy=args.vy, vz=args.vz, yaw=args.yaw, yaw_rate=args.yaw_rate)

    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_heartbeat:
            adapter.send_heartbeat()
            heartbeats_sent += 1
            next_heartbeat = now + heartbeat_period
        if now >= next_command:
            if last_command_sent_s is not None and now - last_command_sent_s < 0.01:
                command_rate_violations += 1
            adapter.send_local_ned_setpoint(
                target,
                frame=args.command_frame,
                yaw_mode=args.command_yaw_mode,
            )
            commands_sent += 1
            last_command_sent_s = now
            next_command = now + command_period
        adapter.poll_telemetry(timeout_s=args.telemetry_timeout_s)
        if args.telemetry_timeout_s <= 0.0:
            time.sleep(args.idle_sleep_s)

    report = make_report(
        args,
        adapter,
        args.mode,
        f"{args.command_frame}_{args.command_yaw_mode}_velocity",
        started,
        heartbeats_sent,
        commands_sent,
        command_rate_violations,
    )
    write_report(args.json_path, report)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


def run_constant_attitude(args):
    adapter = MavlinkSitlAdapter(args.endpoint, dropout_after_s=args.telemetry_dropout_s)
    heartbeat_period = 1.0 / args.heartbeat_hz
    command_period = 1.0 / args.command_hz
    next_heartbeat = 0.0
    next_command = 0.0
    started = time.monotonic()
    deadline = started + args.duration
    heartbeats_sent = 0
    commands_sent = 0
    command_rate_violations = 0
    last_command_sent_s = None
    target = AttitudeSetpoint(
        roll=args.roll,
        pitch=args.pitch,
        yaw=args.yaw,
        body_roll_rate=args.roll_rate,
        body_pitch_rate=args.pitch_rate,
        body_yaw_rate=args.yaw_rate,
        thrust=args.thrust,
    )

    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_heartbeat:
            adapter.send_heartbeat()
            heartbeats_sent += 1
            next_heartbeat = now + heartbeat_period
        if now >= next_command:
            if last_command_sent_s is not None and now - last_command_sent_s < 0.01:
                command_rate_violations += 1
            adapter.send_attitude_setpoint(target, mode=args.attitude_mode)
            commands_sent += 1
            last_command_sent_s = now
            next_command = now + command_period
        adapter.poll_telemetry(timeout_s=args.telemetry_timeout_s)
        if args.telemetry_timeout_s <= 0.0:
            time.sleep(args.idle_sleep_s)

    report = make_report(
        args,
        adapter,
        args.mode,
        f"{args.attitude_mode}_attitude_target",
        started,
        heartbeats_sent,
        commands_sent,
        command_rate_violations,
    )
    write_report(args.json_path, report)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description="MAVLink v2 UDP SITL scaffold for native drone policies")
    parser.add_argument("--endpoint", default="udpout:127.0.0.1:14540")
    parser.add_argument(
        "--mode",
        choices=["dry-run", "monitor", "constant-velocity", "constant-attitude"],
        default="dry-run",
    )
    parser.add_argument("--heartbeat-hz", type=float, default=2.0)
    parser.add_argument("--command-hz", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--telemetry-timeout-s", type=float, default=0.0)
    parser.add_argument("--telemetry-dropout-s", type=float, default=1.0)
    parser.add_argument("--idle-sleep-s", type=float, default=0.001)
    parser.add_argument("--json-path", default="")
    parser.add_argument("--command-kind", choices=["local_ned_velocity"], default="local_ned_velocity")
    parser.add_argument("--command-frame", choices=["body_ned", "local_ned"], default="local_ned")
    parser.add_argument("--command-yaw-mode", choices=["yaw_and_rate", "ignore"], default="yaw_and_rate")
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--vz", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument(
        "--attitude-mode",
        choices=["body_rates", "attitude", "attitude_and_rates"],
        default="body_rates",
    )
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--roll-rate", type=float, default=0.0)
    parser.add_argument("--pitch-rate", type=float, default=0.0)
    parser.add_argument("--thrust", type=float, default=0.5)
    args = parser.parse_args()

    validate_rates(args.heartbeat_hz, args.command_hz)
    if args.idle_sleep_s < 0.0:
        raise ValueError("idle_sleep_s must be non-negative")

    if args.mode == "dry-run":
        run_dry(args)
    elif args.mode == "monitor":
        run_monitor(args)
    elif args.mode == "constant-velocity":
        run_constant_velocity(args)
    else:
        run_constant_attitude(args)


if __name__ == "__main__":
    main()
