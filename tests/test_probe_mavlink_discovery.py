from __future__ import annotations

from types import SimpleNamespace

import probe_mavlink_discovery as probe


class FakeSender:
    def __init__(self):
        self.calls = []

    def command_long_send(self, *args):
        self.calls.append(args)


class FakeMavlink:
    MAV_CMD_REQUEST_MESSAGE = 512
    MAVLINK_MSG_ID_ATTITUDE = 30
    MAVLINK_MSG_ID_MESSAGE_INTERVAL = 244


def fake_adapter():
    sender = FakeSender()
    adapter = SimpleNamespace(
        master=SimpleNamespace(mav=sender),
        mavutil=SimpleNamespace(mavlink=FakeMavlink),
        _target_ids=lambda: (7, 0),
    )
    return adapter, sender


def test_mavlink_message_id_uses_generated_constant():
    assert probe.mavlink_message_id(FakeMavlink, "attitude") == 30


def test_mavlink_message_id_falls_back_for_newer_standard_message():
    assert probe.mavlink_message_id(FakeMavlink, "protocol_version") == 300


def test_request_message_addresses_response_to_requester():
    adapter, sender = fake_adapter()
    probe.send_request_message(adapter, 30)
    assert sender.calls == [
        (7, 0, 512, 0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    ]


def test_interval_query_requests_message_interval_for_target_id():
    adapter, sender = fake_adapter()
    probe.send_message_interval_query(adapter, 30)
    assert sender.calls == [
        (7, 0, 512, 0, 244.0, 30.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    ]


def test_default_requests_exclude_position_and_sim_truth():
    forbidden = {
        "LOCAL_POSITION_NED",
        "ODOMETRY",
        "SIM_STATE",
        "HIL_STATE",
        "HIL_STATE_QUATERNION",
        "GLOBAL_POSITION_INT",
        "GPS_RAW_INT",
    }
    assert not forbidden.intersection(probe.DEFAULT_ONE_SHOT_MESSAGES)
    assert not forbidden.intersection(probe.DEFAULT_INTERVAL_MESSAGES)
