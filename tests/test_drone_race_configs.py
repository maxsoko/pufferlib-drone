import configparser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path):
    parser = configparser.ConfigParser()
    parser.read(path)
    return parser


def test_drone_race_base_keeps_non_recurrent_ab_baseline():
    cfg = _load(ROOT / "pufferlib" / "config" / "drone_race.ini")

    assert cfg["base"]["rnn_name"] == "None"
    assert cfg["vec"]["backend"] == "Multiprocessing"
    assert cfg["vec"]["num_envs"] == "8"
    assert cfg["vec"]["num_workers"] == "8"
    assert cfg["train"]["num_envs"] == "8"
    assert cfg["train"]["env_batch_size"] == "8"
    assert cfg["train"]["batch_size"] == "512"
    assert cfg["train"]["minibatch_size"] == "512"
    assert cfg["train"]["device"] == "cuda"


def test_drone_race_curriculum_configs_default_to_recurrent_gpu_mp():
    filenames = [
        "drone_race_curriculum.ini",
        "drone_race_curriculum_tier1.ini",
        "drone_race_curriculum_tier2.ini",
        "drone_race_curriculum_tier3.ini",
    ]

    for name in filenames:
        cfg = _load(ROOT / "pufferlib" / "config" / name)

        assert cfg["base"]["rnn_name"] == "Recurrent"
        assert cfg["vec"]["backend"] == "Multiprocessing"
        assert cfg["vec"]["num_envs"] == "8"
        assert cfg["vec"]["num_workers"] == "8"
        assert cfg["train"]["num_envs"] == "8"
        assert cfg["train"]["env_batch_size"] == "8"
        assert cfg["train"]["batch_size"] == "512"
        assert cfg["train"]["minibatch_size"] == "512"
        assert cfg["train"]["device"] == "cuda"
