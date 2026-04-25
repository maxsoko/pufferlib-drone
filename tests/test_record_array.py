import gymnasium as gym
import numpy as np


def create_dtype_from_space(space):
    if isinstance(space, gym.spaces.Dict):
        dtype_fields = [(name, create_dtype_from_space(subspace)) for name, subspace in space.spaces.items()]
        return np.dtype(dtype_fields)
    if isinstance(space, gym.spaces.Tuple):
        dtype_fields = [('field' + str(i), create_dtype_from_space(subspace)) for i, subspace in enumerate(space.spaces)]
        return np.dtype(dtype_fields)
    if isinstance(space, gym.spaces.Box):
        return (space.dtype, space.shape)
    if isinstance(space, gym.spaces.Discrete):
        return np.int64
    raise TypeError(f"Unsupported space type: {type(space)}")


def sample_and_convert(space, dtype):
    sample = space.sample()
    flat_sample = {}

    def flatten(sample_item, name_prefix=""):
        for key, item in sample_item.items():
            full_key = name_prefix + key if name_prefix == "" else name_prefix + "_" + key
            if isinstance(item, dict):
                flatten(item, full_key)
            else:
                flat_sample[full_key] = item

    flatten(sample)
    return np.array(tuple(flat_sample.values()), dtype=dtype)


def test_record_array_roundtrip():
    space = gym.spaces.Dict({
        "position": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    })

    space_dtype = create_dtype_from_space(space)
    samples = [sample_and_convert(space, space_dtype) for _ in range(3)]
    record_array = np.rec.array(np.asarray(samples, dtype=space_dtype))
    bytes_array = record_array.tobytes()
    roundtrip = np.rec.array(bytes_array, dtype=space_dtype)

    assert roundtrip.shape == record_array.shape
    assert roundtrip.dtype == record_array.dtype
