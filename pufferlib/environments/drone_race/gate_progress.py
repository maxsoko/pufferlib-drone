import numpy as np


def is_valid_gate_crossing(
    prev_position: np.ndarray,
    position: np.ndarray,
    gate_center: np.ndarray,
    gate_normal: np.ndarray,
    gate_radius: float,
    plane_cross_tolerance: float,
    direction_min: float,
) -> bool:
    """Return True if segment crosses gate plane within aperture and direction constraints."""
    prev_position = np.asarray(prev_position, dtype=np.float32)
    position = np.asarray(position, dtype=np.float32)
    gate_center = np.asarray(gate_center, dtype=np.float32)
    gate_normal = np.asarray(gate_normal, dtype=np.float32)

    move = position - prev_position
    move_norm = float(np.linalg.norm(move))
    if move_norm < 1e-8:
        return False
    move_dir = move / move_norm

    d_prev = float(np.dot(prev_position - gate_center, gate_normal))
    d_curr = float(np.dot(position - gate_center, gate_normal))

    # Require front-to-back crossing over a finite tolerance band.
    if not (d_prev <= -plane_cross_tolerance and d_curr >= plane_cross_tolerance):
        return False

    denom = d_prev - d_curr
    if abs(denom) < 1e-8:
        return False

    t = float(np.clip(d_prev / denom, 0.0, 1.0))
    hit = prev_position + t * (position - prev_position)

    radial = hit - gate_center
    radial -= float(np.dot(radial, gate_normal)) * gate_normal
    radial_dist = float(np.linalg.norm(radial))

    directed = float(np.dot(move_dir, gate_normal)) > float(direction_min)
    return radial_dist <= float(gate_radius) and directed
