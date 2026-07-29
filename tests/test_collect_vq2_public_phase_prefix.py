from __future__ import annotations

import numpy as np

from scripts.collect_vq2_public_phase_prefix import prefix_collection_passes


def test_prefix_admits_real_terminals_and_unterminated_survivors() -> None:
    lengths = np.full(512, 768, dtype=np.int32)
    terminal = np.zeros(512, dtype=np.int32)
    terminal[:400] = 1
    terminal_last = np.zeros(512, dtype=bool)
    terminal_last[:400] = True
    assert prefix_collection_passes(
        {},
        lengths=lengths,
        terminal_count=terminal,
        terminal_is_last=terminal_last,
        labels=int(lengths.sum()),
        executed_action_max_error=0.0,
    )


def test_prefix_rejects_nonfinal_real_terminal() -> None:
    lengths = np.full(512, 768, dtype=np.int32)
    terminal = np.ones(512, dtype=np.int32)
    terminal_last = np.ones(512, dtype=bool)
    terminal_last[0] = False
    assert not prefix_collection_passes(
        {},
        lengths=lengths,
        terminal_count=terminal,
        terminal_is_last=terminal_last,
        labels=int(lengths.sum()),
        executed_action_max_error=0.0,
    )

