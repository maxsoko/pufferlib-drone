#!/usr/bin/env python3
"""Load the configured checkpoint callable and print one zero-state action."""

from policy_callable_checkpoint import infer, reset


def main() -> None:
    reset()
    print(infer([0.0] * 32))


if __name__ == "__main__":
    main()
