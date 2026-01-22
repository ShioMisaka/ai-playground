"""
Model scaling utilities for YOLO models.

Provides functions to compute scaled channel counts and layer depths
for dynamic model scaling (n/s/m/l/x variants).
"""


def make_divisible(x, divisor=8):
    """Returns nearest x divisible by divisor.

    Args:
        x: Value to make divisible
        divisor: Divisor (default 8 for hardware efficiency)

    Returns:
        Nearest integer divisible by divisor
    """
    return int(round(x / divisor) * divisor)


def compute_channels(channels, width_multiple, max_channels=None, min_channels=8):
    """Compute scaled number of channels.

    Args:
        channels: Base number of channels
        width_multiple: Width scaling factor
        max_channels: Maximum number of channels (optional)
        min_channels: Minimum number of channels (default 8)

    Returns:
        Scaled number of channels
    """
    channels = make_divisible(channels * width_multiple)
    if max_channels is not None:
        channels = min(channels, max_channels)
    return int(max(channels, min_channels))  # Ensure at least min_channels


def compute_depth(n, depth_multiple):
    """Compute scaled number of layers/repeats.

    Args:
        n: Base number of layers
        depth_multiple: Depth scaling factor

    Returns:
        Scaled number of layers (at least 1)
    """
    if n <= 1:
        return n
    return max(round(n * depth_multiple), 1)


__all__ = [
    'make_divisible',
    'compute_channels',
    'compute_depth',
]
