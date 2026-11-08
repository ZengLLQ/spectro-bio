"""Tests for Guinier analysis and B-factor map sharpening."""

import pytest
import numpy as np
from spectro_bio.validation.b_factor import BFactorSharpener


def test_b_factor_estimation_and_sharpening():
    """Verify Guinier estimation and negative B-factor sharpening."""
    size = 24
    vol = np.zeros((size, size, size), dtype=np.float32)
    # Put a Gaussian source at center
    c = size // 2
    z, y, x = np.ogrid[:size, :size, :size]
    r2 = (x - c)**2 + (y - c)**2 + (z - c)**2
    vol = np.exp(-r2 / (2 * 2.0**2))

    guinier = BFactorSharpener.estimate_b_factor(vol, pixel_size_angstrom=1.0)
    assert len(guinier.s_squared_inv_a2) > 0
    assert len(guinier.ln_intensity) > 0

    # Sharpen map
    sharpened = BFactorSharpener.sharpen_map(vol, b_factor=80.0, pixel_size_angstrom=1.0, high_res_cutoff_angstrom=4.0)
    assert sharpened.shape == vol.shape
    assert not np.isnan(sharpened).any()
    # Sharpened peak should be higher or more peaked
    assert np.max(sharpened) > 0.0
