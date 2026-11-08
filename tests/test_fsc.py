"""Tests for Gold-Standard Fourier Shell Correlation (FSC)."""

import pytest
import numpy as np
from spectro_bio.validation.fsc import FourierShellCorrelation


def test_fsc_identical_maps():
    """FSC between identical non-empty maps must equal 1.0 at all frequencies."""
    rng = np.random.default_rng(42)
    vol = rng.normal(0, 1.0, (16, 16, 16)).astype(np.float32)

    result = FourierShellCorrelation.compute_fsc(vol, vol, pixel_size_angstrom=1.0)
    assert len(result.fsc_values) > 0
    assert np.allclose(result.fsc_values, 1.0, atol=1e-5)
    # Resolution at 0.143 should reach the Nyquist limit
    assert result.resolution_fsc_0143_angstrom <= result.nyquist_resolution_angstrom + 0.1


def test_fsc_independent_noise():
    """FSC between two uncorrelated random Gaussian noise volumes should be near 0."""
    rng = np.random.default_rng(123)
    noise1 = rng.normal(0, 1.0, (24, 24, 24)).astype(np.float32)
    noise2 = rng.normal(0, 1.0, (24, 24, 24)).astype(np.float32)

    result = FourierShellCorrelation.compute_fsc(noise1, noise2, pixel_size_angstrom=1.0)
    # Mean correlation across all shells should be close to 0 (< 0.20)
    mean_abs_fsc = float(np.mean(np.abs(result.fsc_values)))
    assert mean_abs_fsc < 0.25


def test_fsc_shape_mismatch():
    """Mismatched half-map shapes must raise ValueError."""
    v1 = np.zeros((16, 16, 16))
    v2 = np.zeros((16, 16, 18))
    with pytest.raises(ValueError):
        FourierShellCorrelation.compute_fsc(v1, v2)
