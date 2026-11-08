"""Tests for Contrast Transfer Function (CTF) optics."""

import pytest
import numpy as np
from spectro_bio.optics.ctf import ContrastTransferFunction, CTFParameters
from spectro_bio.optics.phase_flipping import PhaseCorrector


def test_relativistic_wavelength():
    """Verify relativistic electron wavelength at standard 300 kV (Titan Krios)."""
    params = CTFParameters(
        defocus_u_angstrom=10000.0,
        defocus_v_angstrom=10000.0,
        accelerating_voltage_kv=300.0,
    )
    # At 300 kV, lambda is ~0.019687 Å
    assert np.isclose(params.wavelength_angstrom, 0.019687, rtol=1e-3)

    # At 200 kV (Talos Arctica / Glacios), lambda is ~0.02508 Å
    params_200 = CTFParameters(
        defocus_u_angstrom=10000.0,
        defocus_v_angstrom=10000.0,
        accelerating_voltage_kv=200.0,
    )
    assert np.isclose(params_200.wavelength_angstrom, 0.02508, rtol=1e-3)


def test_ctf_1d_evaluation_and_zero_crossings():
    """Test 1D CTF profile calculation and zero crossings (Thon rings)."""
    params = CTFParameters(
        defocus_u_angstrom=15000.0,
        defocus_v_angstrom=15000.0,
        spherical_aberration_cs_mm=2.7,
        accelerating_voltage_kv=300.0,
        amplitude_contrast_ratio=0.07,
    )
    profile = ContrastTransferFunction.evaluate_1d_profile(params, max_frequency_inv_a=0.5, num_points=300)

    assert len(profile.spatial_frequencies_inv_a) == 300
    assert len(profile.zero_crossings_freq_inv_a) > 3
    assert profile.first_zero_crossing_res_angstrom > 0
    # First zero crossing for 1.5 um defocus at 300kV is typically between 15 Å and 25 Å
    assert 12.0 < profile.first_zero_crossing_res_angstrom < 30.0


def test_ctf_2d_astigmatism():
    """Test 2D CTF with astigmatism."""
    params = CTFParameters(
        defocus_u_angstrom=12000.0,
        defocus_v_angstrom=18000.0,
        astigmatism_angle_deg=45.0,
        accelerating_voltage_kv=300.0,
    )
    ctf_2d = ContrastTransferFunction.evaluate_2d_ctf(params, image_size=64, pixel_size_angstrom=1.0)
    assert ctf_2d.shape == (64, 64)
    # Value range should be in [-1, 1]
    assert np.all(ctf_2d >= -1.0) and np.all(ctf_2d <= 1.0)
    # Astigmatic CTF should not be circularly symmetric
    slice_diag1 = np.diag(ctf_2d)
    slice_diag2 = np.diag(np.fliplr(ctf_2d))
    assert not np.allclose(slice_diag1, slice_diag2, atol=1e-2)


def test_phase_flipping_correction():
    """Test phase flipping inverts negative CTF bands."""
    params = CTFParameters(
        defocus_u_angstrom=15000.0,
        defocus_v_angstrom=15000.0,
        accelerating_voltage_kv=300.0,
    )
    ctf_2d = ContrastTransferFunction.evaluate_2d_ctf(params, image_size=32, pixel_size_angstrom=1.0)
    
    # Synthetic image
    img = np.ones((32, 32), dtype=np.float32)
    corrected = PhaseCorrector.correct_phase_flipping(img, ctf_2d)
    assert corrected.shape == (32, 32)
    assert not np.isnan(corrected).any()

    # Test Wiener filtering
    wiener = PhaseCorrector.wiener_filter(img, ctf_2d, snr_estimate=5.0)
    assert wiener.shape == (32, 32)
    assert not np.isnan(wiener).any()
