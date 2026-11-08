"""Tests for Euler rotations and forward projection."""

import pytest
import numpy as np
from spectro_bio.reconstruction.fourier_slice import (
    EulerAngles,
    ProjectionGeometry,
    ForwardProjector,
    FourierSliceProjector,
    euler_rotation_matrix_zyz,
)


def test_euler_matrix_orthogonality():
    """Verify that ZYZ Euler rotation matrix is orthonormal with det(R) = +1."""
    angles = [
        (0.0, 0.0, 0.0),
        (45.0, 30.0, 60.0),
        (120.0, 90.0, 210.0),
        (315.0, 150.0, 180.0),
    ]
    for phi, theta, psi in angles:
        r = euler_rotation_matrix_zyz(phi, theta, psi)
        # R * R^T = I
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-7)
        # det(R) = +1
        assert np.isclose(np.linalg.det(r), 1.0, atol=1e-7)


def test_forward_projection_conservation():
    """Verify integral conservation in transmission forward projection."""
    vol = np.zeros((20, 20, 20), dtype=np.float32)
    vol[8:12, 8:12, 8:12] = 1.0
    vol_mass = np.sum(vol)

    geom = ProjectionGeometry(euler_angles=EulerAngles(phi_deg=30.0, theta_deg=45.0, psi_deg=60.0))
    proj = FourierSliceProjector.forward_project(vol, geom)

    assert proj.shape == (20, 20)
    # Total density should be approximately conserved by integration
    assert np.isclose(np.sum(proj), vol_mass, rtol=0.10)


def test_forward_projector_zero_angles():
    """Zero angles corresponds to simple z-axis integration."""
    vol = np.zeros((16, 16, 16), dtype=np.float32)
    vol[7:9, 7:9, 7:9] = 2.0

    proj = ForwardProjector.project_volume(vol, 0.0, 0.0, 0.0)
    direct_z_sum = np.sum(vol, axis=0)
    assert np.allclose(proj, direct_z_sum, atol=1e-5)
