"""Tests for 3D Filtered Backprojection."""

import pytest
import numpy as np
from spectro_bio.reconstruction.fourier_slice import EulerAngles, ProjectionGeometry, FourierSliceProjector
from spectro_bio.reconstruction.backprojector import FilteredBackprojector, SingleParticleReconstructor, ParticleProjection


def test_backprojection_reconstruction():
    """Verify that a 3D point source / sphere can be reconstructed from multi-angle projections."""
    size = 16
    vol = np.zeros((size, size, size), dtype=np.float32)
    vol[7:9, 7:9, 7:9] = 1.0  # Center blob

    projections = []
    geometries = []

    # Sample several view directions
    for phi in [0.0, 45.0, 90.0, 135.0]:
        for theta in [30.0, 60.0, 90.0]:
            geom = ProjectionGeometry(euler_angles=EulerAngles(phi, theta, 0.0))
            p = FourierSliceProjector.forward_project(vol, geom)
            projections.append(p)
            geometries.append(geom)

    reconstructed = FilteredBackprojector.reconstruct(
        projections,
        geometries,
        output_shape=(size, size, size),
        filter_type="shepp_logan",
    )

    assert reconstructed.shape == (size, size, size)
    # Peak density should be near center
    center_idx = size // 2
    peak_val = reconstructed[center_idx, center_idx, center_idx]
    corner_val = reconstructed[1, 1, 1]
    assert peak_val > corner_val


def test_empty_projections():
    """Empty projections list should return zero volume without error."""
    rec = FilteredBackprojector.reconstruct([], output_shape=(10, 10, 10))
    assert rec.shape == (10, 10, 10)
    assert np.all(rec == 0.0)
