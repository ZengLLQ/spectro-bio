"""3D Backprojection and Reconstruction from oriented single-particle projections."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional, Union
import numpy as np
from scipy.ndimage import affine_transform
from spectro_bio.reconstruction.fourier_slice import (
    ParticleProjection,
    ProjectionGeometry,
    EulerAngles,
    euler_rotation_matrix_zyz,
)


@dataclass
class ReconstructionGrid:
    """3D Voxel grid parameters."""
    shape: Tuple[int, int, int]
    pixel_size_angstrom: float = 1.0


class SingleParticleReconstructor:
    """Reconstructs 3D Coulomb potential volume from oriented 2D single-particle projections."""

    def __init__(self, volume_shape: Tuple[int, int, int], pixel_size_angstrom: float = 1.0):
        self.shape = volume_shape
        self.pixel_size = pixel_size_angstrom

    def _filter_projection(self, proj_2d: np.ndarray, filter_type: str = "ramp") -> np.ndarray:
        """Apply 2D ramp or Shepp-Logan filter |s| in Fourier space prior to backprojection."""
        ny, nx = proj_2d.shape
        fy = np.fft.fftfreq(ny)
        fx = np.fft.fftfreq(nx)
        fx_g, fy_g = np.meshgrid(fx, fy)
        ramp = np.sqrt(fx_g**2 + fy_g**2)

        if filter_type == "shepp_logan":
            with np.errstate(divide="ignore", invalid="ignore"):
                sinc = np.where(ramp > 0, np.sin(np.pi * ramp) / (np.pi * ramp), 1.0)
            filt = ramp * sinc
        else:
            filt = ramp

        fft_p = np.fft.fft2(proj_2d)
        filtered = np.real(np.fft.ifft2(fft_p * filt))
        return filtered

    def reconstruct(
        self,
        projections: Sequence[ParticleProjection],
        apply_filter: bool = True,
        filter_type: str = "ramp",
    ) -> np.ndarray:
        """
        Backproject all oriented 2D projections into the 3D volume.
        """
        nz, ny, nx = self.shape
        vol = np.zeros(self.shape, dtype=np.float64)
        center = np.array(self.shape) / 2.0

        for p in projections:
            filt_proj = self._filter_projection(p.image_2d, filter_type) if apply_filter else p.image_2d
            
            # Broadcast 2D projection into a 3D ray beam cylinder along z-axis (axis 0)
            cylinder = np.repeat(filt_proj[np.newaxis, :, :], nz, axis=0)

            # Rotate cylinder back into laboratory frame: R = R(phi, theta, psi)
            r_mat = euler_rotation_matrix_zyz(p.phi_deg, p.theta_deg, p.psi_deg)
            offset = center - r_mat @ center

            backprojected_ray = affine_transform(
                cylinder,
                matrix=r_mat,
                offset=offset,
                order=1,
                cval=0.0,
                prefilter=False,
            )
            vol += backprojected_ray

        n_projs = max(1, len(projections))
        vol /= n_projs
        return vol


class FilteredBackprojector:
    """Convenience class for single-particle backprojection."""

    @classmethod
    def reconstruct(
        cls,
        particles: Sequence[np.ndarray] | Sequence[ParticleProjection],
        geometries: Optional[Sequence[ProjectionGeometry]] = None,
        output_shape: Tuple[int, int, int] = (32, 32, 32),
        pixel_size_angstrom: float = 1.0,
        filter_type: str = "shepp_logan",
    ) -> np.ndarray:
        """Reconstruct 3D density map."""
        if len(particles) == 0:
            return np.zeros(output_shape, dtype=np.float64)

        # Check format of input
        if isinstance(particles[0], ParticleProjection):
            projs = list(particles)  # type: ignore
        else:
            if geometries is None or len(geometries) != len(particles):
                raise ValueError("Must provide matching geometries for raw particle arrays.")
            projs = [
                ParticleProjection(
                    image_2d=p,
                    phi_deg=g.euler_angles.phi_deg,
                    theta_deg=g.euler_angles.theta_deg,
                    psi_deg=g.euler_angles.psi_deg,
                )
                for p, g in zip(particles, geometries)
            ]

        reconstructor = SingleParticleReconstructor(output_shape, pixel_size_angstrom)
        return reconstructor.reconstruct(projs, apply_filter=True, filter_type=filter_type)
