"""Fourier Slice Theorem and 3D-to-2D forward projections."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
from scipy.ndimage import affine_transform


@dataclass
class EulerAngles:
    """Euler angles under ZYZ convention."""
    phi_deg: float = 0.0    # First rotation around Z
    theta_deg: float = 0.0  # Rotation around Y (tilt angle)
    psi_deg: float = 0.0    # Second rotation around Z


@dataclass
class ProjectionGeometry:
    """Geometry for a single cryo-EM projection."""
    euler_angles: EulerAngles
    shift_x_angstrom: float = 0.0
    shift_y_angstrom: float = 0.0


@dataclass
class ParticleProjection:
    """A 2D cryo-EM particle image with associated Euler angles."""
    image_2d: np.ndarray
    phi_deg: float    # In-plane rotation
    theta_deg: float  # Tilt angle (0 to 180 deg)
    psi_deg: float    # Third Euler angle


def euler_rotation_matrix_zyz(phi_deg: float, theta_deg: float, psi_deg: float) -> np.ndarray:
    """
    Construct 3x3 rotation matrix using ZYZ Euler angle convention:
    R = R_z(psi) * R_y(theta) * R_z(phi)
    """
    phi = np.radians(phi_deg)
    theta = np.radians(theta_deg)
    psi = np.radians(psi_deg)

    r_z1 = np.array([
        [np.cos(phi), -np.sin(phi), 0.0],
        [np.sin(phi), np.cos(phi), 0.0],
        [0.0, 0.0, 1.0],
    ])
    r_y = np.array([
        [np.cos(theta), 0.0, np.sin(theta)],
        [0.0, 1.0, 0.0],
        [-np.sin(theta), 0.0, np.cos(theta)],
    ])
    r_z2 = np.array([
        [np.cos(psi), -np.sin(psi), 0.0],
        [np.sin(psi), np.cos(psi), 0.0],
        [0.0, 0.0, 1.0],
    ])

    return r_z2 @ r_y @ r_z1


class ForwardProjector:
    """Generates 2D cryo-EM transmission projections from 3D Coulomb potential density."""

    @staticmethod
    def project_volume(
        volume_3d: np.ndarray,
        phi_deg: float,
        theta_deg: float,
        psi_deg: float,
    ) -> np.ndarray:
        """
        Rotate 3D volume by Euler angles and integrate along z-axis to yield 2D projection.
        """
        shape = volume_3d.shape
        center = np.array(shape) / 2.0

        r_mat = euler_rotation_matrix_zyz(phi_deg, theta_deg, psi_deg)
        # Affine transform expects mapping from output to input coordinates: inv(R)
        inv_r = r_mat.T
        offset = center - inv_r @ center

        rotated = affine_transform(
            volume_3d,
            matrix=inv_r,
            offset=offset,
            order=1,
            cval=0.0,
            prefilter=False,
        )

        # Transmitted electron beam projection: integral along depth z (axis 0)
        proj_2d = np.sum(rotated, axis=0)
        return proj_2d


class FourierSliceProjector:
    """Forward projection interface using ProjectionGeometry."""

    @staticmethod
    def forward_project(
        volume_3d: np.ndarray,
        geometry: ProjectionGeometry,
    ) -> np.ndarray:
        """Project volume along geometry direction."""
        return ForwardProjector.project_volume(
            volume_3d,
            phi_deg=geometry.euler_angles.phi_deg,
            theta_deg=geometry.euler_angles.theta_deg,
            psi_deg=geometry.euler_angles.psi_deg,
        )
