"""Contrast Transfer Function (CTF) of Transmission Electron Microscopes."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Sequence, Optional, List
import numpy as np


@dataclass
class CTF1DProfile:
    """1D CTF profile along radial spatial frequency."""
    spatial_frequencies_inv_a: np.ndarray
    ctf_values: np.ndarray
    zero_crossings_freq_inv_a: np.ndarray
    first_zero_crossing_freq_inv_a: float
    first_zero_crossing_res_angstrom: float


@dataclass
class CTFParameters:
    """Optical parameters of transmission electron microscope."""
    accelerating_voltage_kv: float = 300.0  # Accelerating voltage (e.g. 300 kV Titan Krios, 200 kV Talos)
    spherical_aberration_cs_mm: float = 2.7 # Cs in mm (e.g. 2.7 mm, or 0.0 for Cs corrector)
    defocus_u_angstrom: float = 15000.0     # Major axis defocus in Angstroms
    defocus_v_angstrom: float = 15000.0     # Minor axis defocus in Angstroms
    astigmatism_angle_deg: float = 0.0     # Angle of astigmatism in degrees
    amplitude_contrast_ratio: float = 0.08  # Fraction of amplitude contrast (typically 0.07 - 0.10)
    b_factor_envelope: float = 30.0        # Envelope decay factor in Angstrom^2
    phase_shift_rad: float = 0.0           # Phase shift (e.g., Volta phase plate)

    # Alternate naming properties for backward compatibility
    @property
    def voltage_kv(self) -> float:
        return self.accelerating_voltage_kv

    @property
    def spherical_aberration_mm(self) -> float:
        return self.spherical_aberration_cs_mm

    @property
    def defocus_u_um(self) -> float:
        return self.defocus_u_angstrom / 1e4

    @property
    def defocus_v_um(self) -> float:
        return self.defocus_v_angstrom / 1e4

    @property
    def amplitude_contrast(self) -> float:
        return self.amplitude_contrast_ratio

    @property
    def b_factor_a2(self) -> float:
        return self.b_factor_envelope

    @property
    def wavelength_angstrom(self) -> float:
        """
        Relativistic de Broglie wavelength of electron in Angstroms (Å):
        lambda = 12.2643 / sqrt(V * (1 + 0.978466e-6 * V)) where V is in Volts.
        """
        v_volts = self.accelerating_voltage_kv * 1000.0
        rel_factor = 1.0 + 0.978466e-6 * v_volts
        return float(12.2643 / np.sqrt(v_volts * rel_factor))

    @property
    def electron_wavelength_angstrom(self) -> float:
        return self.wavelength_angstrom

    @property
    def mean_defocus_angstrom(self) -> float:
        return 0.5 * (self.defocus_u_angstrom + self.defocus_v_angstrom)

    @property
    def mean_defocus_um(self) -> float:
        return self.mean_defocus_angstrom / 1e4


class ContrastTransferFunction:
    """Evaluates 1D and 2D CTF and Thon rings."""

    def __init__(self, params: Optional[CTFParameters] = None):
        self.params = params or CTFParameters()

    def evaluate_1d(self, spatial_freqs_inv_a: np.ndarray) -> np.ndarray:
        """
        Compute rotationally symmetric 1D CTF across spatial frequencies (1/Å).
        """
        s = np.asarray(spatial_freqs_inv_a, dtype=np.float64)
        lam = self.params.wavelength_angstrom
        cs_ang = self.params.spherical_aberration_cs_mm * 1e7  # mm to Å
        df_ang = self.params.mean_defocus_angstrom
        w = self.params.amplitude_contrast_ratio
        b = self.params.b_factor_envelope

        # Phase aberration chi(s) = pi * lambda * df * s^2 - 0.5 * pi * Cs * lambda^3 * s^4 - phase_shift
        chi = np.pi * lam * df_ang * (s ** 2) - 0.5 * np.pi * cs_ang * (lam ** 3) * (s ** 4) - self.params.phase_shift_rad

        # CTF(s) = -sqrt(1 - w^2) * sin(chi) - w * cos(chi) (standard cryo-EM convention)
        ctf = -np.sqrt(max(0.0, 1.0 - w**2)) * np.sin(chi) - w * np.cos(chi)
        
        # Envelope decay
        envelope = np.exp(-b * (s ** 2) / 4.0)
        return ctf * envelope

    def evaluate_2d(
        self,
        grid_shape: Tuple[int, int] | int,
        pixel_size_angstrom: float = 1.0,
    ) -> np.ndarray:
        """
        Compute 2D CTF matrix over 2D frequency grid.
        """
        if isinstance(grid_shape, int):
            grid_shape = (grid_shape, grid_shape)
        ny, nx = grid_shape

        fy = np.fft.fftfreq(ny, d=pixel_size_angstrom)
        fx = np.fft.fftfreq(nx, d=pixel_size_angstrom)
        fx_grid, fy_grid = np.meshgrid(fx, fy)

        s2 = fx_grid**2 + fy_grid**2
        theta = np.arctan2(fy_grid, fx_grid)

        p = self.params
        lam = p.wavelength_angstrom
        cs_ang = p.spherical_aberration_cs_mm * 1e7
        ang_rad = np.radians(p.astigmatism_angle_deg)

        # Astigmatic defocus df(theta) = 0.5 * (df_u + df_v) + 0.5 * (df_u - df_v) * cos(2*(theta - theta_ast))
        df_mean = 0.5 * (p.defocus_u_angstrom + p.defocus_v_angstrom)
        df_diff = 0.5 * (p.defocus_u_angstrom - p.defocus_v_angstrom)
        df_theta_ang = df_mean + df_diff * np.cos(2.0 * (theta - ang_rad))

        chi = np.pi * lam * df_theta_ang * s2 - 0.5 * np.pi * cs_ang * (lam ** 3) * (s2 ** 2) - p.phase_shift_rad

        w = p.amplitude_contrast_ratio
        ctf_2d = -np.sqrt(max(0.0, 1.0 - w**2)) * np.sin(chi) - w * np.cos(chi)
        envelope = np.exp(-p.b_factor_envelope * s2 / 4.0)

        return ctf_2d * envelope

    def find_zero_crossings(self, max_freq_inv_a: float = 0.5, num_points: int = 1000) -> np.ndarray:
        """Compute frequency locations (1/Å) where CTF crosses zero (Thon ring centers)."""
        freqs = np.linspace(1e-4, max_freq_inv_a, num_points)
        ctf_vals = self.evaluate_1d(freqs)
        
        sign_changes = np.where(np.diff(np.sign(ctf_vals)))[0]
        zeros = freqs[sign_changes]
        return zeros

    @classmethod
    def evaluate_1d_profile(
        cls,
        params: CTFParameters,
        max_frequency_inv_a: float = 0.5,
        num_points: int = 300,
    ) -> CTF1DProfile:
        """Classmethod to compute 1D profile with diagnostics and zero crossings."""
        instance = cls(params)
        freqs = np.linspace(1e-4, max_frequency_inv_a, num_points)
        ctf_vals = instance.evaluate_1d(freqs)
        zeros = instance.find_zero_crossings(max_freq_inv_a=max_frequency_inv_a, num_points=1000)

        z1 = float(zeros[0]) if len(zeros) > 0 else 0.0
        res1 = float(1.0 / z1) if z1 > 0 else 0.0

        return CTF1DProfile(
            spatial_frequencies_inv_a=freqs,
            ctf_values=ctf_vals,
            zero_crossings_freq_inv_a=zeros,
            first_zero_crossing_freq_inv_a=z1,
            first_zero_crossing_res_angstrom=res1,
        )

    @classmethod
    def evaluate_2d_ctf(
        cls,
        params: CTFParameters,
        image_size: int | Tuple[int, int],
        pixel_size_angstrom: float = 1.0,
    ) -> np.ndarray:
        """Classmethod to compute 2D CTF matrix."""
        instance = cls(params)
        return instance.evaluate_2d(image_size, pixel_size_angstrom=pixel_size_angstrom)
