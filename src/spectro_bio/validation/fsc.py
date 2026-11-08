"""Gold-Standard Fourier Shell Correlation (FSC) Resolution Metric."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class FSCResult:
    """Fourier Shell Correlation resolution assessment."""
    spatial_frequencies_inv_a: np.ndarray  # 1/Å
    resolutions_angstrom: np.ndarray       # Å
    fsc_values: np.ndarray                 # Correlation [-1, 1]
    resolution_fsc_0143_angstrom: float   # Gold-standard resolution at FSC = 0.143
    resolution_fsc_05_angstrom: float     # Conservative resolution at FSC = 0.50
    nyquist_resolution_angstrom: float    # Detector Nyquist limit (2 * pixel_size)


class FourierShellCorrelation:
    """Computes concentric radial Fourier cross-correlation between independent half-maps."""

    @staticmethod
    def compute_fsc(
        half_map_1: np.ndarray,
        half_map_2: np.ndarray,
        pixel_size_angstrom: float = 1.0,
        num_shells: Optional[int] = None,
    ) -> FSCResult:
        """
        Compute Gold-Standard FSC curve between two independently reconstructed 3D volumes.
        """
        if half_map_1.shape != half_map_2.shape:
            raise ValueError(f"Half-map shapes must match: {half_map_1.shape} vs {half_map_2.shape}")

        nz, ny, nx = half_map_1.shape
        fft1 = np.fft.fftn(half_map_1)
        fft2 = np.fft.fftn(half_map_2)

        # 3D reciprocal frequency grid in 1/Å
        fz = np.fft.fftfreq(nz, d=pixel_size_angstrom)
        fy = np.fft.fftfreq(ny, d=pixel_size_angstrom)
        fx = np.fft.fftfreq(nx, d=pixel_size_angstrom)

        fz_g, fy_g, fx_g = np.meshgrid(fz, fy, fx, indexing="ij")
        s_radius = np.sqrt(fz_g**2 + fy_g**2 + fx_g**2)

        # Max frequency is Nyquist: 1 / (2 * pixel_size)
        nyquist_freq = 0.5 / pixel_size_angstrom
        nyquist_res = 2.0 * pixel_size_angstrom

        n_shells = num_shells or (min(half_map_1.shape) // 2)
        shell_edges = np.linspace(0.0, nyquist_freq, n_shells + 1)

        f_centers: List[float] = []
        fsc_vals: List[float] = []

        for i in range(n_shells):
            s_min = shell_edges[i]
            s_max = shell_edges[i + 1]
            mask = (s_radius >= s_min) & (s_radius < s_max)
            
            if np.sum(mask) == 0:
                continue

            vals1 = fft1[mask]
            vals2 = fft2[mask]

            cross_prod = np.real(np.sum(vals1 * np.conj(vals2)))
            norm1 = np.sum(np.abs(vals1) ** 2)
            norm2 = np.sum(np.abs(vals2) ** 2)

            denom = np.sqrt(norm1 * norm2)
            corr = float(cross_prod / denom) if denom > 0 else 0.0

            f_centers.append(0.5 * (s_min + s_max))
            fsc_vals.append(np.clip(corr, -1.0, 1.0))

        freq_arr = np.array(f_centers)
        fsc_arr = np.array(fsc_vals)
        with np.errstate(divide='ignore'):
            res_arr = np.where(freq_arr > 0, 1.0 / freq_arr, 999.0)

        # Find 0.143 and 0.50 cutoff crossings via linear interpolation
        def find_cutoff_resolution(target_val: float) -> float:
            below_idx = np.where(fsc_arr < target_val)[0]
            if len(below_idx) == 0:
                return float(nyquist_res)  # Reaches Nyquist without dropping below
            idx = below_idx[0]
            if idx == 0:
                return float(res_arr[0])
            
            # Linear interpolation in frequency domain
            f1, f2 = freq_arr[idx - 1], freq_arr[idx]
            v1, v2 = fsc_arr[idx - 1], fsc_arr[idx]
            if abs(v2 - v1) < 1e-12:
                f_cross = f2
            else:
                f_cross = f1 + (target_val - v1) * (f2 - f1) / (v2 - v1)
            
            return float(1.0 / f_cross) if f_cross > 0 else 999.0

        res_0143 = find_cutoff_resolution(0.143)
        res_05 = find_cutoff_resolution(0.50)

        return FSCResult(
            spatial_frequencies_inv_a=freq_arr,
            resolutions_angstrom=res_arr,
            fsc_values=fsc_arr,
            resolution_fsc_0143_angstrom=res_0143,
            resolution_fsc_05_angstrom=res_05,
            nyquist_resolution_angstrom=nyquist_res,
        )
