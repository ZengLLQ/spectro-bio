"""Guinier Analysis and B-Factor Map Sharpening for Cryo-EM 3D Density Maps."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class GuinierResult:
    """Results of Guinier plot analysis."""
    s_squared_inv_a2: np.ndarray       # s^2 in 1/Å^2
    ln_intensity: np.ndarray          # ln(<|F(s)|^2>)
    fitted_b_factor: float            # Estimated B-factor in Å^2 (positive value indicates damping)
    slope: float                      # Slope of linear fit: -B / 2
    intercept: float                  # Intercept
    fit_freq_range: Tuple[float, float]  # Range of s (1/Å) used for fit [s_min, s_max]


class BFactorSharpener:
    """Performs Rosenthal-Henderson Guinier analysis and B-factor map sharpening."""

    @staticmethod
    def compute_radial_profile(
        volume: np.ndarray,
        pixel_size_angstrom: float = 1.0,
        num_shells: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute spherically averaged radial power spectrum.
        Returns:
            frequencies_s: 1D array of spatial frequency s (1/Å)
            mean_intensity: 1D array of mean Fourier intensity <|F(s)|^2>
        """
        nz, ny, nx = volume.shape
        fft_vol = np.fft.fftn(volume)
        power_spectrum = np.abs(fft_vol) ** 2

        fz = np.fft.fftfreq(nz, d=pixel_size_angstrom)
        fy = np.fft.fftfreq(ny, d=pixel_size_angstrom)
        fx = np.fft.fftfreq(nx, d=pixel_size_angstrom)

        fz_g, fy_g, fx_g = np.meshgrid(fz, fy, fx, indexing="ij")
        s_radius = np.sqrt(fz_g**2 + fy_g**2 + fx_g**2)

        nyquist_freq = 0.5 / pixel_size_angstrom
        n_shells = num_shells or (min(volume.shape) // 2)
        shell_edges = np.linspace(0.0, nyquist_freq, n_shells + 1)

        s_centers = []
        intensities = []

        for i in range(n_shells):
            s_min = shell_edges[i]
            s_max = shell_edges[i + 1]
            mask = (s_radius >= s_min) & (s_radius < s_max)
            if np.sum(mask) == 0:
                continue
            
            s_centers.append(0.5 * (s_min + s_max))
            intensities.append(float(np.mean(power_spectrum[mask])))

        return np.array(s_centers), np.array(intensities)

    @classmethod
    def estimate_b_factor(
        cls,
        volume: np.ndarray,
        pixel_size_angstrom: float = 1.0,
        fit_res_range: Tuple[float, float] = (10.0, 3.5),
    ) -> GuinierResult:
        """
        Estimate overall B-factor via Guinier plot ln(I(s)) vs s^2.
        fit_res_range: tuple of (low_res_A, high_res_A), e.g. (10.0, 3.5)
        Note: slope = -B / 2  =>  B = -2 * slope
        """
        s_centers, intensities = cls.compute_radial_profile(volume, pixel_size_angstrom)
        
        # Avoid non-positive intensity
        valid = intensities > 0
        s_valid = s_centers[valid]
        i_valid = intensities[valid]

        s2 = s_valid ** 2
        ln_i = np.log(i_valid)

        # Convert resolution bounds to frequency s = 1 / resolution
        high_res, low_res = sorted(fit_res_range)
        s_min = 1.0 / low_res
        s_max = 1.0 / high_res

        fit_mask = (s_valid >= s_min) & (s_valid <= s_max)
        if np.sum(fit_mask) < 3:
            # Fallback to broader range if fit range is too narrow
            fit_mask = (s_valid > 0.05) & (s_valid < 0.35)

        if np.sum(fit_mask) < 2:
            return GuinierResult(
                s_squared_inv_a2=s2,
                ln_intensity=ln_i,
                fitted_b_factor=0.0,
                slope=0.0,
                intercept=0.0,
                fit_freq_range=(s_min, s_max),
            )

        slope, intercept = np.polyfit(s2[fit_mask], ln_i[fit_mask], deg=1)
        # Intensity I(s) = I0 * exp(-B * s^2 / 2) => ln(I) = ln(I0) - (B / 2) * s^2
        b_factor = float(-2.0 * slope)

        return GuinierResult(
            s_squared_inv_a2=s2,
            ln_intensity=ln_i,
            fitted_b_factor=b_factor,
            slope=float(slope),
            intercept=float(intercept),
            fit_freq_range=(s_min, s_max),
        )

    @classmethod
    def sharpen_map(
        cls,
        volume: np.ndarray,
        b_factor: float,
        pixel_size_angstrom: float = 1.0,
        high_res_cutoff_angstrom: Optional[float] = None,
        falloff_width_angstrom: float = 1.0,
    ) -> np.ndarray:
        """
        Apply negative B-factor sharpening to density map in Fourier space:
        F_sharp(s) = F(s) * exp(b_factor * s^2 / 4) * CosineCutoff(s)
        """
        nz, ny, nx = volume.shape
        fft_vol = np.fft.fftn(volume)

        fz = np.fft.fftfreq(nz, d=pixel_size_angstrom)
        fy = np.fft.fftfreq(ny, d=pixel_size_angstrom)
        fx = np.fft.fftfreq(nx, d=pixel_size_angstrom)

        fz_g, fy_g, fx_g = np.meshgrid(fz, fy, fx, indexing="ij")
        s = np.sqrt(fz_g**2 + fy_g**2 + fx_g**2)

        # Amplitude boost: exp(B * s^2 / 4)
        sharpening_filter = np.exp(b_factor * (s**2) / 4.0)

        # Damping window to prevent noise explosion past resolution cutoff
        if high_res_cutoff_angstrom is not None and high_res_cutoff_angstrom > 0:
            s_cutoff = 1.0 / high_res_cutoff_angstrom
            s_falloff = 1.0 / max(high_res_cutoff_angstrom - falloff_width_angstrom, 0.5 * high_res_cutoff_angstrom)
            delta_s = s_falloff - s_cutoff

            # Raised cosine smooth roll-off
            filter_window = np.ones_like(s)
            in_falloff = (s >= s_cutoff) & (s < s_falloff)
            past_falloff = s >= s_falloff

            filter_window[in_falloff] = 0.5 * (1.0 + np.cos(np.pi * (s[in_falloff] - s_cutoff) / delta_s))
            filter_window[past_falloff] = 0.0

            sharpening_filter *= filter_window

        fft_sharpened = fft_vol * sharpening_filter
        volume_sharpened = np.real(np.fft.ifftn(fft_sharpened))

        return volume_sharpened
