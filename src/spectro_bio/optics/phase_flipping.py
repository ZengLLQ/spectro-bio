"""Phase flipping and Wiener restoration in Fourier space."""

from __future__ import annotations
import numpy as np


class CTFCorrector:
    """Corrects phase inversions and modulates amplitudes in Fourier space."""

    @staticmethod
    def phase_flip(image: np.ndarray, ctf_2d: np.ndarray) -> np.ndarray:
        """
        Invert Fourier phases where CTF is negative:
        I_corr = ifft2( fft2(I) * sgn(CTF) )
        """
        fft_img = np.fft.fft2(image)
        # Sign of CTF (+1 or -1)
        sign_ctf = np.sign(ctf_2d)
        sign_ctf[sign_ctf == 0] = 1.0

        corrected_fft = fft_img * sign_ctf
        corrected_img = np.real(np.fft.ifft2(corrected_fft))
        return corrected_img

    @staticmethod
    def wiener_filter(
        image: np.ndarray,
        ctf_2d: np.ndarray,
        snr_estimate: float = 10.0,
        noise_to_signal_ratio: float | None = None,
    ) -> np.ndarray:
        """
        Apply Wiener filter:
        W = CTF / (CTF^2 + NSR)
        """
        if noise_to_signal_ratio is None:
            nsr = 1.0 / max(snr_estimate, 1e-4)
        else:
            nsr = noise_to_signal_ratio

        fft_img = np.fft.fft2(image)
        wiener_weights = ctf_2d / (ctf_2d ** 2 + nsr)

        filtered_fft = fft_img * wiener_weights
        restored_img = np.real(np.fft.ifft2(filtered_fft))
        return restored_img


class PhaseCorrector:
    """Convenience alias for phase flipping and Wiener deconvolution."""

    @staticmethod
    def correct_phase_flipping(image: np.ndarray, ctf_2d: np.ndarray) -> np.ndarray:
        return CTFCorrector.phase_flip(image, ctf_2d)

    @staticmethod
    def wiener_filter(image: np.ndarray, ctf_2d: np.ndarray, snr_estimate: float = 10.0) -> np.ndarray:
        return CTFCorrector.wiener_filter(image, ctf_2d, snr_estimate=snr_estimate)
