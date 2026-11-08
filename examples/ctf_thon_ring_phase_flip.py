"""Contrast Transfer Function (CTF), Thon Rings, and Phase Flipping."""

from __future__ import annotations
import numpy as np

from spectro_bio.optics.ctf import ContrastTransferFunction, CTFParameters
from spectro_bio.optics.phase_flipping import PhaseCorrector
from spectro_bio.visualizer import AsciiVisualizer


def main():
    print("=" * 70)
    print("SpectroBio :: CTF Astigmatism, Thon Rings, and Phase Flipping")
    print("=" * 70)

    # 300 kV Field Emission Gun TEM parameters
    voltage = 300.0
    defocus_u = 12000.0  # 1.2 um along major axis
    defocus_v = 16000.0  # 1.6 um along minor axis (4000 Å astigmatism)
    astig_angle = 30.0   # 30 deg

    params = CTFParameters(
        defocus_u_angstrom=defocus_u,
        defocus_v_angstrom=defocus_v,
        astigmatism_angle_deg=astig_angle,
        spherical_aberration_cs_mm=2.7,
        accelerating_voltage_kv=voltage,
        amplitude_contrast_ratio=0.07,
        b_factor_envelope=40.0,
    )

    print(f"Accelerating Voltage:            {params.accelerating_voltage_kv:.1f} kV")
    print(f"Relativistic Electron Wavelength: {params.wavelength_angstrom:.6f} A")
    print(f"Defocus U / V:                   {params.defocus_u_angstrom:.1f} / {params.defocus_v_angstrom:.1f} A")
    print(f"Astigmatism Magnitude:           {abs(defocus_u - defocus_v):.1f} A @ {astig_angle:.1f} deg")

    profile = ContrastTransferFunction.evaluate_1d_profile(params, max_frequency_inv_a=0.5, num_points=250)
    print(f"\nFirst Zero-Crossing Resolution:  {profile.first_zero_crossing_res_angstrom:.2f} A")
    print("First 4 Thon Rings:")
    for idx, z in enumerate(profile.zero_crossings_freq_inv_a[:4], 1):
        print(f"  Ring {idx}: frequency = {z:7.4f} 1/A -> resolution = {1.0/z:6.2f} A")

    print("\nComputing 2D Astigmatic CTF...")
    ctf_2d = ContrastTransferFunction.evaluate_2d_ctf(params, image_size=40, pixel_size_angstrom=1.0)

    print("\n2D CTF Oscillating Thon Rings Pattern:")
    print(AsciiVisualizer.render_slice(ctf_2d, width=40, height=20))

    # Synthetic particle micrograph
    rng = np.random.default_rng(42)
    micrograph = np.zeros((40, 40), dtype=np.float32)
    micrograph[16:24, 16:24] = 1.0  # Particle
    
    # Convolve with CTF in Fourier space
    distorted = np.real(np.fft.ifft2(np.fft.fft2(micrograph) * ctf_2d))
    distorted_noisy = distorted + rng.normal(0, 0.1, distorted.shape)

    # Correct via phase-flipping
    phase_flipped = PhaseCorrector.correct_phase_flipping(distorted_noisy, ctf_2d)
    wiener_restored = PhaseCorrector.wiener_filter(distorted_noisy, ctf_2d, snr_estimate=5.0)

    print("\nPhase-Flipped Restored Particle:")
    print(AsciiVisualizer.render_slice(phase_flipped, width=40, height=20))
    print("=" * 70)


if __name__ == "__main__":
    main()
