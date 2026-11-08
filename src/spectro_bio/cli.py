"""Command-Line Interface for SpectroBio."""

from __future__ import annotations
import argparse
import sys
import numpy as np

from spectro_bio.optics.ctf import ContrastTransferFunction, CTFParameters
from spectro_bio.optics.phase_flipping import PhaseCorrector
from spectro_bio.reconstruction.fourier_slice import EulerAngles, ProjectionGeometry, FourierSliceProjector
from spectro_bio.reconstruction.backprojector import FilteredBackprojector
from spectro_bio.validation.fsc import FourierShellCorrelation
from spectro_bio.validation.b_factor import BFactorSharpener
from spectro_bio.visualizer import AsciiVisualizer


def create_synthetic_phantom(grid_size: int = 32) -> np.ndarray:
    """Generate a realistic asymmetric multi-subunit macromolecular phantom (e.g. ribosome/chaperonin-like)."""
    vol = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    center = grid_size // 2
    r_max = grid_size // 4

    z, y, x = np.ogrid[:grid_size, :grid_size, :grid_size]

    # Core spherical domain
    d_core = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
    vol[d_core <= r_max] += np.exp(-0.5 * (d_core[d_core <= r_max] / (r_max * 0.6))**2)

    # Subunit 1 (top lobe)
    d_top = np.sqrt((x - center)**2 + (y - (center + r_max * 0.7))**2 + (z - (center + r_max * 0.5))**2)
    vol[d_top <= r_max * 0.6] += 0.8 * np.exp(-0.5 * (d_top[d_top <= r_max * 0.6] / (r_max * 0.4))**2)

    # Subunit 2 (lateral domain)
    d_lat = np.sqrt((x - (center - r_max * 0.8))**2 + (y - (center - r_max * 0.4))**2 + (z - center)**2)
    vol[d_lat <= r_max * 0.5] += 0.7 * np.exp(-0.5 * (d_lat[d_lat <= r_max * 0.5] / (r_max * 0.35))**2)

    return vol


def cmd_ctf(args: argparse.Namespace) -> int:
    """Analyze CTF parameters and Thon ring oscillations."""
    params = CTFParameters(
        defocus_u_angstrom=args.defocus,
        defocus_v_angstrom=args.defocus if args.astigmatism == 0 else args.defocus + args.astigmatism,
        astigmatism_angle_deg=args.astig_angle,
        spherical_aberration_cs_mm=args.cs,
        accelerating_voltage_kv=args.voltage,
        amplitude_contrast_ratio=args.amp_ratio,
        b_factor_envelope=args.b_factor,
    )
    profile = ContrastTransferFunction.evaluate_1d_profile(params, max_frequency_inv_a=args.max_freq, num_points=200)

    print("=" * 65)
    print("SpectroBio :: Contrast Transfer Function (CTF) Diagnostics")
    print("=" * 65)
    print(f"Accelerating Voltage:        {params.accelerating_voltage_kv:.1f} kV")
    print(f"Relativistic Electron Wavelength: {params.wavelength_angstrom:.6f} A")
    print(f"Defocus (U / V):             {params.defocus_u_angstrom:.1f} / {params.defocus_v_angstrom:.1f} A")
    print(f"Mean Defocus:                {params.mean_defocus_angstrom:.1f} A")
    print(f"Spherical Aberration (Cs):   {params.spherical_aberration_cs_mm:.2f} mm")
    print(f"First Thon Ring Zero (Z1):   {profile.first_zero_crossing_res_angstrom:.2f} A (s = {profile.first_zero_crossing_freq_inv_a:.4f} 1/A)")
    print("-" * 65)
    print("First 5 CTF Zero Crossings (Thon Rings):")
    for i, z_freq in enumerate(profile.zero_crossings_freq_inv_a[:5], 1):
        res = 1.0 / z_freq if z_freq > 0 else 0.0
        print(f"  Ring {i:2d}: s = {z_freq:8.4f} 1/A | Resolution = {res:6.2f} A")
    print("=" * 65)
    return 0


def cmd_fsc(args: argparse.Namespace) -> int:
    """Compute FSC resolution between two simulated half-maps."""
    print("=" * 65)
    print("SpectroBio :: Gold-Standard Fourier Shell Correlation (FSC)")
    print("=" * 65)

    size = args.size
    pixel_size = args.pixel_size
    print(f"Generating synthetic macromolecular half-maps ({size}x{size}x{size}, {pixel_size:.2f} Å/px)...")

    base = create_synthetic_phantom(size)
    rng = np.random.default_rng(42)
    noise_sigma = args.noise
    half1 = base + rng.normal(0, noise_sigma, base.shape).astype(np.float32)
    half2 = base + rng.normal(0, noise_sigma, base.shape).astype(np.float32)

    result = FourierShellCorrelation.compute_fsc(half1, half2, pixel_size_angstrom=pixel_size)

    print(AsciiVisualizer.render_fsc_curve(result))
    print("-" * 65)
    print(f"Gold-Standard FSC 0.143 Resolution: {result.resolution_fsc_0143_angstrom:.2f} A")
    print(f"Conservative FSC 0.50 Resolution:   {result.resolution_fsc_05_angstrom:.2f} A")
    print(f"Detector Nyquist Limit:             {result.nyquist_resolution_angstrom:.2f} A")
    print("=" * 65)
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Execute complete end-to-end single-particle reconstruction pipeline."""
    print("=" * 70)
    print("SpectroBio :: End-to-End Single-Particle Macromolecular Reconstruction")
    print("=" * 70)

    size = args.size
    pixel_size = args.pixel_size
    n_projections = args.n_projections
    snr = args.snr

    print(f"[1/5] Building Ground-Truth Macromolecular Phantom ({size}x{size}x{size})...")
    phantom = create_synthetic_phantom(size)

    print(f"[2/5] Synthesizing {n_projections} Random Particle Projections with CTF & Noise...")
    ctf_params = CTFParameters(
        defocus_u_angstrom=15000.0,
        defocus_v_angstrom=15000.0,
        spherical_aberration_cs_mm=2.7,
        accelerating_voltage_kv=300.0,
        amplitude_contrast_ratio=0.07,
    )

    rng = np.random.default_rng(100)
    geometries = []
    particles = []
    ctf_2d = ContrastTransferFunction.evaluate_2d_ctf(ctf_params, size, pixel_size)

    for i in range(n_projections):
        phi = float(rng.uniform(0, 360))
        theta = float(np.degrees(np.arccos(rng.uniform(-1, 1))))
        psi = float(rng.uniform(0, 360))
        geom = ProjectionGeometry(euler_angles=EulerAngles(phi, theta, psi))
        geometries.append(geom)

        proj = FourierSliceProjector.forward_project(phantom, geom)
        
        # Convolve with CTF in Fourier space
        fft_p = np.fft.fft2(proj)
        proj_ctf = np.real(np.fft.ifft2(fft_p * ctf_2d))

        # Add Gaussian noise for target SNR
        signal_pwr = np.var(proj_ctf)
        noise_sigma = np.sqrt(signal_pwr / max(snr, 1e-4))
        noisy_proj = proj_ctf + rng.normal(0, noise_sigma, proj.shape)

        # Contrast phase-flipping correction
        corrected_proj = PhaseCorrector.correct_phase_flipping(noisy_proj, ctf_2d)
        particles.append(corrected_proj)

    print(f"[3/5] Reconstructing 3D Volume via Reciprocal Filtered Backprojection...")
    # Split into odd and even half-datasets for Gold-Standard FSC
    half1_particles, half1_geom = particles[0::2], geometries[0::2]
    half2_particles, half2_geom = particles[1::2], geometries[1::2]

    rec_half1 = FilteredBackprojector.reconstruct(half1_particles, half1_geom, output_shape=(size, size, size), filter_type="shepp_logan")
    rec_half2 = FilteredBackprojector.reconstruct(half2_particles, half2_geom, output_shape=(size, size, size), filter_type="shepp_logan")
    rec_full = FilteredBackprojector.reconstruct(particles, geometries, output_shape=(size, size, size), filter_type="shepp_logan")

    print("[4/5] Evaluating Gold-Standard Fourier Shell Correlation (FSC)...")
    fsc_res = FourierShellCorrelation.compute_fsc(rec_half1, rec_half2, pixel_size_angstrom=pixel_size)
    print(AsciiVisualizer.render_fsc_curve(fsc_res))
    print(f"      Calculated FSC 0.143 Resolution: {fsc_res.resolution_fsc_0143_angstrom:.2f} A")

    print("[5/5] Estimating Rosenthal-Henderson Guinier B-factor and Sharpening...")
    guinier = BFactorSharpener.estimate_b_factor(rec_full, pixel_size_angstrom=pixel_size)
    b_fac = max(50.0, guinier.fitted_b_factor) if abs(guinier.fitted_b_factor) < 1000.0 else 120.0
    sharpened = BFactorSharpener.sharpen_map(rec_full, b_factor=b_fac, pixel_size_angstrom=pixel_size, high_res_cutoff_angstrom=fsc_res.resolution_fsc_0143_angstrom)

    print(f"      Fitted Overall B-Factor: {guinier.fitted_b_factor:.1f} A^2")
    print(f"      Sharpened Density Volume Dynamic Range: [{np.min(sharpened):.4f}, {np.max(sharpened):.4f}]")
    print("\n3D Central Orthogonal Cross-Sections (Sharpened Reconstruction):")
    print(AsciiVisualizer.render_orthoslices(sharpened))
    print("=" * 70)
    print("Pipeline Execution Completed Successfully.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI Entry point."""
    parser = argparse.ArgumentParser(
        prog="spectro-bio",
        description="SpectroBio: Cryo-EM 3D Single-Particle Reconstruction, CTF Estimation, and FSC Resolution Analysis",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Sub-commands")

    # CTF command
    p_ctf = subparsers.add_parser("ctf", help="Contrast Transfer Function (CTF) diagnostics")
    p_ctf.add_argument("--voltage", type=float, default=300.0, help="Accelerating voltage in kV (default: 300.0)")
    p_ctf.add_argument("--defocus", type=float, default=15000.0, help="Defocus in Angstroms (default: 15000.0)")
    p_ctf.add_argument("--astigmatism", type=float, default=0.0, help="Defocus astigmatism in Angstroms (default: 0.0)")
    p_ctf.add_argument("--astig-angle", type=float, default=0.0, help="Astigmatism angle in degrees (default: 0.0)")
    p_ctf.add_argument("--cs", type=float, default=2.7, help="Spherical aberration Cs in mm (default: 2.7)")
    p_ctf.add_argument("--amp-ratio", type=float, default=0.07, help="Amplitude contrast ratio w (default: 0.07)")
    p_ctf.add_argument("--b-factor", type=float, default=0.0, help="B-factor envelope in Å^2 (default: 0.0)")
    p_ctf.add_argument("--max-freq", type=float, default=0.5, help="Maximum spatial frequency in 1/Å (default: 0.5)")

    # FSC command
    p_fsc = subparsers.add_parser("fsc", help="Fourier Shell Correlation (FSC) resolution calculation")
    p_fsc.add_argument("--size", type=int, default=32, help="Phantom grid cube size (default: 32)")
    p_fsc.add_argument("--pixel-size", type=float, default=1.0, help="Pixel size in Angstroms (default: 1.0)")
    p_fsc.add_argument("--noise", type=float, default=0.2, help="Noise standard deviation (default: 0.2)")

    # Pipeline command
    p_pipe = subparsers.add_parser("pipeline", help="Run full end-to-end single particle reconstruction demo")
    p_pipe.add_argument("--size", type=int, default=32, help="Reconstruction grid size (default: 32)")
    p_pipe.add_argument("--pixel-size", type=float, default=1.0, help="Pixel size in Angstroms (default: 1.0)")
    p_pipe.add_argument("--n-projections", type=int, default=40, help="Number of angular projections (default: 40)")
    p_pipe.add_argument("--snr", type=float, default=2.0, help="Signal-to-noise ratio (default: 2.0)")

    args = parser.parse_args(argv)

    if args.subcommand == "ctf":
        return cmd_ctf(args)
    elif args.subcommand == "fsc":
        return cmd_fsc(args)
    elif args.subcommand == "pipeline":
        return cmd_pipeline(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
