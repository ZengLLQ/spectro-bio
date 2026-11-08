"""Cryo-EM Single-Particle Reconstruction of Trimeric Viral Spike Macromolecule."""

from __future__ import annotations
import numpy as np

from spectro_bio.optics.ctf import ContrastTransferFunction, CTFParameters
from spectro_bio.optics.phase_flipping import PhaseCorrector
from spectro_bio.reconstruction.fourier_slice import EulerAngles, ProjectionGeometry, FourierSliceProjector
from spectro_bio.reconstruction.backprojector import FilteredBackprojector
from spectro_bio.validation.fsc import FourierShellCorrelation
from spectro_bio.validation.b_factor import BFactorSharpener
from spectro_bio.visualizer import AsciiVisualizer


def generate_trimeric_spike_phantom(size: int = 36) -> np.ndarray:
    """
    Generate synthetic 3D Coulomb potential density resembling a trimeric viral spike glycoprotein.
    Has C3 threefold rotational symmetry with three receptor-binding domains (RBD) and central stalk.
    """
    vol = np.zeros((size, size, size), dtype=np.float32)
    c = size // 2
    r_stalk = size // 6

    z, y, x = np.ogrid[:size, :size, :size]

    # Central stalk (S2 subunit) along z-axis
    r2_stalk = (x - c)**2 + (y - c)**2
    z_mask = (z >= c - r_stalk * 1.5) & (z <= c + r_stalk * 1.2)
    vol[z_mask & (r2_stalk <= r_stalk**2)] += 1.0

    # Three globular S1 heads spaced at 120-degree intervals
    angles_deg = [0, 120, 240]
    head_dist = size // 4
    head_radius = size // 7
    head_z = c + r_stalk // 2

    for ang in angles_deg:
        rad = np.radians(ang)
        hx = c + head_dist * np.cos(rad)
        hy = c + head_dist * np.sin(rad)
        d_head = np.sqrt((x - hx)**2 + (y - hy)**2 + (z - head_z)**2)
        mask = d_head <= head_radius
        vol[mask] += 1.2 * np.exp(-0.5 * (d_head[mask] / (head_radius * 0.6))**2)

    return vol


def main():
    print("=" * 75)
    print("SpectroBio :: Simulation of Cryo-EM Trimeric Spike Single-Particle Reconstruction")
    print("=" * 75)

    size = 32
    pixel_size = 1.05  # A per pixel (typical K3 direct electron detector super-resolution)
    n_particles = 48
    print(f"Macromolecule: C3 Symmetric Trimeric Spike (Grid: {size}x{size}x{size}, {pixel_size:.2f} A/px)")
    print(f"Acquiring {n_particles} cryo-EM micrographs on simulated 300 kV Titan Krios...")

    phantom = generate_trimeric_spike_phantom(size)

    # Microscope settings: 300 kV, Cs = 2.7 mm, mean defocus = 1.6 um
    ctf_params = CTFParameters(
        defocus_u_angstrom=16000.0,
        defocus_v_angstrom=16000.0,
        spherical_aberration_cs_mm=2.7,
        accelerating_voltage_kv=300.0,
        amplitude_contrast_ratio=0.08,
    )
    ctf_2d = ContrastTransferFunction.evaluate_2d_ctf(ctf_params, size, pixel_size)

    rng = np.random.default_rng(2026)
    particles = []
    geometries = []

    for i in range(n_particles):
        phi = float(rng.uniform(0, 360))
        theta = float(np.degrees(np.arccos(rng.uniform(-1, 1))))
        psi = float(rng.uniform(0, 360))
        geom = ProjectionGeometry(euler_angles=EulerAngles(phi, theta, psi))
        geometries.append(geom)

        proj = FourierSliceProjector.forward_project(phantom, geom)
        # Apply CTF modulation
        proj_ctf = np.real(np.fft.ifft2(np.fft.fft2(proj) * ctf_2d))
        # Add shot noise (SNR = 1.5)
        noisy = proj_ctf + rng.normal(0, np.std(proj_ctf) * 0.7, proj.shape)
        # Phase correction
        corr = PhaseCorrector.correct_phase_flipping(noisy, ctf_2d)
        particles.append(corr)

    print("Reconstructing Gold-Standard independent half-datasets...")
    h1_p, h1_g = particles[0::2], geometries[0::2]
    h2_p, h2_g = particles[1::2], geometries[1::2]

    rec_h1 = FilteredBackprojector.reconstruct(h1_p, h1_g, output_shape=(size, size, size), filter_type="shepp_logan")
    rec_h2 = FilteredBackprojector.reconstruct(h2_p, h2_g, output_shape=(size, size, size), filter_type="shepp_logan")
    rec_full = FilteredBackprojector.reconstruct(particles, geometries, output_shape=(size, size, size), filter_type="shepp_logan")

    print("\nEvaluating Fourier Shell Correlation (FSC)...")
    fsc = FourierShellCorrelation.compute_fsc(rec_h1, rec_h2, pixel_size_angstrom=pixel_size)
    print(AsciiVisualizer.render_fsc_curve(fsc))

    print(f"\nGold-Standard FSC 0.143 Resolution: {fsc.resolution_fsc_0143_angstrom:.2f} A")
    print(f"Conservative FSC 0.50 Resolution:   {fsc.resolution_fsc_05_angstrom:.2f} A")

    print("\nApplying Rosenthal-Henderson Guinier B-factor map sharpening...")
    guinier = BFactorSharpener.estimate_b_factor(rec_full, pixel_size_angstrom=pixel_size)
    sharpened = BFactorSharpener.sharpen_map(
        rec_full,
        b_factor=max(60.0, guinier.fitted_b_factor),
        pixel_size_angstrom=pixel_size,
        high_res_cutoff_angstrom=fsc.resolution_fsc_0143_angstrom,
    )

    print("\nReconstructed Trimeric Spike Density Slices:")
    print(AsciiVisualizer.render_orthoslices(sharpened))
    print("=" * 75)


if __name__ == "__main__":
    main()
