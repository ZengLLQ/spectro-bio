"""SpectroBio: Cryo-EM 3D Single-Particle Macromolecular Reconstruction, CTF Estimation, and FSC Resolution Analysis."""

from spectro_bio.optics.ctf import ContrastTransferFunction, CTFParameters, CTF1DProfile
from spectro_bio.optics.phase_flipping import PhaseCorrector
from spectro_bio.reconstruction.fourier_slice import EulerAngles, ProjectionGeometry, FourierSliceProjector
from spectro_bio.reconstruction.backprojector import FilteredBackprojector, ReconstructionGrid
from spectro_bio.validation.fsc import FourierShellCorrelation, FSCResult
from spectro_bio.validation.b_factor import BFactorSharpener, GuinierResult
from spectro_bio.visualizer import AsciiVisualizer

__version__ = "0.1.0"
__all__ = [
    "ContrastTransferFunction",
    "CTFParameters",
    "CTF1DProfile",
    "PhaseCorrector",
    "EulerAngles",
    "ProjectionGeometry",
    "FourierSliceProjector",
    "FilteredBackprojector",
    "ReconstructionGrid",
    "FourierShellCorrelation",
    "FSCResult",
    "BFactorSharpener",
    "GuinierResult",
    "AsciiVisualizer",
]
