"""ASCII and terminal visualization tools for Cryo-EM power spectra, FSC curves, and 3D densities."""

from __future__ import annotations
import numpy as np
from spectro_bio.validation.fsc import FSCResult


class AsciiVisualizer:
    """Terminal ASCII visualizer for Cryo-EM diagnostics."""

    CHARS = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"]

    @classmethod
    def render_slice(
        cls,
        image_2d: np.ndarray,
        width: int = 40,
        height: int = 20,
    ) -> str:
        """Render a 2D slice or projection as an ASCII grayscale map."""
        # Normalize to [0, 1]
        img_min = float(np.min(image_2d))
        img_max = float(np.max(image_2d))
        if img_max > img_min:
            norm = (image_2d - img_min) / (img_max - img_min)
        else:
            norm = np.zeros_like(image_2d)

        ny, nx = norm.shape
        y_indices = np.linspace(0, ny - 1, height).astype(int)
        x_indices = np.linspace(0, nx - 1, width).astype(int)

        sampled = norm[y_indices[:, None], x_indices]
        char_idx = np.clip((sampled * (len(cls.CHARS) - 1)).astype(int), 0, len(cls.CHARS) - 1)

        lines = []
        lines.append("+" + "-" * width + "+")
        for r in range(height):
            line_str = "".join(cls.CHARS[c] for c in char_idx[r])
            lines.append(f"|{line_str}|")
        lines.append("+" + "-" * width + "+")
        return "\n".join(lines)

    @classmethod
    def render_fsc_curve(
        cls,
        fsc_result: FSCResult,
        width: int = 50,
        height: int = 15,
    ) -> str:
        """Render ASCII plot of the Fourier Shell Correlation curve."""
        freqs = fsc_result.spatial_frequencies_inv_a
        vals = fsc_result.fsc_values
        if len(freqs) == 0:
            return "No FSC data available."

        grid = [[" " for _ in range(width)] for _ in range(height)]
        
        # FSC axis range: [0.0, 1.0]
        # Y=0 is 1.0, Y=height-1 is 0.0
        y_cut_05 = int(round((1.0 - 0.50) * (height - 1)))
        y_cut_0143 = int(round((1.0 - 0.143) * (height - 1)))

        # Draw guideline markers
        for c in range(width):
            if 0 <= y_cut_05 < height:
                grid[y_cut_05][c] = "."
            if 0 <= y_cut_0143 < height:
                grid[y_cut_0143][c] = "-"

        # Resample FSC curve across columns
        col_freqs = np.linspace(freqs[0], freqs[-1], width)
        col_fsc = np.interp(col_freqs, freqs, vals)

        for col, v in enumerate(col_fsc):
            v_clamped = max(0.0, min(1.0, float(v)))
            row = int(round((1.0 - v_clamped) * (height - 1)))
            row = min(height - 1, max(0, row))
            grid[row][col] = "#"

        out_lines = []
        out_lines.append(f"  FSC Curve (Resolution @ 0.143: {fsc_result.resolution_fsc_0143_angstrom:.2f} A)")
        out_lines.append(f"  {'1.0':>5} +" + "-" * width + "+")
        for r in range(height):
            label = ""
            if r == 0:
                label = " 1.0 "
            elif r == y_cut_05:
                label = " 0.5 "
            elif r == y_cut_0143:
                label = "0.143"
            elif r == height - 1:
                label = " 0.0 "
            else:
                label = "     "

            row_str = "".join(grid[r])
            out_lines.append(f"  {label} |{row_str}|")
        out_lines.append(f"        +" + "-" * width + "+")
        out_lines.append(f"        0.00{'':^{width-10}}{freqs[-1]:.3f} 1/A")
        return "\n".join(out_lines)

    @classmethod
    def render_orthoslices(
        cls,
        volume: np.ndarray,
        width: int = 24,
        height: int = 12,
    ) -> str:
        """Render Axial (XY), Coronal (XZ), and Sagittal (YZ) central slices side-by-side."""
        nz, ny, nx = volume.shape
        cz, cy, cx = nz // 2, ny // 2, nx // 2

        xy = volume[cz, :, :]
        xz = volume[:, cy, :]
        yz = volume[:, :, cx]

        rendered_xy = cls.render_slice(xy, width=width, height=height).split("\n")
        rendered_xz = cls.render_slice(xz, width=width, height=height).split("\n")
        rendered_yz = cls.render_slice(yz, width=width, height=height).split("\n")

        headers = f"{'Axial (Z=' + str(cz) + ')':^{width+2}}  {'Coronal (Y=' + str(cy) + ')':^{width+2}}  {'Sagittal (X=' + str(cx) + ')':^{width+2}}"
        lines = [headers]
        for row_xy, row_xz, row_yz in zip(rendered_xy, rendered_xz, rendered_yz):
            lines.append(f"{row_xy}  {row_xz}  {row_yz}")

        return "\n".join(lines)
