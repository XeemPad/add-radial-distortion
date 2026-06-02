#!/usr/bin/env python3
"""Add COLMAP-style radial distortion to images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add radial distortion using COLMAP SIMPLE_RADIAL/RADIAL convention: "
            "f, cx, cy, k1[, k2], or fx, fy, cx, cy, k1[, k2]."
        )
    )
    parser.add_argument("input", type=Path, help="Input image file or directory.")
    parser.add_argument("output", type=Path, help="Output image file or directory.")
    parser.add_argument(
        "--k1",
        type=float,
        default=None,
        help="Additional COLMAP radial k1 warp to apply to the current image.",
    )
    parser.add_argument(
        "--k1-input",
        type=float,
        default=None,
        help="Known input image k1 for final-k1 mode.",
    )
    parser.add_argument(
        "--k1-delta",
        type=float,
        default=None,
        help="Delta added to --k1-input in final-k1 mode.",
    )
    parser.add_argument(
        "--k1-final",
        type=float,
        default=None,
        help="Target final k1 in final-k1 mode.",
    )
    parser.add_argument(
        "--k2",
        type=float,
        default=0.0,
        help="Optional COLMAP radial k2. Default: 0.0.",
    )
    parser.add_argument(
        "--f",
        type=float,
        default=None,
        help=(
            "Shared focal length in pixels. Used when --fx/--fy are not set. "
            "Default: max(image width, image height)."
        ),
    )
    parser.add_argument(
        "--fx",
        type=float,
        default=None,
        help="Focal length in x direction in pixels.",
    )
    parser.add_argument(
        "--fy",
        type=float,
        default=None,
        help="Focal length in y direction in pixels.",
    )
    parser.add_argument(
        "--cx",
        type=float,
        default=None,
        help="Principal point x in pixels. Default: image width / 2.",
    )
    parser.add_argument(
        "--cy",
        type=float,
        default=None,
        help="Principal point y in pixels. Default: image height / 2.",
    )
    parser.add_argument(
        "--crop-valid",
        action="store_true",
        help="Crop output to pixels sampled from real input pixels.",
    )
    parser.add_argument(
        "--interpolation",
        choices=("nearest", "linear", "cubic", "lanczos"),
        default="linear",
        help="OpenCV interpolation mode. Default: linear.",
    )
    parser.add_argument(
        "--border-value",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "Fill value for invalid pixels without --crop-valid. "
            "Pass one value for grayscale, or B G R [A]. Default: 0."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When input is a directory, process images recursively.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )
    args = parser.parse_args()
    direct_mode = args.k1 is not None
    final_mode = (
        args.k1_input is not None
        or args.k1_delta is not None
        or args.k1_final is not None
    )
    if direct_mode == final_mode:
        parser.error("pass either --k1, or --k1-input with --k1-delta/--k1-final")
    if final_mode:
        if args.k1_input is None:
            parser.error("final-k1 mode requires --k1-input")
        if (args.k1_delta is None) == (args.k1_final is None):
            parser.error("pass exactly one of --k1-delta or --k1-final")
        if args.k2 != 0.0:
            parser.error("--k2 is only supported with direct --k1 mode")
    if args.f is not None and (args.fx is not None or args.fy is not None):
        parser.error("pass either --f or --fx/--fy, not both")
    if (args.fx is None) != (args.fy is None):
        parser.error("pass both --fx and --fy for unequal focal lengths")
    return args


def image_paths(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    paths = input_path.rglob("*") if recursive else input_path.glob("*")
    return sorted(
        path
        for path in paths
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def interpolation_flag(name: str) -> int:
    return {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }[name]


def make_border_value(values: list[int], channels: int) -> tuple[int, ...]:
    if len(values) == 1:
        return tuple([values[0]] * channels)
    if len(values) != channels:
        raise ValueError(
            f"Border value has {len(values)} components, expected 1 or {channels}."
        )
    return tuple(values)


def resolve_focal_lengths(
    width: int,
    height: int,
    f: float | None,
    fx: float | None,
    fy: float | None,
) -> tuple[float, float]:
    if fx is None and fy is None:
        focal = float(f if f is not None else max(width, height))
        return focal, focal
    if fx is None or fy is None:
        raise ValueError("Pass both --fx and --fy for unequal focal lengths.")
    return float(fx), float(fy)


def invert_radial_radius(
    rd: np.ndarray,
    k1: float,
    k2: float,
    iterations: int = 8,
) -> np.ndarray:
    """Solve rd = r * (1 + k1*r^2 + k2*r^4) for r with Newton iterations."""
    r = rd.copy()
    for _ in range(iterations):
        r2 = r * r
        r4 = r2 * r2
        f = r * (1.0 + k1 * r2 + k2 * r4) - rd
        df = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r4
        safe_df = np.where(np.abs(df) < 1e-12, 1e-12, df)
        r = np.maximum(r - f / safe_df, 0.0)
    return r


def distortion_maps(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float,
    k2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float32) + 0.5,
        np.arange(height, dtype=np.float32) + 0.5,
    )
    xd = (xs.astype(np.float64) - cx) / fx
    yd = (ys.astype(np.float64) - cy) / fy
    rd = np.hypot(xd, yd)

    r = invert_radial_radius(rd, k1, k2)
    scale = np.divide(r, rd, out=np.ones_like(rd), where=rd > 1e-12)

    src_x = (cx + fx * xd * scale - 0.5).astype(np.float32)
    src_y = (cy + fy * yd * scale - 0.5).astype(np.float32)
    valid = (
        (src_x >= 0.0)
        & (src_x <= width - 1.0)
        & (src_y >= 0.0)
        & (src_y <= height - 1.0)
    )
    return src_x, src_y, valid


def final_k1_maps(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1_input: float,
    k1_final: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float32) + 0.5,
        np.arange(height, dtype=np.float32) + 0.5,
    )
    x_final = (xs.astype(np.float64) - cx) / fx
    y_final = (ys.astype(np.float64) - cy) / fy
    r_final = np.hypot(x_final, y_final)

    r_undistorted = invert_radial_radius(r_final, k1_final, 0.0)
    final_scale = np.divide(
        r_undistorted,
        r_final,
        out=np.ones_like(r_final),
        where=r_final > 1e-12,
    )
    x_undistorted = x_final * final_scale
    y_undistorted = y_final * final_scale

    r2 = x_undistorted * x_undistorted + y_undistorted * y_undistorted
    input_scale = 1.0 + k1_input * r2
    src_x = (cx + fx * x_undistorted * input_scale - 0.5).astype(np.float32)
    src_y = (cy + fy * y_undistorted * input_scale - 0.5).astype(np.float32)
    valid = (
        (src_x >= 0.0)
        & (src_x <= width - 1.0)
        & (src_y >= 0.0)
        & (src_y <= height - 1.0)
    )
    return src_x, src_y, valid


def valid_crop(mask: np.ndarray) -> tuple[slice, slice]:
    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        raise ValueError("No valid pixels remain after distortion.")
    return slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1)


def distort_image(
    image: np.ndarray,
    k1: float,
    k2: float,
    k1_input: float | None,
    k1_final: float | None,
    f: float | None,
    fx: float | None,
    fy: float | None,
    cx: float | None,
    cy: float | None,
    crop_valid: bool,
    interpolation: int,
    border_value: tuple[int, ...],
) -> np.ndarray:
    height, width = image.shape[:2]
    focal_x, focal_y = resolve_focal_lengths(width, height, f, fx, fy)
    principal_x = float(cx if cx is not None else width / 2.0)
    principal_y = float(cy if cy is not None else height / 2.0)

    if k1_input is None or k1_final is None:
        map_x, map_y, valid = distortion_maps(
            width, height, focal_x, focal_y, principal_x, principal_y, k1, k2
        )
    else:
        map_x, map_y, valid = final_k1_maps(
            width,
            height,
            focal_x,
            focal_y,
            principal_x,
            principal_y,
            k1_input,
            k1_final,
        )
    distorted = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    if not crop_valid:
        return distorted

    row_slice, col_slice = valid_crop(valid)
    return distorted[row_slice, col_slice]


def output_path_for(input_file: Path, input_root: Path, output: Path) -> Path:
    if input_root.is_file():
        return output
    return output / input_file.relative_to(input_root)


def write_image(path: Path, image: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists, pass --overwrite to replace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def main() -> None:
    args = parse_args()
    inputs = image_paths(args.input, args.recursive)
    if not inputs:
        raise FileNotFoundError(f"No supported images found in: {args.input}")

    for input_file in inputs:
        image = cv2.imread(str(input_file), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Failed to read image: {input_file}")

        channels = 1 if image.ndim == 2 else image.shape[2]
        border_value = make_border_value(args.border_value, channels)
        direct_k1 = args.k1 if args.k1 is not None else 0.0
        final_k1 = None
        if args.k1_input is not None:
            final_k1 = (
                args.k1_final
                if args.k1_final is not None
                else args.k1_input + args.k1_delta
            )
        distorted = distort_image(
            image=image,
            k1=direct_k1,
            k2=args.k2,
            k1_input=args.k1_input,
            k1_final=final_k1,
            f=args.f,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            crop_valid=args.crop_valid,
            interpolation=interpolation_flag(args.interpolation),
            border_value=border_value,
        )
        output_file = output_path_for(input_file, args.input, args.output)
        write_image(output_file, distorted, args.overwrite)
        print(f"{input_file} -> {output_file}")


if __name__ == "__main__":
    main()
