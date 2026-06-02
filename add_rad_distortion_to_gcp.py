#!/usr/bin/env python3
"""Transform GCP pixel coordinates with the same radial model as image warping."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from add_rad_distortion import (
    distortion_maps,
    final_k1_maps,
    resolve_focal_lengths,
    valid_crop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform pixel coordinates in a GCP list using COLMAP-style radial distortion."
    )
    parser.add_argument("input", type=Path, help="Input GCP list.")
    parser.add_argument("output", type=Path, help="Output GCP list.")
    parser.add_argument(
        "--k1",
        type=float,
        default=None,
        help="Additional COLMAP radial k1 warp to apply to the current coordinates.",
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
        help="Optional COLMAP radial k2 for direct --k1 mode. Default: 0.0.",
    )
    parser.add_argument(
        "--f",
        type=float,
        default=None,
        help="Shared focal length in pixels. Used when --fx/--fy are not set.",
    )
    parser.add_argument("--fx", type=float, default=None, help="Focal length x.")
    parser.add_argument("--fy", type=float, default=None, help="Focal length y.")
    parser.add_argument("--cx", type=float, required=True, help="Principal point x.")
    parser.add_argument("--cy", type=float, required=True, help="Principal point y.")
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Target image width after coordinate resize. Alias for --image-width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Target image height after coordinate resize. Alias for --image-height.",
    )
    parser.add_argument(
        "--input-width",
        type=float,
        default=None,
        help="Original GCP coordinate image width before resize.",
    )
    parser.add_argument(
        "--input-height",
        type=float,
        default=None,
        help="Original GCP coordinate image height before resize.",
    )
    parser.add_argument(
        "--image-width",
        type=float,
        default=None,
        help="Target image width after coordinate resize.",
    )
    parser.add_argument(
        "--image-height",
        type=float,
        default=None,
        help="Target image height after coordinate resize.",
    )
    parser.add_argument(
        "--crop-valid",
        action="store_true",
        help="Apply the same valid-rectangle crop offset as the image script.",
    )
    parser.add_argument(
        "--x-col",
        type=int,
        default=4,
        help="1-based x pixel coordinate column. Default: 4.",
    )
    parser.add_argument(
        "--y-col",
        type=int,
        default=5,
        help="1-based y pixel coordinate column. Default: 5.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=12,
        help="Decimal digits for transformed pixel coordinates. Default: 12.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output file.",
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
    if args.x_col < 1 or args.y_col < 1:
        parser.error("--x-col and --y-col are 1-based and must be positive")
    if args.width is not None and args.image_width is not None:
        parser.error("pass either --width or --image-width, not both")
    if args.height is not None and args.image_height is not None:
        parser.error("pass either --height or --image-height, not both")
    args.image_width = args.image_width if args.image_width is not None else args.width
    args.image_height = args.image_height if args.image_height is not None else args.height
    if (args.input_width is None) != (args.input_height is None):
        parser.error("pass both --input-width and --input-height")
    if args.input_width is not None and (
        args.image_width is None or args.image_height is None
    ):
        parser.error("--input-width/--input-height require --image-width/--image-height")
    if args.f is None and args.fx is None and (
        args.image_width is None or args.image_height is None
    ):
        parser.error(
            "pass --f, --fx/--fy, or --image-width/--image-height for focal-length default"
        )
    if args.crop_valid and (args.image_width is None or args.image_height is None):
        parser.error("--crop-valid requires --image-width/--image-height")
    return args


def format_float(value: float, precision: int) -> str:
    text = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    return text


def invert_radius_scalar(rd: float, k1: float, k2: float = 0.0) -> tuple[float, bool]:
    if rd <= 1e-12:
        return 0.0, True

    if k2 == 0.0 and k1 < 0.0:
        high = float(np.sqrt(-1.0 / (3.0 * k1)))
        high_rd = high * (1.0 + k1 * high * high)
        if rd > high_rd + 1e-9:
            return high, False
    else:
        high = max(1.0, rd)
        for _ in range(64):
            high2 = high * high
            high_rd = high * (1.0 + k1 * high2 + k2 * high2 * high2)
            if high_rd >= rd:
                break
            high *= 2.0
        else:
            return high, False

    low = 0.0
    for _ in range(64):
        mid = 0.5 * (low + high)
        mid2 = mid * mid
        mid_rd = mid * (1.0 + k1 * mid2 + k2 * mid2 * mid2)
        if mid_rd <= rd:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high), True


def distort_normalized(x: float, y: float, k1: float, k2: float = 0.0) -> tuple[float, float]:
    r2 = x * x + y * y
    scale = 1.0 + k1 * r2 + k2 * r2 * r2
    return x * scale, y * scale


def undistort_normalized(xd: float, yd: float, k1: float, k2: float = 0.0) -> tuple[float, float, bool]:
    rd = float(np.hypot(xd, yd))
    r, valid = invert_radius_scalar(rd, k1, k2)
    scale = 1.0 if rd <= 1e-12 else r / rd
    return xd * scale, yd * scale, valid


def transform_point(
    x: float,
    y: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float | None,
    k2: float,
    k1_input: float | None,
    k1_final: float | None,
) -> tuple[float, float, bool]:
    xn = (x - cx) / fx
    yn = (y - cy) / fy

    if k1_input is None or k1_final is None:
        x_out, y_out = distort_normalized(xn, yn, float(k1), k2)
        return cx + fx * x_out, cy + fy * y_out, True

    xu, yu, valid = undistort_normalized(xn, yn, k1_input)
    x_out, y_out = distort_normalized(xu, yu, k1_final)
    return cx + fx * x_out, cy + fy * y_out, valid


def crop_offset(args: argparse.Namespace, fx: float, fy: float) -> tuple[int, int, slice, slice] | None:
    if not args.crop_valid:
        return None

    direct_k1 = args.k1 if args.k1 is not None else 0.0
    final_k1 = None
    if args.k1_input is not None:
        final_k1 = args.k1_final if args.k1_final is not None else args.k1_input + args.k1_delta
        if np.isclose(final_k1, args.k1_input, rtol=0.0, atol=1e-15):
            row_slice = slice(0, int(args.image_height))
            col_slice = slice(0, int(args.image_width))
            return 0, 0, row_slice, col_slice
    elif np.isclose(direct_k1, 0.0, rtol=0.0, atol=1e-15) and np.isclose(
        args.k2, 0.0, rtol=0.0, atol=1e-15
    ):
        row_slice = slice(0, int(args.image_height))
        col_slice = slice(0, int(args.image_width))
        return 0, 0, row_slice, col_slice

    if args.k1_input is None:
        _, _, valid = distortion_maps(
            int(args.image_width),
            int(args.image_height),
            fx,
            fy,
            args.cx,
            args.cy,
            direct_k1,
            args.k2,
        )
    else:
        _, _, valid = final_k1_maps(
            int(args.image_width),
            int(args.image_height),
            fx,
            fy,
            args.cx,
            args.cy,
            args.k1_input,
            final_k1,
        )

    row_slice, col_slice = valid_crop(valid)
    return col_slice.start, row_slice.start, row_slice, col_slice


def transform_gcp_file(args: argparse.Namespace) -> None:
    width = int(args.image_width) if args.image_width is not None else 1
    height = int(args.image_height) if args.image_height is not None else 1
    fx, fy = resolve_focal_lengths(width, height, args.f, args.fx, args.fy)
    crop = crop_offset(args, fx, fy)
    x_offset = crop[0] if crop is not None else 0
    y_offset = crop[1] if crop is not None else 0

    final_k1 = None
    if args.k1_input is not None:
        final_k1 = args.k1_final if args.k1_final is not None else args.k1_input + args.k1_delta
    unchanged_distortion = (
        args.k1_input is not None
        and final_k1 is not None
        and np.isclose(final_k1, args.k1_input, rtol=0.0, atol=1e-15)
    ) or (
        args.k1 is not None
        and np.isclose(args.k1, 0.0, rtol=0.0, atol=1e-15)
        and np.isclose(args.k2, 0.0, rtol=0.0, atol=1e-15)
    )

    x_idx = args.x_col - 1
    y_idx = args.y_col - 1
    scale_x = 1.0
    scale_y = 1.0
    if args.input_width is not None:
        scale_x = float(args.image_width) / float(args.input_width)
        scale_y = float(args.image_height) / float(args.input_height)
    invalid_points = 0
    transformed_lines: list[str] = []

    for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            transformed_lines.append(line)
            continue

        parts = stripped.split()
        if len(parts) <= max(x_idx, y_idx):
            transformed_lines.append(line)
            continue

        try:
            x = float(parts[x_idx])
            y = float(parts[y_idx])
        except ValueError:
            transformed_lines.append(line)
            continue

        resized_x = x * scale_x
        resized_y = y * scale_y

        if unchanged_distortion:
            new_x, new_y, valid = resized_x, resized_y, True
        else:
            new_x, new_y, valid = transform_point(
                resized_x,
                resized_y,
                fx,
                fy,
                args.cx,
                args.cy,
                args.k1,
                args.k2,
                args.k1_input,
                final_k1,
            )
        if not valid:
            invalid_points += 1

        parts[x_idx] = format_float(new_x - x_offset, args.precision)
        parts[y_idx] = format_float(new_y - y_offset, args.precision)
        transformed_lines.append("\t".join(parts))

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists, pass --overwrite to replace: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(transformed_lines) + "\n", encoding="utf-8")

    if crop is not None:
        _, _, row_slice, col_slice = crop
        print(
            "Applied crop offset "
            f"x={col_slice.start}, y={row_slice.start}, "
            f"width={col_slice.stop - col_slice.start}, "
            f"height={row_slice.stop - row_slice.start}",
            file=sys.stderr,
        )
    if invalid_points:
        print(
            f"Warning: {invalid_points} points were outside the central invertible branch.",
            file=sys.stderr,
        )


def main() -> None:
    transform_gcp_file(parse_args())


if __name__ == "__main__":
    main()
