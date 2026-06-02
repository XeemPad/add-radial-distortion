# Add radial distortion to images

Tool to add radial distortion ($ k_1 \cdot r^2 $ only) to a set of images.

Uses the COLMAP `SIMPLE_RADIAL` / `RADIAL` convention for one shared focal
length, and the same normalized-coordinate convention with `fx`, `fy` for
unequal focal lengths:

```text
u = (x - cx) / fx
v = (y - cy) / fy
r2 = u*u + v*v
u_distorted = u * (1 + k1*r2 + k2*r2*r2)
v_distorted = v * (1 + k1*r2 + k2*r2*r2)
```

For `SIMPLE_RADIAL`, `fx = fy = f`.

COLMAP image coordinates are used: the upper-left image corner is `(0, 0)`,
so the upper-left pixel center is `(0.5, 0.5)`.

## Install

```bash
uv sync
```

## Usage

Single image:

```bash
.venv/bin/python add_rad_distortion.py input.jpg output.jpg --k1 0.1 --f 1200
```

Directory:

```bash
.venv/bin/python add_rad_distortion.py images distorted --k1 0.1 --f 1200
```

This mode applies an additional warp with `k1 = 0.1` to the current image. If
the input image already has radial distortion, this is not the same as exact
physical addition to the original camera coefficient.

Optional shared camera parameters:

```bash
.venv/bin/python add_rad_distortion.py images distorted --k1 0.1 --k2 0.01 --f 1200 --cx 960 --cy 540
```

Unequal focal lengths:

```bash
.venv/bin/python add_rad_distortion.py images distorted --k1 0.1 --fx 1200 --fy 1180 --cx 960 --cy 540
```

Exact final `k1` mode for images with known input distortion:

```bash
.venv/bin/python add_rad_distortion.py images distorted --k1-input 0.05 --k1-delta -0.1 --f 1200
```

This targets:

```text
k1_final = k1_input + k1_delta
```

You can also pass the target directly:

```bash
.venv/bin/python add_rad_distortion.py images distorted --k1-input 0.05 --k1-final -0.05 --f 1200
```

Crop to the largest axis-aligned rectangle where every output pixel is sampled
from real input pixels:

```bash
.venv/bin/python add_rad_distortion.py images distorted --k1 -0.1 --f 1200 --crop-valid
```

If `--f`, `--fx`, and `--fy` are not provided, the script uses
`fx = fy = max(width, height)`. If `--cx` or `--cy` are not provided, it uses
`width / 2` and `height / 2`.

## GCP coordinates

Transform the 4th and 5th whitespace-separated fields in a GCP list with the
same radial model. If GCP coordinates are marked on full-resolution images, but
the distorted images are resized, pass both sizes. The script first maps GCP
coordinates to the resized image and then applies distortion:

```bash
.venv/bin/python add_rad_distortion_to_gcp.py My_GCP/gcp_list.txt My_GCP/gcp_list_distorted.txt --k1-input 0.01 --k1-delta -0.5 --f 1679 --cx 1065 --cy 694
```

Full-resolution GCP coordinates to resized distorted images:

```bash
.venv/bin/python add_rad_distortion_to_gcp.py My_GCP/gcp_list.txt My_GCP/gcp_list_distorted.txt --k1-input 0.01 --k1-delta -0.5 --f 1679 --cx 1065 --cy 694 --input-width 5472 --input-height 3648 --image-width 2120 --image-height 1413
```

If the distorted images were saved with `--crop-valid`, pass the original image
size and resized image size so the script can subtract the same crop offset:

```bash
.venv/bin/python add_rad_distortion_to_gcp.py My_GCP/gcp_list.txt My_GCP/gcp_list_distorted.txt --k1-input 0.01 --k1-delta -0.5 --f 1679 --cx 1065 --cy 694 --input-width 5472 --input-height 3648 --image-width 2120 --image-height 1413 --crop-valid
```
