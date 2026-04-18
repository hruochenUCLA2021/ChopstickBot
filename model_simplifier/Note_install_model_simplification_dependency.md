## Install dependencies for `ChopstickBot/model_simplifier`

You said you created a conda env called `model_convert`. The suggestions below keep installs isolated.

---

## 1) Create / activate the env

```bash
conda create -n model_convert python=3.10 -y
conda activate model_convert
```

Basic python deps used by the dispatcher/config:

```bash
pip install pyyaml
```

---

## 2) Decimation / Remesh+Simplify via Blender (headless)

### Install Blender

Options:
- System package (may be older)
- Download official Blender tarball

If `blender` is on your PATH, you can keep:
- `tools.blender.exe: blender`

Otherwise set `tools.blender.exe` in `model_simplifier_config.yaml` to the absolute path, e.g.:

```yaml
tools:
  blender:
    exe: /path/to/blender
```

Scripts:
- `decimation/blender/run.py`
- `remesh_simplify/blender/run.py`

---

## 3) Decimation / Remesh+Simplify via MeshLab (meshlabserver)

Install MeshLab so you have `meshlabserver` available.

on ubuntu i use : sudo apt install meshlab

Then keep:
- `tools.meshlab.exe: meshlabserver`

Or set it to an absolute path.

Scripts:
- `decimation/meshlab/run.py` (uses `filters_decimate.mlx`)
- `remesh_simplify/meshlab/run.py` (uses `filters_remesh_and_decimate.mlx`)

Note: MeshLab filter names/params can vary across versions. If meshlabserver errors that a filter
is missing, open the `.mlx` and adjust the filter name to match your MeshLab build.

---

## 4) Convex decomposition via CoACD (recommended)

This method is implemented in Python and expects:
- `coacd`
- `trimesh`
- `numpy`

Install:

```bash
pip install numpy trimesh coacd
```

Script:
- `convex_decomposition/coacd/run.py`

---

## 5) Convex decomposition via VHACD (legacy/optional)

This repo’s VHACD runner is wired to the **`TestVHACD`** executable you build from the VHACD repo.
It follows the same working flow you used before:

- convert STL → OBJ (needs `open3d` or `trimesh`)
- run: `TestVHACD <wavefront.obj> -o stl -h <max_hulls> -r <resolution>`
- move `decomp.stl` to the configured output filename

Install one of these for STL→OBJ conversion:

```bash
pip install open3d
# OR
pip install trimesh
```

Then set in `model_simplifier_config.yaml`:

```yaml
tools:
  vhacd:
    exe: /absolute/path/to/TestVHACD
```

---

## 6) How to run

Edit `model_simplifier_config.yaml`:
- choose `method`
- set `io.input_dir` and `io.output_dir`

Then run:

```bash
python ChopstickBot/model_simplifier/run_model_simplifier.py \
  --config ChopstickBot/model_simplifier/model_simplifier_config.yaml
```

