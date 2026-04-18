## `model_simplifier`

This folder provides scripts to simplify mesh assets (STL/OBJ/PLY) using multiple approaches:

- **Decimation**
  - Blender (headless)
  - MeshLab (meshlabserver)
- **Remeshing + simplification**
  - Blender (voxel remesh + decimate)
  - MeshLab (isotropic remeshing + decimation)
- **Convex decomposition**
  - VHACD
  - CoACD

Entry point:

- `run_model_simplifier.py` (reads `model_simplifier_config.yaml` and dispatches to the chosen method)

Example input meshes live in:

- `example/meshes_input/`

