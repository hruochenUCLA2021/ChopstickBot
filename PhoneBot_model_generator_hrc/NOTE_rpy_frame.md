# NOTE: `rpy` + frame conventions in this PhoneBot generator

This file is a “memory aid” for how `xyz` / `rpy` are interpreted in:

- `model_config.yaml`
- `model_symetric_generator.py` (URDF generation)
- URDF → MJCF conversion (MuJoCo)

## URDF `origin.rpy` convention (very important)

In URDF, `origin rpy = [roll, pitch, yaw]` means:

$$R = R_z(yaw)\,R_y(pitch)\,R_x(roll)$$

Two equivalent ways to think about the same rotation:

- **Extrinsic (fixed / parent axes)**: rotate about **fixed X**, then **fixed Y**, then **fixed Z**.
- **Intrinsic (moving / local axes)**: rotate about **local Z**, then **local Y**, then **local X**.

If you try to describe it as “do X then do Z” without stating intrinsic/extrinsic, you can get the wrong result.

## What frame are `xyz` / `rpy` expressed in?

In URDF, a joint’s `<origin xyz="..." rpy="...">` is **expressed in the parent link frame**.

That means:

- `xyz`: translation in the **parent frame**
- `rpy`: rotation about the **parent frame’s fixed axes** (per URDF convention above)

## PhoneBot YAML: meaning of `joint_origin_xyz` and `mount_xyz/mount_rpy`

For each chain stage in `left_leg.chain` (conceptually `parent -> child`):

- **`joint_origin_xyz`**: where the stage joint is located, expressed in the **parent stage output frame**.
- **`mount_xyz` / `mount_rpy`**: the pose of the **child motor link frame**, expressed in the **same parent stage output frame**, applied after you conceptually “go to the joint”.

In the current implementation, the URDF joint origin is built using:

- `origin_xyz = joint_origin_xyz + mount_xyz`
- `origin_rpy = mount_rpy`

So `mount_rpy` is *not* in world coordinates; it is in the parent stage’s **local** frame.

## MuJoCo MJCF: joint axis frame

In MJCF:

- A `<joint axis="0 0 1">` is expressed in the **local coordinate frame of the body that contains the joint**.
- If that body has a non-identity quaternion, then the joint axis in **world space** is that rotation applied to `(0,0,1)`.

This is why a joint can say `axis="0 0 1"` in XML but *look* different depending on the body’s `quat`.

## About `*_joint_frame` helper bodies (when enabled)

When `*_joint_frame` helper links/bodies are enabled, they exist to:

- avoid MuJoCo “zero inertia” issues by giving small inertial to intermediate moving frames
- optionally support “motor fixed first, joint after motor” modeling

**Critical rule** (the bug we hit earlier):

- Do **not** “cancel” a motor mount rotation by applying an inverse `mount_rpy` on the joint_frame.
- If you do, multiple downstream joint frames can become aligned and joint axes can appear wrong / identical.


