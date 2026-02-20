from __future__ import annotations

import math
from typing import Iterable


def fmt_xyz(x: float, y: float, z: float) -> str:
    return f"{x:.6f} {y:.6f} {z:.6f}"


def fmt_quat_wxyz(q: tuple[float, float, float, float]) -> str:
    # MuJoCo expects (w x y z)
    return f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"


def as_vec3(x: object) -> tuple[float, float, float]:
    if not (isinstance(x, (list, tuple)) and len(x) == 3):
        raise ValueError(f"Expected 3-vector, got: {x!r}")
    return (float(x[0]), float(x[1]), float(x[2]))


def matmul3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ],
        [
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ],
    ]


def rotx(r: float) -> list[list[float]]:
    c = math.cos(r)
    s = math.sin(r)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def roty(p: float) -> list[list[float]]:
    c = math.cos(p)
    s = math.sin(p)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def rotz(y: float) -> list[list[float]]:
    c = math.cos(y)
    s = math.sin(y)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def R_from_rpy(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    # URDF/ROS convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    return matmul3(matmul3(rotz(yaw), roty(pitch)), rotx(roll))


def transpose3(R: list[list[float]]) -> list[list[float]]:
    return [
        [R[0][0], R[1][0], R[2][0]],
        [R[0][1], R[1][1], R[2][1]],
        [R[0][2], R[1][2], R[2][2]],
    ]


def matvec3(R: list[list[float]], v: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    )


def normalize3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < 1e-12:
        return (0.0, 0.0, 1.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def mirror_xyz_yz(xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    # Mirror across YZ plane: x -> -x
    return (-xyz[0], xyz[1], xyz[2])


def mirror_R_yz(R: list[list[float]]) -> list[list[float]]:
    # Reflection matrix M = diag(-1,1,1): Rm = M R M
    M = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return matmul3(matmul3(M, R), M)


def quat_from_R(R: list[list[float]]) -> tuple[float, float, float, float]:
    # Returns (w, x, y, z)
    tr = R[0][0] + R[1][1] + R[2][2]
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2][1] - R[1][2]) / S
        y = (R[0][2] - R[2][0]) / S
        z = (R[1][0] - R[0][1]) / S
    elif (R[0][0] > R[1][1]) and (R[0][0] > R[2][2]):
        S = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2.0
        w = (R[2][1] - R[1][2]) / S
        x = 0.25 * S
        y = (R[0][1] + R[1][0]) / S
        z = (R[0][2] + R[2][0]) / S
    elif R[1][1] > R[2][2]:
        S = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2.0
        w = (R[0][2] - R[2][0]) / S
        x = (R[0][1] + R[1][0]) / S
        y = 0.25 * S
        z = (R[1][2] + R[2][1]) / S
    else:
        S = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2.0
        w = (R[1][0] - R[0][1]) / S
        x = (R[0][2] + R[2][0]) / S
        y = (R[1][2] + R[2][1]) / S
        z = 0.25 * S

    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / n, x / n, y / n, z / n)


def rpy_from_R(R: list[list[float]]) -> tuple[float, float, float]:
    """
    Extract URDF/ROS rpy such that:
      R == Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    r20 = max(-1.0, min(1.0, R[2][0]))
    pitch = math.asin(-r20)
    cp = math.cos(pitch)
    if abs(cp) < 1e-9:
        roll = 0.0
        yaw = math.atan2(-R[0][1], R[1][1])
    else:
        roll = math.atan2(R[2][1], R[2][2])
        yaw = math.atan2(R[1][0], R[0][0])
    return (roll, pitch, yaw)


def box_diaginertia(mass: float, size_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = size_xyz
    ixx = (1.0 / 12.0) * mass * (y * y + z * z)
    iyy = (1.0 / 12.0) * mass * (x * x + z * z)
    izz = (1.0 / 12.0) * mass * (x * x + y * y)
    return (ixx, iyy, izz)


def cylinder_diaginertia_z(mass: float, radius: float, length: float) -> tuple[float, float, float]:
    """
    Solid cylinder inertia about its COM, cylinder axis along +Z.

    Ixx = Iyy = (1/12) m (3 r^2 + L^2)
    Izz = (1/2) m r^2
    """
    r2 = radius * radius
    ixx = (1.0 / 12.0) * mass * (3.0 * r2 + length * length)
    iyy = ixx
    izz = 0.5 * mass * r2
    return (ixx, iyy, izz)


def parallel_axis_add_diag(
    I_diag_com: tuple[float, float, float], mass: float, offset_xyz: tuple[float, float, float]
) -> tuple[float, float, float]:
    """
    Parallel-axis contribution to the *diagonal terms* when shifting from COM to another point.

    NOTE:
    - The full parallel-axis theorem introduces off-diagonal terms (-m*dx*dy, ...).
    - We expose this helper for completeness/debugging, but in URDF/MJCF you usually keep
      inertia about COM and set inertial origin/pos to COM instead.
    """
    dx, dy, dz = offset_xyz
    add_xx = mass * (dy * dy + dz * dz)
    add_yy = mass * (dx * dx + dz * dz)
    add_zz = mass * (dx * dx + dy * dy)
    return (I_diag_com[0] + add_xx, I_diag_com[1] + add_yy, I_diag_com[2] + add_zz)


def fmt_inertia_diag(ixx: float, iyy: float, izz: float) -> str:
    # Keep high precision to avoid MuJoCo triangle-inequality failures from rounding.
    return f"{ixx:.12f} {iyy:.12f} {izz:.12f}"


def require_keys(d: dict, keys: Iterable[str], ctx: str) -> None:
    for k in keys:
        if k not in d:
            raise KeyError(f"Missing required key {k!r} in {ctx}")


