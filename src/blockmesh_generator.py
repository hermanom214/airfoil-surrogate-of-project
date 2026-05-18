from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


@dataclass
class NACA4Params:
    camber_percent: int
    camber_position_tenths: int
    thickness_percent: int
    chord: float = 1.0


@dataclass
class CGridBlockMeshConfig:
    z_half: float = 0.05

    x_min: float = -5.0
    x_max: float = 12.0
    y_min: float = -5.0
    y_max: float = 5.0

    n_airfoil_half: int = 80
    le_cluster_exp: float = 2.5

    n_streamwise_le: int = 60  # Fine refinement in LE cluster block
    n_streamwise_near: int = 100
    n_wall_normal: int = 80
    n_wake_x: int = 120

    n_z: int = 1

    grading_to_wall: float = 800.0
    grading_from_wall: float = 0.00125
    grading_le_tangent: float = 0.45
    grading_wake_x: float = 8.0

    # Small artificial wake cut behind TE for block topology.
    # Keep small for first tests.
    wake_cut_length: float = 0.03

    # Geometric leading-edge cap smoothing (keeps topology unchanged).
    # This primarily targets the first few cells near LE.
    enable_le_cap: bool = True
    le_cap_fraction: float = 0.03
    le_cap_power: float = 1.15

    # Topological LE cap (removes the singular LE vertex in block topology).
    le_topology_fraction: float = 0.028
    n_le_cap_normal: int = 24


def cosine_spacing(n: int) -> List[float]:
    return [
        0.5 * (1.0 - math.cos(math.pi * i / (n - 1)))
        for i in range(n)
    ]


def le_clustered_spacing(n: int, cluster_exp: float = 2.5) -> List[float]:
    """Cosine spacing with clustering near leading edge.
    cluster_exp > 1: more clustering at x=0 (LE).
    cluster_exp = 1: same as regular cosine spacing.
    """
    base = cosine_spacing(n)
    if cluster_exp <= 0.0:
        raise ValueError("cluster_exp must be > 0")

    xs = [x ** cluster_exp for x in base]

    # Smooth only the first few parametric steps near LE to avoid visible
    # jagged transition in the first cells while keeping strong LE clustering.
    prefix = min(8, max(4, n // 40))
    if prefix >= 3 and prefix < n:
        anchor = xs[prefix - 1]
        for i in range(1, prefix - 1):
            s = i / (prefix - 1)
            xs[i] = anchor * (s**1.6)

    return xs


def thickness_distribution(x: float, thickness_fraction: float) -> float:
    return 5.0 * thickness_fraction * (
        0.2969 * math.sqrt(max(x, 1e-12))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )


def camber_line_and_slope(
    x: float,
    camber_fraction: float,
    camber_pos_fraction: float,
) -> tuple[float, float]:
    if camber_fraction == 0.0 or camber_pos_fraction == 0.0:
        return 0.0, 0.0

    m = camber_fraction
    p = camber_pos_fraction

    if x < p:
        yc = m / (p**2) * (2.0 * p * x - x**2)
        dyc_dx = 2.0 * m / (p**2) * (p - x)
    else:
        yc = m / ((1.0 - p) ** 2) * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2)
        dyc_dx = 2.0 * m / ((1.0 - p) ** 2) * (p - x)

    return yc, dyc_dx


def generate_naca4_upper_lower(
    params: NACA4Params,
    n_points: int,
    le_cluster_exp: float = 2.5,
) -> tuple[List[Point2D], List[Point2D]]:
    xs = le_clustered_spacing(n_points, cluster_exp=le_cluster_exp)

    m = params.camber_percent / 100.0
    p = params.camber_position_tenths / 10.0
    t = params.thickness_percent / 100.0
    c = params.chord

    upper: List[Point2D] = []
    lower: List[Point2D] = []

    for x_norm in xs:
        yt = thickness_distribution(x_norm, t)
        yc, dyc_dx = camber_line_and_slope(x_norm, m, p)
        theta = math.atan(dyc_dx)

        xu = x_norm - yt * math.sin(theta)
        yu = yc + yt * math.cos(theta)

        xl = x_norm + yt * math.sin(theta)
        yl = yc - yt * math.cos(theta)

        upper.append((xu * c, yu * c))
        lower.append((xl * c, yl * c))

    # force sharp TE
    y_te = 0.5 * (upper[-1][1] + lower[-1][1])
    upper[-1] = (c, y_te)
    lower[-1] = (c, y_te)

    return upper, lower


def apply_geometric_le_cap(
    upper: List[Point2D],
    lower: List[Point2D],
    cap_fraction: float,
    cap_power: float,
) -> tuple[List[Point2D], List[Point2D]]:
    """Replace first LE points by a smooth elliptical cap on each side.

    This reduces visible faceting in the first cells around the singular LE point
    while preserving the original block topology.
    """
    if not (0.0 < cap_fraction < 0.2):
        return upper, lower

    n = min(len(upper), len(lower))
    if n < 12:
        return upper, lower

    k = max(4, min(int(cap_fraction * (n - 1)), (n - 1) // 6))
    if k < 2:
        return upper, lower

    xu_end, yu_end = upper[k]
    xl_end, yl_end = lower[k]

    yu_abs = abs(yu_end)
    yl_abs = abs(yl_end)
    if xu_end <= 0.0 or xl_end <= 0.0 or yu_abs <= 0.0 or yl_abs <= 0.0:
        return upper, lower

    new_upper = upper[:]
    new_lower = lower[:]

    for i in range(k + 1):
        s = i / k
        t = (s**max(0.5, cap_power)) * (0.5 * math.pi)

        # Elliptic LE cap parameterization: x = a(1-cos t), y = b sin t.
        # It gives near-vertical tangent at LE and smooth transition to side.
        new_upper[i] = (xu_end * (1.0 - math.cos(t)), yu_abs * math.sin(t))
        new_lower[i] = (xl_end * (1.0 - math.cos(t)), -yl_abs * math.sin(t))

    return new_upper, new_lower


def rotate_point(
    p: Point2D,
    angle_deg: float,
    center: Point2D = (0.25, 0.0),
) -> Point2D:
    x, y = p
    cx, cy = center

    a = math.radians(angle_deg)
    dx = x - cx
    dy = y - cy

    xr = dx * math.cos(a) - dy * math.sin(a)
    yr = dx * math.sin(a) + dy * math.cos(a)

    return xr + cx, yr + cy


def rotate_points(points: List[Point2D], angle_deg: float) -> List[Point2D]:
    return [rotate_point(p, angle_deg) for p in points]


def fmt_point3(p: Point3D) -> str:
    return f"({p[0]:.8f} {p[1]:.8f} {p[2]:.8f})"


def point3(p: Point2D, z: float) -> Point3D:
    return p[0], p[1], z


def fmt_curve(
    start: int,
    end: int,
    points: List[Point2D],
    z: float,
    curve_type: str = "spline",
) -> str:
    out = [f"    {curve_type} {start} {end}", "    ("]

    # For spline, OpenFOAM expects interpolation points between start/end vertices.
    if curve_type == "spline" and len(points) >= 2:
        pts = points[1:-1]
    else:
        pts = points

    for x, y in pts:
        out.append(f"        ({x:.8f} {y:.8f} {z:.8f})")
    out.append("    )")
    return "\n".join(out)


def build_cgrid_blockmesh_dict(
    naca: NACA4Params,
    aoa_deg: float,
    cfg: CGridBlockMeshConfig,
) -> str:
    upper, lower = generate_naca4_upper_lower(
        params=naca,
        n_points=cfg.n_airfoil_half,
        le_cluster_exp=cfg.le_cluster_exp,
    )

    if cfg.enable_le_cap:
        upper, lower = apply_geometric_le_cap(
            upper=upper,
            lower=lower,
            cap_fraction=cfg.le_cap_fraction,
            cap_power=cfg.le_cap_power,
        )

    # Same convention as rotated STL after correction:
    # positive AoA -> nose up.
    upper = rotate_points(upper, -aoa_deg)
    lower = rotate_points(lower, -aoa_deg)

    te = rotate_point((naca.chord, 0.0), -aoa_deg)

    n_pts = min(len(upper), len(lower))
    mid_idx = n_pts // 2

    # Choose cap extent by physical x-location, not only by point index.
    # This keeps the topological cap very small even with heavy LE clustering.
    x_cap_target = naca.chord * max(0.004, min(cfg.le_topology_fraction, 0.06))
    cap_idx = 2
    for i in range(2, mid_idx - 1):
        if upper[i][0] >= x_cap_target and lower[i][0] >= x_cap_target:
            cap_idx = i
            break

    lower_cap = lower[cap_idx]
    upper_cap = upper[cap_idx]

    lower_mid = lower[mid_idx]
    upper_mid = upper[mid_idx]

    # Small wake-cut point behind TE.
    # This follows the topology idea from the reference blockMesh.
    te_cut = (te[0] + cfg.wake_cut_length, te[1])

    zf = cfg.z_half
    zb = -cfg.z_half

    # Keep upstream split on inlet plane to avoid skewed/triangular inlet patch.
    x_mid = cfg.x_min
    x_te = te[0]

    # Topological LE cap indexing:
    # 0: lower cap point (front), 24: upper cap point (front)
    # 4/25: corresponding back points
    # 26-29: inlet points aligned with lower/upper cap y-levels
    all_front_back: List[Point3D] = [
        point3(lower_cap, zf),          # 0
        point3(lower_mid, zf),          # 1
        point3(te_cut, zf),             # 2
        point3(upper_mid, zf),          # 3

        point3(lower_cap, zb),          # 4
        point3(lower_mid, zb),          # 5
        point3(te_cut, zb),             # 6
        point3(upper_mid, zb),          # 7

        point3((x_mid, cfg.y_min), zf), # 8
        point3((cfg.x_max, cfg.y_min), zf), # 9
        point3((cfg.x_max, cfg.y_max), zf), # 10
        point3((x_mid, cfg.y_max), zf), # 11

        point3((x_mid, cfg.y_min), zb), # 12
        point3((cfg.x_max, cfg.y_min), zb), # 13
        point3((cfg.x_max, cfg.y_max), zb), # 14
        point3((x_mid, cfg.y_max), zb), # 15

        point3((cfg.x_min, 0.0), zf),   # 16
        point3((cfg.x_max, 0.0), zf),   # 17

        point3((cfg.x_min, 0.0), zb),   # 18
        point3((cfg.x_max, 0.0), zb),   # 19

        point3((x_te, cfg.y_max), zf),  # 20
        point3((x_te, cfg.y_min), zf),  # 21
        point3((x_te, cfg.y_max), zb),  # 22
        point3((x_te, cfg.y_min), zb),  # 23

        point3(upper_cap, zf),          # 24
        point3(upper_cap, zb),          # 25

        point3((x_mid, lower_cap[1]), zf),  # 26
        point3((x_mid, upper_cap[1]), zf),  # 27
        point3((x_mid, lower_cap[1]), zb),  # 28
        point3((x_mid, upper_cap[1]), zb),  # 29
    ]

    lower_cap_mid = lower[cap_idx: mid_idx + 1]
    lower_mid_te = lower[mid_idx:] + [te_cut]
    upper_cap_mid = upper[cap_idx: mid_idx + 1]
    upper_mid_te = upper[mid_idx:] + [te_cut]

    # Nose curve from lower cap -> LE -> upper cap.
    le_nose_curve = list(reversed(lower[: cap_idx + 1])) + upper[1: cap_idx + 1]

    text = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}

scale 1.0;

vertices
(
"""

    for p in all_front_back:
        text += f"    {fmt_point3(p)}\n"

    text += f""");

blocks
(
    // upper upstream block
    hex (24 27 11 3 25 29 15 7)
        ({cfg.n_streamwise_near} {cfg.n_wall_normal} {cfg.n_z})
        simpleGrading ({cfg.grading_to_wall} {cfg.grading_le_tangent} 1)

    // topological LE cap block
    hex (0 26 27 24 4 28 29 25)
        ({cfg.n_streamwise_near} {cfg.n_le_cap_normal} {cfg.n_z})
        simpleGrading ({cfg.grading_to_wall} 1 1)

    // lower upstream block
    hex (0 1 8 26 4 5 12 28)
        ({cfg.n_wall_normal} {cfg.n_streamwise_near} {cfg.n_z})
        simpleGrading ({cfg.grading_le_tangent} {cfg.grading_to_wall} 1)

    // upper airfoil-to-TE block
    hex (3 11 20 2 7 15 22 6)
        ({cfg.n_streamwise_near} {cfg.n_wall_normal} {cfg.n_z})
        simpleGrading ({cfg.grading_to_wall} 1 1)

    // lower airfoil-to-TE block
    hex (1 2 21 8 5 6 23 12)
        ({cfg.n_wall_normal} {cfg.n_streamwise_near} {cfg.n_z})
        simpleGrading (1 {cfg.grading_to_wall} 1)

    // upper wake block
    hex (2 20 10 17 6 22 14 19)
        ({cfg.n_streamwise_near} {cfg.n_wake_x} {cfg.n_z})
        simpleGrading ({cfg.grading_to_wall} {cfg.grading_wake_x} 1)

    // lower wake block
    hex (2 17 9 21 6 19 13 23)
        ({cfg.n_wake_x} {cfg.n_streamwise_near} {cfg.n_z})
        simpleGrading ({cfg.grading_wake_x} {cfg.grading_to_wall} 1)
);

edges
(
"""

    text += fmt_curve(0, 1, lower_cap_mid, zf, curve_type="spline") + "\n\n"
    text += fmt_curve(1, 2, lower_mid_te, zf, curve_type="spline") + "\n\n"
    text += fmt_curve(24, 3, upper_cap_mid, zf, curve_type="spline") + "\n\n"
    text += fmt_curve(3, 2, upper_mid_te, zf, curve_type="spline") + "\n\n"
    text += fmt_curve(0, 24, le_nose_curve, zf, curve_type="spline") + "\n\n"

    text += fmt_curve(4, 5, lower_cap_mid, zb, curve_type="spline") + "\n\n"
    text += fmt_curve(5, 6, lower_mid_te, zb, curve_type="spline") + "\n\n"
    text += fmt_curve(25, 7, upper_cap_mid, zb, curve_type="spline") + "\n\n"
    text += fmt_curve(7, 6, upper_mid_te, zb, curve_type="spline") + "\n"
    text += fmt_curve(4, 25, le_nose_curve, zb, curve_type="spline") + "\n"

    text += f""");

boundary
(

    inlet
    {{
        type patch;
        faces
        (
            (27 11 15 29)
            (26 27 29 28)
            (8 26 28 12)
        );
    }}

    outlet
    {{
        type patch;
        faces
        (
            (10 17 19 14)
            (17 9 13 19)
        );
    }}

    frontAndBack
    {{
        type empty;
        faces
        (
            (24 3 11 27)
            (0 24 27 26)
            (0 26 8 1)
            (1 8 21 2)
            (3 2 20 11)
            (2 17 10 20)
            (2 21 9 17)

            (25 7 15 29)
            (4 25 29 28)
            (4 28 12 5)
            (7 6 22 15)
            (5 12 23 6)
            (6 19 14 22)
            (6 23 13 19)
        );
    }}

    airfoil
    {{
        type wall;
        faces
        (
            (24 3 7 25)
            (0 24 25 4)
            (0 4 5 1)
            (3 2 6 7)
            (1 2 6 5)
        );
    }}

    farfield
    {{
        type patch;
        faces
        (
            (20 10 14 22)
            (21 23 13 9)

            (11 20 22 15)
            (8 21 23 12)
        );
    }}
);

mergePatchPairs
(
);
"""

    return text


def write_blockmesh_dict(
    output_path: Path,
    naca: NACA4Params,
    aoa_deg: float,
    cfg: CGridBlockMeshConfig | None = None,
) -> None:
    if cfg is None:
        cfg = CGridBlockMeshConfig()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = build_cgrid_blockmesh_dict(naca=naca, aoa_deg=aoa_deg, cfg=cfg)
    output_path.write_text(text, encoding="utf-8")