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

    n_streamwise_near: int = 100
    n_wall_normal: int = 80
    n_wake_x: int = 120

    n_z: int = 1

    grading_to_wall: float = 800.0
    grading_le_tangent: float = 0.45
    grading_wake_x: float = 8.0

    # Far downstream extension - new blocks added after x_max.
    # Cell sizes start where the near-wake ends and expand gently further
    # downstream without affecting the mesh around the airfoil.
    x_far: float = 20.0
    n_far_wake_x: int = 80
    grading_far_wake_x: float = 4.0

    # Small artificial wake cut behind TE for the C-grid block topology.
    wake_cut_length: float = 0.03

    # Geometric leading-edge cap smoothing (keeps topology unchanged).
    # This primarily targets the first few cells near LE.
    enable_le_cap: bool = True
    le_cap_fraction: float = 0.03
    le_cap_power: float = 1.15

    # Topological LE cap (removes the singular LE vertex in block topology).
    le_topology_fraction: float = 0.028
    n_le_cap_normal: int = 24

    # Chord-wise location where the near-TE sub-curve starts.
    # Using physical x-location avoids index-based artifacts when LE clustering
    # is strong (index midpoint may sit too close to LE).
    te_transition_fraction: float = 0.82


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


def find_first_x_aligned_index(
    upper: List[Point2D],
    lower: List[Point2D],
    x_target: float,
    start_idx: int,
    stop_idx: int,
    default_idx: int,
) -> int:
    for i in range(start_idx, stop_idx):
        if upper[i][0] >= x_target and lower[i][0] >= x_target:
            return i
    return default_idx


def fmt_point3(p: Point3D) -> str:
    return f"({p[0]:.8f} {p[1]:.8f} {p[2]:.8f})"


def point3(p: Point2D, z: float) -> Point3D:
    return p[0], p[1], z


def fmt_spline(
    start: int,
    end: int,
    points: List[Point2D],
    z: float,
) -> str:
    out = [f"    spline {start} {end}", "    ("]

    # OpenFOAM expects interpolation points between the start/end vertices.
    pts = points[1:-1] if len(points) >= 2 else points

    for x, y in pts:
        out.append(f"        ({x:.8f} {y:.8f} {z:.8f})")
    out.append("    )")
    return "\n".join(out)


def build_cgrid_blockmesh_dict(
    naca: NACA4Params,
    aoa_deg: float,
    cfg: CGridBlockMeshConfig,
) -> str:
    if cfg.x_far <= cfg.x_max:
        raise ValueError("x_far must be greater than x_max")

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

    # Positive AoA means nose-up airfoil orientation.
    upper = rotate_points(upper, -aoa_deg)
    lower = rotate_points(lower, -aoa_deg)

    te = rotate_point((naca.chord, 0.0), -aoa_deg)

    n_pts = min(len(upper), len(lower))

    x_te_transition = naca.chord * max(0.55, min(cfg.te_transition_fraction, 0.98))
    mid_idx = find_first_x_aligned_index(
        upper=upper,
        lower=lower,
        x_target=x_te_transition,
        start_idx=2,
        stop_idx=n_pts - 2,
        default_idx=n_pts // 2,
    )

    # Choose cap extent by physical x-location, not only by point index.
    # This keeps the topological cap very small even with heavy LE clustering.
    x_cap_target = naca.chord * max(0.004, min(cfg.le_topology_fraction, 0.06))
    cap_idx = find_first_x_aligned_index(
        upper=upper,
        lower=lower,
        x_target=x_cap_target,
        start_idx=2,
        stop_idx=mid_idx - 1,
        default_idx=2,
    )

    lower_cap = lower[cap_idx]
    upper_cap = upper[cap_idx]

    lower_mid = lower[mid_idx]
    upper_mid = upper[mid_idx]

    # Small wake-cut point behind TE used to close the C-grid topology.
    te_cut = (te[0] + cfg.wake_cut_length, te[1])

    zf = cfg.z_half
    zb = -cfg.z_half

    # Inlet-side split plane used by the upstream blocks.
    x_inlet = cfg.x_min
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

        point3((x_inlet, cfg.y_min), zf), # 8
        point3((cfg.x_max, cfg.y_min), zf), # 9
        point3((cfg.x_max, cfg.y_max), zf), # 10
        point3((x_inlet, cfg.y_max), zf), # 11

        point3((x_inlet, cfg.y_min), zb), # 12
        point3((cfg.x_max, cfg.y_min), zb), # 13
        point3((cfg.x_max, cfg.y_max), zb), # 14
        point3((x_inlet, cfg.y_max), zb), # 15

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

        point3((x_inlet, lower_cap[1]), zf),  # 26
        point3((x_inlet, upper_cap[1]), zf),  # 27
        point3((x_inlet, lower_cap[1]), zb),  # 28
        point3((x_inlet, upper_cap[1]), zb),  # 29

        # Far downstream extension vertices at x_far.
        point3((cfg.x_far, 0.0), zf),          # 30
        point3((cfg.x_far, cfg.y_max), zf),    # 31
        point3((cfg.x_far, cfg.y_min), zf),    # 32
        point3((cfg.x_far, 0.0), zb),          # 33
        point3((cfg.x_far, cfg.y_max), zb),    # 34
        point3((cfg.x_far, cfg.y_min), zb),    # 35
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

    // upper far wake block (x_max -> x_far, y=0 -> y_max)
    // j=0 face (17,10,14,19) shared with j=max face of upper wake block
    hex (17 10 31 30  19 14 34 33)
        ({cfg.n_streamwise_near} {cfg.n_far_wake_x} {cfg.n_z})
        simpleGrading ({cfg.grading_to_wall} {cfg.grading_far_wake_x} 1)

    // lower far wake block (x_max -> x_far, y=0 -> y_min)
    // i=0 face (17,9,13,19) shared with i=max face of lower wake block
    hex (17 30 32 9  19 33 35 13)
        ({cfg.n_far_wake_x} {cfg.n_streamwise_near} {cfg.n_z})
        simpleGrading ({cfg.grading_far_wake_x} {cfg.grading_to_wall} 1)
);

edges
(
"""

    edge_specs = [
        (0, 1, lower_cap_mid, zf),
        (1, 2, lower_mid_te, zf),
        (24, 3, upper_cap_mid, zf),
        (3, 2, upper_mid_te, zf),
        (0, 24, le_nose_curve, zf),
        (4, 5, lower_cap_mid, zb),
        (5, 6, lower_mid_te, zb),
        (25, 7, upper_cap_mid, zb),
        (7, 6, upper_mid_te, zb),
        (4, 25, le_nose_curve, zb),
    ]
    text += "\n\n".join(
        fmt_spline(start, end, points, z)
        for start, end, points, z in edge_specs
    )
    text += "\n"

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
            (33 34 31 30)
            (35 33 30 32)
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

            // far wake front/back
            (30 31 10 17)
            (17 9 32 30)
            (19 14 34 33)
            (19 33 35 13)
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

            // far wake top/bottom
            (10 31 34 14)
            (13 35 32 9)
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