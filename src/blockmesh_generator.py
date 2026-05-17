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

    n_streamwise_near: int = 100
    n_wall_normal: int = 80
    n_wake_x: int = 120

    n_z: int = 1

    grading_to_wall: float = 800.0
    grading_from_wall: float = 0.00125
    grading_wake_x: float = 8.0

    # Small artificial wake cut behind TE for block topology.
    # Keep small for first tests.
    wake_cut_length: float = 0.03


def cosine_spacing(n: int) -> List[float]:
    return [
        0.5 * (1.0 - math.cos(math.pi * i / (n - 1)))
        for i in range(n)
    ]


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
) -> tuple[List[Point2D], List[Point2D]]:
    xs = cosine_spacing(n_points)

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


def fmt_polyline(start: int, end: int, points: List[Point2D], z: float) -> str:
    out = [f"    polyLine {start} {end}", "    ("]
    for x, y in points:
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
    )

    # Same convention as rotated STL after correction:
    # positive AoA -> nose up.
    upper = rotate_points(upper, -aoa_deg)
    lower = rotate_points(lower, -aoa_deg)

    le = rotate_point((0.0, 0.0), -aoa_deg)
    te = rotate_point((naca.chord, 0.0), -aoa_deg)

    lower_mid = lower[len(lower) // 2]
    upper_mid = upper[len(upper) // 2]

    # Small wake-cut point behind TE.
    # This follows the topology idea from the reference blockMesh.
    te_cut = (te[0] + cfg.wake_cut_length, te[1])

    zf = cfg.z_half
    zb = -cfg.z_half

    # Keep inlet as a flat plane at x = x_min.
    x_mid = cfg.x_min
    x_te = te[0]
    y_te_cut = te_cut[1]

    # Vertex numbering intentionally follows the reference topology style.
    vertices_2d: List[Point2D] = [
        le,                          # 0
        lower_mid,                   # 1
        te_cut,                      # 2
        upper_mid,                   # 3

        le,                          # 4 back copy starts later, placeholder not used here
    ]

    # Explicit front vertices.
    front_2d: List[Point2D] = [
        le,                              # 0
        lower_mid,                       # 1
        te_cut,                          # 2
        upper_mid,                       # 3

        # back-side airfoil vertices are added separately as 4-7 in original topology
    ]

    # We need the exact reference numbering:
    # 0-3 front airfoil control
    # 4-7 back airfoil control
    # 8-15 farfield lower/upper rectangles
    # 16-19 inlet/outlet midline
    # 20-23 TE vertical farfield line
    # 24-29 duplicate wake interface vertices
    all_front_back: List[Point3D] = [
        point3(le, zf),                 # 0
        point3(lower_mid, zf),          # 1
        point3(te_cut, zf),             # 2
        point3(upper_mid, zf),          # 3

        point3(le, zb),                 # 4
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

    ]

    lower_le_mid = lower[: len(lower) // 2 + 1]
    lower_mid_te = lower[len(lower) // 2 :] + [te_cut]

    upper_le_mid = upper[: len(upper) // 2 + 1]
    upper_mid_te = upper[len(upper) // 2 :] + [te_cut]

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
    // upper upstream / leading-edge block
    hex (0 16 11 3 4 18 15 7)
        ({cfg.n_streamwise_near} {cfg.n_wall_normal} {cfg.n_z})
        simpleGrading ({cfg.grading_to_wall} {cfg.grading_from_wall} 1)

    // lower upstream / leading-edge block
    hex (0 1 8 16 4 5 12 18)
        ({cfg.n_wall_normal} {cfg.n_streamwise_near} {cfg.n_z})
        simpleGrading ({cfg.grading_from_wall} {cfg.grading_to_wall} 1)

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

    text += fmt_polyline(0, 1, lower_le_mid, zf) + "\n\n"
    text += fmt_polyline(1, 2, lower_mid_te, zf) + "\n\n"
    text += fmt_polyline(0, 3, upper_le_mid, zf) + "\n\n"
    text += fmt_polyline(3, 2, upper_mid_te, zf) + "\n\n"

    text += fmt_polyline(4, 5, lower_le_mid, zb) + "\n\n"
    text += fmt_polyline(5, 6, lower_mid_te, zb) + "\n\n"
    text += fmt_polyline(4, 7, upper_le_mid, zb) + "\n\n"
    text += fmt_polyline(7, 6, upper_mid_te, zb) + "\n"

    text += f""");

boundary
(

    inlet
    {{
        type patch;
        faces
        (
            (16 11 15 18)
            (16 18 12 8)
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
            (0 3 11 16)
            (0 16 8 1)
            (1 8 21 2)
            (3 2 20 11)
            (2 17 10 20)
            (2 21 9 17)

            (4 18 15 7)
            (4 5 12 18)
            (7 15 22 6)
            (5 6 23 12)
            (6 22 14 19)
            (6 19 13 23)
        );
    }}

    airfoil
    {{
        type wall;
        faces
        (
            (0 3 7 4)
            (0 4 5 1)
            (3 2 6 7)
            (1 5 6 2)
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