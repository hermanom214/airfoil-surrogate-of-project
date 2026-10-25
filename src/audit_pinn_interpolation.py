"""Read-only comparison of p/U Delaunay support and stored PINN masks."""
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
from src.flow_extractor import read_vtk_with_pyvista, generate_naca4_polygon
from shapely.geometry import Polygon
from scipy.ndimage import binary_dilation

def audit_interpolation(path, meta):
    source=Path("data/flow_fields")/meta["case_id"]
    vtk=sorted((source/"VTK").glob("*.vtk"))[-1]
    points,p,U=read_vtk_with_pyvista(vtk)
    # griddata(linear) uses the same projected source coordinates/Qhull support.
    tri=Delaunay(points[:,:2])
    with np.load(path,allow_pickle=False) as z:
        xy=z["xy"]; fluid=z["fluid_mask"].astype(bool); physics=z["physics_valid_mask"].astype(bool)
        interpolation=z["interpolation_valid_mask"].astype(bool)
    simplex=tri.find_simplex(xy.reshape(-1,2)).reshape(fluid.shape)
    outside=simplex<0
    polygon=generate_naca4_polygon(meta["naca_code"],meta["chord"],meta["aoa_deg"])
    support=binary_dilation(physics,structure=np.ones((5,5),bool))
    used=np.unique(simplex[support]); used=used[used>=0]
    crossing=[]; overlap_fractions=[]
    for index in used:
        triangle=Polygon(tri.points[tri.simplices[index]])
        overlap=triangle.intersection(polygon).area
        if overlap>1e-12:
            crossing.append(index)
            overlap_fractions.append(overlap/triangle.area)
    crosses=np.isin(simplex,crossing)&physics
    crossed=np.isin(simplex,crossing)
    # Exact dependencies: center, axial offsets +/-1,+/-2, mixed +/-1,+/-1.
    affected=crossed.copy()
    for dy,dx in [(0,1),(0,-1),(1,0),(-1,0),(0,2),(0,-2),(2,0),(-2,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
        shifted=np.zeros_like(crossed)
        sy=slice(max(0,dy),min(crossed.shape[0],crossed.shape[0]+dy))
        sx=slice(max(0,dx),min(crossed.shape[1],crossed.shape[1]+dx))
        ty=slice(max(0,-dy),min(crossed.shape[0],crossed.shape[0]-dy))
        tx=slice(max(0,-dx),min(crossed.shape[1],crossed.shape[1]-dx))
        shifted[sy,sx]=crossed[ty,tx]; affected|=shifted
    return {"source_point_count":len(points),"p_U_nearest_filled_fluid_count":int((outside&fluid).sum()),
            "p_U_nearest_filled_physics_count":int((outside&physics).sum()),
            "p_U_nearest_filled_turbulence_valid_count":int((outside&interpolation).sum()),
            "physics_centers_in_triangles_intersecting_solid":int(crosses.sum()),
            "stencil_support_in_triangles_intersecting_solid":int((np.isin(simplex,crossing)&support).sum()),
            "p_U_nearest_filled_stencil_support_count":int((outside&support).sum()),
            "physics_stencils_touching_solid_crossing_triangles":int((affected&physics).sum()),
            "maximum_triangle_solid_overlap_fraction":max(overlap_fractions,default=0.),
            "crossing_triangle_count":len(crossing)}
