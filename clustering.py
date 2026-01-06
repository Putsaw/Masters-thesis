from shapely.geometry import Point, MultiPoint, Polygon
from shapely.ops import triangulate
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay, ConvexHull

def get_polygon_outline(geom):
    """Return an exterior coordinate list from any Shapely geometry type."""
    if geom.is_empty:
        return None

    # If already a Polygon
    if isinstance(geom, Polygon):
        return geom.exterior.coords

    # If MultiPolygon, take largest polygon
    if isinstance(geom, MultiPolygon):
        largest = max(geom.geoms, key=lambda g: g.area)
        return largest.exterior.coords

    # If it's a line or point, no valid outline
    return None


def alpha_shape(points, alpha):

    if len(points) < 4:
        return points

    pts = MultiPoint(points)
    triangles = triangulate(pts)

    def triangle_circumradius(tri):
        a, b, c = tri.exterior.coords[:3]
        a, b, c = np.array(a), np.array(b), np.array(c)
        side = [np.linalg.norm(a-b),
                np.linalg.norm(b-c),
                np.linalg.norm(c-a)]
        s = sum(side) / 2
        area = max(s*(s-side[0])*(s-side[1])*(s-side[2]), 1e-12)**0.5
        return (side[0]*side[1]*side[2]) / (4.0 * area)

    # keep triangles with small circumradius
    alpha_tris = [tri for tri in triangles if triangle_circumradius(tri) < alpha]

    if not alpha_tris:
        # fallback to convex hull
        hull = pts.convex_hull
        outline = get_polygon_outline(hull)
        return np.array(outline).astype(int) if outline else points

    # combine triangles
    shape = alpha_tris[0]
    for tri in alpha_tris[1:]:
        shape = shape.union(tri)

    # SAFE extraction of exterior points
    outline = get_polygon_outline(shape)
    if outline is None:
        # fallback: convex hull
        ch = pts.convex_hull
        outline = get_polygon_outline(ch)

    return np.array(outline).astype(int)

def QhullError(Exception):
    pass


def fast_alpha_shape(points, alpha, max_points=2000):
    """Faster alpha-shape using Delaunay with safety checks.

    - Downsamples if there are too many points (keeps convex-hull points).
    - Catches Qhull errors and falls back to convex hull.
    - Adds a guard on the loop extraction to avoid infinite walks.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 4:
        return pts.astype(int)

    pts = np.unique(pts, axis=0)  # remove duplicates
    if len(pts) < 4:
        return pts.astype(int)

    # If too many points, sample while keeping hull vertices so the outline is preserved
    if len(pts) > max_points:
        try:
            hull = ConvexHull(pts)
            hull_idx = set(hull.vertices.tolist())
        except QhullError:
            hull_idx = set()

        keep_idx = list(hull_idx)
        remaining = [i for i in range(len(pts)) if i not in hull_idx]
        n_needed = max_points - len(keep_idx)
        if n_needed > 0 and remaining:
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(remaining, size=min(n_needed, len(remaining)), replace=False)
            keep_idx += sample_idx.tolist()
        pts = pts[keep_idx]

    # Delaunay may fail on degenerate sets - catch and fallback
    try:
        tri = Delaunay(pts)
        simplices = tri.simplices  # (m,3)
    except QhullError:
        try:
            hull = ConvexHull(pts)
            return pts[hull.vertices].astype(int)
        except QhullError:
            # give up, return points (will be handled by caller)
            return pts.astype(int)

    pa, pb, pc = pts[simplices[:, 0]], pts[simplices[:, 1]], pts[simplices[:, 2]]
    a = np.linalg.norm(pb - pc, axis=1)
    b = np.linalg.norm(pc - pa, axis=1)
    c = np.linalg.norm(pa - pb, axis=1)

    s = 0.5 * (a + b + c)
    area = np.maximum(s * (s - a) * (s - b) * (s - c), 1e-12) ** 0.5
    circum_r = (a * b * c) / (4.0 * area)

    good = simplices[circum_r < alpha]
    if good.size == 0:
        hull = ConvexHull(pts)
        return pts[hull.vertices].astype(int)

    # edges from good triangles
    edges = np.vstack([good[:, [0, 1]], good[:, [1, 2]], good[:, [2, 0]]])
    edges = np.sort(edges, axis=1)  # canonical
    uniq_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = uniq_edges[counts == 1]

    if boundary.size == 0:
        hull = ConvexHull(pts)
        return pts[hull.vertices].astype(int)

    # build adjacency
    adj = {}
    for u, v in boundary:
        adj.setdefault(int(u), []).append(int(v))
        adj.setdefault(int(v), []).append(int(u))

    # extract loops with a safety guard to avoid infinite loops
    loops = []
    visited = set()
    max_steps_per_loop = max(1000, len(adj) * 3)
    for start in list(adj.keys()):
        if start in visited:
            continue
        cur = start
        prev = None
        loop = [cur]
        steps = 0
        while True:
            steps += 1
            if steps > max_steps_per_loop:
                # abort this loop - likely malformed adjacency
                break
            neighs = [n for n in adj.get(cur, []) if n != prev]
            if not neighs:
                break
            nxt = neighs[0]
            if nxt == start:
                # closed loop
                loop.append(nxt)
                visited.update(loop)
                break
            if nxt in visited:
                # encountered previously visited node, abort
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        if len(loop) > 2:
            # ensure we don't return a trailing duplicate index
            if loop[0] == loop[-1]:
                loop = loop[:-1]
            loops.append(pts[loop])

    if not loops:
        # fallback
        hull = ConvexHull(pts)
        return pts[hull.vertices].astype(int)

    # pick largest loop by polygon area
    best = max(loops, key=lambda arr: Polygon(arr).area)
    return np.asarray(best).astype(int)


def create_cluster_mask(mask, cluster_distance=30, alpha=30):
    """
    Create a filled mask from clustered contours.
    Returns a binary mask with filled clusters.
    """
    # Get contour points from mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for cnt in contours:
        for p in cnt:
            points.append(p[0])  # (x, y)

    if len(points) == 0:
        return np.zeros_like(mask)

    pts = np.array(points)

    # Cluster nearby points
    clustering = DBSCAN(eps=cluster_distance, min_samples=10).fit(pts) # min_samples = min points to form a cluster
    labels = clustering.labels_

    canvas = np.zeros_like(mask)

    # For each cluster, compute concave hull and fill
    for label in set(labels):
        if label == -1:  # exclude noise
            continue

        cluster_pts = pts[labels == label]

        if len(cluster_pts) >= 5: # minimum points to run concave hull filling
            hull_pts = fast_alpha_shape(cluster_pts, alpha)
            cv2.fillPoly(canvas, [hull_pts], 255)

    return canvas


def overlay_cluster_outline(frame, cluster_mask):
    """
    Take a filled cluster mask and overlay its outline on the frame.
    Returns the frame with cluster outlines drawn.
    """
    # Get contours from the filled mask
    contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    canvas = np.zeros_like(cluster_mask)
    
    # Draw outlines only
    for cnt in contours:
        cv2.polylines(canvas, [cnt], isClosed=True, color=255, thickness=2)
    
    overlay = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)
    
    return overlay

def convex_hull_mask(mask):
    """
    Create a filled mask from the convex hull of the contours in the input mask.
    Returns a binary mask with the convex hull filled.
    """
    # Get contour points from mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for cnt in contours:
        for p in cnt:
            points.append(p[0])  # (x, y)

    if len(points) == 0:
        return np.zeros_like(mask)

    pts = np.array(points)

    # Compute convex hull
    try:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
    except QhullError:
        return np.zeros_like(mask)
    
    canvas = np.zeros_like(mask)
    cv2.fillPoly(canvas, [hull_pts], 255)
    
    return canvas


