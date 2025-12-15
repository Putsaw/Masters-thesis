from shapely.geometry import Point, MultiPoint, Polygon
from shapely.ops import triangulate
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

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
    clustering = DBSCAN(eps=cluster_distance, min_samples=5).fit(pts)
    labels = clustering.labels_

    canvas = np.zeros_like(mask)

    # For each cluster, compute concave hull and fill
    for label in set(labels):
        if label == -1:  # noise
            continue

        cluster_pts = pts[labels == label]

        if len(cluster_pts) >= 3:
            hull_pts = alpha_shape(cluster_pts, alpha)
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

