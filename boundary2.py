import numpy as np
from skimage.measure import label, regionprops, find_contours
from types import SimpleNamespace
import concurrent.futures

def bw2boundaries_all(bw):
    """
    Extracts the boundaries and area of every connected component in a binary image.
    
    Parameters:
        bw (numpy.ndarray): A binary image (2D array)
        
    Returns:
        components (list): List of SimpleNamespace, each with
            - boundaries (list of ndarray of contour coords)
            - area (int)
            - label_id (int)
    """
    # 1) Label all components (8-connectivity)
    labeled = label(bw, connectivity=2)
    regions = regionprops(labeled)
    
    components = []
    for region in regions:
        lab = region.label
        mask = (labeled == lab)
        area = int(region.area)
        # find_contours expects 2D array of 0/1 values
        contours = find_contours(mask.astype(np.uint8), level=0.5)
        components.append(SimpleNamespace(
            label_id=lab,
            area=area,
            boundaries=contours
        ))
    
    return components

def process_frame_all(frame):
    """
    Process a single frame, extracting every component's boundaries and area.
    
    Returns a SimpleNamespace containing:
      - components: list of SimpleNamespace(label_id, area, boundaries)
    """
    components = bw2boundaries_all(frame)
    return SimpleNamespace(components=components)

def compute_boundaries_parallel_all(frames):
    """
    Parallel processing of frames, returning a list of SimpleNamespace per frame,
    each containing all components.
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        return list(executor.map(process_frame_all, frames))
