import os
from pathlib import Path
import numpy as np
import concurrent.futures
from types import SimpleNamespace
from skimage.measure import label, regionprops, find_contours

# Assuming you have a function similar to this:
def bw2boundaries(bw):
    """
    Extracts the boundaries and area of the largest connected component in a binary image.
    
    Parameters:
        bw (numpy.ndarray): A binary image (2D array)
        
    Returns:
        boundaries (list): List of numpy arrays containing contour coordinates.
        area (int): Area (pixel count) of the largest connected component.
    """
    # Label connected components (8-connectivity)
    labeled = label(bw, connectivity=2)
    regions = regionprops(labeled)
    
    # If no components are detected, return empty results.
    if not regions:
        return [], 0
    
   
    # Identify the largest component by area.
    largest_region = max(regions, key=lambda r: r.area)
    largest_label = largest_region.label
    
    # Create a mask for the largest component.
    filtered_mask = (labeled == largest_label)
    
    # Compute area.
    area = int(np.sum(filtered_mask))
    
    # Extract boundaries using the threshold level of 0.5.
    boundaries = find_contours(filtered_mask.astype(np.uint8), level=0.5)


    return boundaries, area

def process_frame(frame):
    """
    Process a single frame, computing boundaries and area, and return a structure.
    
    Parameters:
        frame (numpy.ndarray): A binary image (2D array)
    
    Returns:
        SimpleNamespace: Contains fields 'boundaries' (list) and 'area' (int).
    """
    boundaries, area = bw2boundaries(frame)
    return SimpleNamespace(boundaries=boundaries, area=area)

def compute_boundaries_parallel(frames):
    """
    Processes a list of frames in parallel and returns a list of structures containing the results.
    
    Parameters:
        frames (list or iterable): A list of binary image frames.
    
    Returns:
        list: A list of SimpleNamespace objects for each frame.
              Each object has 'boundaries' and 'area' attributes.
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_frame, frames))
    return results
'''
# Example usage:
if __name__ == "__main__":
    # For demonstration, create an artificial binary video (e.g., 5 frames).
    num_frames = 100
    height, width = 100, 100
    bw_video = np.zeros((num_frames, height, width), dtype=np.uint8)
    for i in range(num_frames):
        # Create a shifting square in each frame
        start = 10 + i * 20
        end = 60 + i * 20
        bw_video[i, start:end, start:end] = 1

    # Convert to a list of frames.
    frames = [bw_video[i] for i in range(num_frames)]
    
    # Process all frames in parallel.
    results = compute_boundaries_parallel(frames)
    
    # Now you can access the results by attribute name.
    for idx, res in enumerate(results):
        print(f"Frame {idx}: Area = {res.area}, Number of boundaries = {len(res.boundaries)}")
        # For example, plot the first boundary if it exists:
        if res.boundaries:
            import matplotlib.pyplot as plt
            contour = res.boundaries[0]
            plt.figure()
            plt.imshow(frames[idx], cmap='gray')
            plt.plot(contour[:, 1], contour[:, 0], '-r', linewidth=2)
            plt.title(f"Frame {idx} Boundary")
            plt.show()
'''
