import numpy as np
from scipy.ndimage import binary_fill_holes
from concurrent.futures import ProcessPoolExecutor

def _fill_frame(frame_bool):
    from scipy.ndimage import binary_fill_holes
    filled = binary_fill_holes(frame_bool)
    return filled


def fill_video_holes_parallel(bw_video: np.ndarray,
                               n_workers: int = None) -> np.ndarray:
    """
    Fill holes in each frame of a binary video in parallel.
    
    Parameters
    ----------
    bw_video : np.ndarray
        Binary video data of shape (n_frames, height, width), values 0/1 or 0/255.
    n_workers : int, optional
        Number of worker processes to use. Defaults to os.cpu_count() if None.
    
    Returns
    -------
    np.ndarray
        Hole‑filled binary video, same shape and dtype as input.
    """
    # Ensure we have boolean frames
    # Any nonzero becomes True; zero remains False.
    bw_bool_video = (bw_video > 0)
    
    
    # Launch parallel filling
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        # executor.map returns in order; we collect into a list
        filled_frames = list(exe.map(_fill_frame, bw_bool_video))
    
    # Stack back into a (n_frames, H, W) array
    return np.stack(filled_frames, axis=0)

# -------------------------
# Example usage:
# -------------------------
'''if __name__ == "__main__":
    # Suppose bw_flow is your binary video from earlier:
    # bw_flow = mask_video(...); bw_flow = binarize_video_global_threshold(...)

    # Fill all holes in parallel
    bw_flow_filled = fill_video_holes_parallel(bw_flow)

    # Quick sanity check
    print("Before fill: any holes left?", not np.all(bw_flow[0] == binary_fill_holes(bw_flow[0])))
    print("After fill shape:", bw_flow_filled.shape)'''
