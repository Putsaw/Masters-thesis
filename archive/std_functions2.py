import numpy as np
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed

def local_std_integral(frame, std_size):
    """
    Compute the local standard deviation over a (2*std_size+1)x(2*std_size+1) window
    using an integral image approach. No padding is applied; the result is computed
    for the valid region only.
    
    Parameters:
        frame (np.ndarray): 2D image in float32.
        std_size (int): Half-window size.
    
    Returns:
        np.ndarray: Standard deviation image of shape (H - 2*std_size, W - 2*std_size).
    """
    k = 2 * std_size + 1  # full window size
    # Precompute squared image
    I = frame
    I2 = frame * frame
    
    # Compute integral images with an extra row/column of zeros for fast area sums.
    S = np.pad(np.cumsum(np.cumsum(I, axis=0), axis=1), ((1, 0), (1, 0)), mode='constant', constant_values=0)
    S2 = np.pad(np.cumsum(np.cumsum(I2, axis=0), axis=1), ((1, 0), (1, 0)), mode='constant', constant_values=0)
    
    # Determine output shape (valid region only)
    H, W = frame.shape
    out_H = H - k + 1
    out_W = W - k + 1

    # Compute sum over each k x k window using vectorized slicing:
    sum_window = S[k:, k:] - S[:-k, k:] - S[k:, :-k] + S[:-k, :-k]
    sum2_window = S2[k:, k:] - S2[:-k, k:] - S2[k:, :-k] + S2[:-k, :-k]
    
    area = k * k
    mean = sum_window / area
    var = sum2_window / area - mean * mean
    # Numerical precision might lead to tiny negative values; clip to zero.
    var = np.clip(var, 0, None)
    std = np.sqrt(var)
    
    # Free memory from intermediates.
    del S, S2, sum_window, sum2_window, I2
    gc.collect()
    return std

def max_pooling(image, pool_size):
    """
    Perform max pooling on a 2D image using non-overlapping windows of size pool_size.
    
    Parameters:
        image (np.ndarray): 2D input image.
        pool_size (int): Pooling factor.
    
    Returns:
        np.ndarray: Pooled image.
    """
    H, W = image.shape
    new_H = H // pool_size
    new_W = W // pool_size
    # Crop to ensure divisibility
    cropped = image[:new_H * pool_size, :new_W * pool_size]
    pooled = cropped.reshape(new_H, pool_size, new_W, pool_size).max(axis=(1, 3))
    return pooled

def upsample(image, pool_size):
    """
    Upsample a 2D image by repeating each element pool_size times in both dimensions.
    
    Parameters:
        image (np.ndarray): 2D input image.
        pool_size (int): Upsampling factor.
    
    Returns:
        np.ndarray: Upsampled image.
    """
    return np.repeat(np.repeat(image, pool_size, axis=0), pool_size, axis=1)

def process_frame_std_optimized(frame, std_size, pool_size):
    """
    Process one frame:
      1. Convert to float32.
      2. Compute the local standard deviation via an integral image approach.
      3. Apply max pooling (to reduce resolution and memory overhead).
      4. Immediately upsample the pooled result.
    
    The final result is in float32.
    
    Parameters:
        frame (np.ndarray): Input 2D frame (uint16 expected).
        std_size (int): Half-window size for local std computation.
        pool_size (int): Pooling factor.
    
    Returns:
        np.ndarray: Processed frame (float32) of shape matching the valid region of local std.
    """
    # Step 1: Convert to float32.
    frame_f = frame.astype(np.float32)
    
    # Step 2: Compute local standard deviation (valid region only).
    std_image = local_std_integral(frame_f, std_size)
    
    # Step 3: Pipeline max pooling on the std image.
    pooled = max_pooling(std_image, pool_size)
    
    # Step 4: Upsample the pooled result back to the std_image resolution.
    upsampled = upsample(pooled, pool_size)
    # If upsampled image is slightly larger due to integer division, crop it.
    H_valid, W_valid = std_image.shape
    upsampled = upsampled[:H_valid, :W_valid]
    
    # Free intermediate variables.
    del frame_f, std_image, pooled
    gc.collect()
    return upsampled

def stdfilt_video_parallel_optimized(video, std_size, pool_size, max_workers=None):
    """
    Process a video (3D numpy array) in parallel. Each frame is processed using
    an optimized pipeline that computes the local standard deviation using an
    integral image approach (without padding), followed by max pooling and upsampling.
    
    Parameters:
        video (np.ndarray): Input video with shape (num_frames, height, width) in uint16.
        std_size (int): Half-window size for the standard deviation filter.
        pool_size (int): Pooling factor.
        max_workers (int, optional): Number of parallel worker processes.
    
    Returns:
        np.ndarray: Processed video with shape (num_frames, H_valid, W_valid) in float32,
                    where H_valid = height - (2*std_size) and W_valid = width - (2*std_size).
    """
    num_frames = video.shape[0]
    # Determine valid output dimensions for one frame.
    H, W = video.shape[1:3]
    k = 2 * std_size + 1
    H_valid = H - k + 1
    W_valid = W - k + 1

    video_out = np.empty((num_frames, H_valid, W_valid), dtype=np.float32)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_frame_std_optimized, video[i], std_size, pool_size): i
            for i in range(num_frames)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                video_out[idx] = future.result()
            except Exception as exc:
                print(f'Frame {idx} processing generated an exception: {exc}')
    
    return video_out

# Example usage:
# Suppose 'strip' is your input video (shape: num_frames x height x width, uint16).
# std_video = stdfilt_video_parallel_optimized(strip, std_size=3, pool_size=2)
# play_video_cv2(std_video)

