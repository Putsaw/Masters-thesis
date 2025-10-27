# functions.py
from packages import *



def medfilt(image, M, N):
    return ndi_median_filter(image, size=(M, N), mode='constant', cval=0)

def median_filter_video(video_array, M, N, max_workers=None):
    num_frames = video_array.shape[0]
    filtered_frames = [None] * num_frames
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(medfilt, video_array[i], M, N): i 
                           for i in range(num_frames)}
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                filtered_frames[idx] = future.result()
            except Exception as exc:
                print(f"Frame {idx} generated an exception during median filtering: {exc}")
    return np.array(filtered_frames)

def gaussian_low_pass_filter(img, cutoff):
    rows, cols = img.shape
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    u = u - cols // 2
    v = v - rows // 2
    H = np.exp(-(u**2 + v**2) / (2 * cutoff**2))
    img_fft = np.fft.fft2(img.astype(np.float64))
    img_fft_shifted = np.fft.fftshift(img_fft)
    filtered_fft_shifted = img_fft_shifted * H
    filtered_fft = np.fft.ifftshift(filtered_fft_shifted)
    filtered_img = np.fft.ifft2(filtered_fft)
    return np.abs(filtered_img)

def Gaussian_LP_video(video_array, cutoff, max_workers=None):
    num_frames = video_array.shape[0]
    filtered_frames = [None] * num_frames
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(gaussian_low_pass_filter, video_array[i], cutoff): i 
                           for i in range(num_frames)}
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                filtered_frames[idx] = future.result()
            except Exception as exc:
                print(f"Frame {idx} generated an exception during Gaussian LP filtering: {exc}")
    return np.array(filtered_frames)


# -----------------------------
# Time-Distance Map and Area Calculation
# -----------------------------
def calculate_TD_map(horizontal_video: np.ndarray):
    num_frames, height, width = horizontal_video.shape
    time_distance_map = np.zeros((width, num_frames), dtype=np.float32)
    for n in range(num_frames):
        time_distance_map[:, n] = np.sum(horizontal_video[n], axis=0)
    return time_distance_map

def calculate_bw_area(BW: np.ndarray):
    num_frames, height, width = BW.shape
    area = np.zeros(num_frames, dtype=np.float32)
    for n in range(num_frames):
        area[n] = np.sum(BW[n] == 255)
    return area


def apply_morph_open(intermediate_frame, disk_size):
    selem = disk(disk_size)
    opened = binary_opening(intermediate_frame, selem)
    return opened

def apply_hole_filling(opened_frame):
    filled = binary_fill_holes(opened_frame)
    processed_frame = (filled * 255).astype(np.uint8)
    frame_area = np.sum(filled)
    return processed_frame, frame_area



def apply_morph_open_video(intermediate_video: np.ndarray, disk_size: int) -> np.ndarray:
    """
    Apply morphological opening to each frame in the video in parallel.
    
    Parameters:
        intermediate_video (np.ndarray): Video after thresholding (frames, height, width).
        disk_size (int): Radius for the disk-shaped structuring element.
    
    Returns:
        np.ndarray: Video after morphological opening.
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(
            lambda frame: apply_morph_open(frame, disk_size),
            intermediate_video
        ))
    return np.array(results)

def apply_hole_filling_video(opened_video: np.ndarray):
    """
    Apply hole filling and compute the white pixel area for each frame in the video in parallel.
    
    Parameters:
        opened_video (np.ndarray): Video after morphological opening (frames, height, width).
    
    Returns:
        processed_video (np.ndarray): Video after hole filling, with values 0 or 255.
        area (np.ndarray): Array of white pixel counts per frame.
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(
            lambda frame: apply_hole_filling(frame),
            opened_video
        ))
    num_frames = opened_video.shape[0]
    processed_video = np.zeros_like(opened_video, dtype=np.uint8)
    area = np.zeros(num_frames, dtype=np.float32)
    for i, (processed_frame, frame_area) in enumerate(results):
        processed_video[i] = processed_frame
        area[i] = frame_area
    return processed_video, area



