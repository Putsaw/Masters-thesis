from clustering import *
from functions import *
from functions_videos import *
from functions_optical_flow import *
from boundary2 import *

import opticalFlow as of
import videoProcessingFunctions as vpf
from std_functions3 import *
import tkinter as tk
from tkinter import filedialog

from skimage.segmentation import chan_vese

# Pipeline:
    # Compute optical flow between frames
    # Compute magnitude + direction maps
    # Cluster or threshold flow difference
    # Generate a mask
    # Clean mask with morphology

# IDEAS:
    # use confidence mapping to combine optical flow and thresholding
    # instead of combining binary masks use confidence values from otsu thresholding and optical flow to create a weighted mask

# TODO:
    # combine optical flow with otsu thresholding for better results

# Hide the main tkinter window
root = tk.Tk()
root.withdraw()

################################
# Main function
###############################
all_files = filedialog.askopenfilenames(title="Select one or more files")
for file in all_files:
    print("Processing:", file)
    video = load_cine_video(file)  # Ensure load_cine_video is defined or imported

    # play_video_cv2(video)

    nframes, height, width = video.shape[:3]
    dtype = video[0].dtype
    
    for i in range(nframes):
        frame = video[i]
        if np.issubdtype(dtype, np.integer):
            # For integer types (e.g., uint16): scale down from 16-bit to 8-bit.
            frame_uint8 = (frame / 16).astype(np.uint8)
        elif np.issubdtype(dtype, np.floating):
            # For float types (e.g., float32): assume values in [0,1] and scale up to 8-bit.
            frame_uint8 = np.clip((frame * 255), 0, 255).astype(np.uint8)
            # Boolean case: map False→0, True→255
        elif np.issubdtype(dtype, np.bool_):
            # logger.debug("Frame %d: boolean dtype; converting to uint8", i)
            # Convert bool→uint8 and apply gain, then clip
            frame_uint8 = np.clip(frame.astype(np.uint8) * 255, 0, 255).astype(np.uint8)
        else:
            # Fallback for any other type
            frame_uint8 = (frame / 16).astype(np.uint8)

        video[i] = frame_uint8
    video = video.astype(np.uint8)

    ##############################
    # Video Rotation and Stripping
    ##############################
    rotated_video = vpf.createRotatedVideo(video, 60)
    video_strip = vpf.createVideoStrip(rotated_video)

    ##############################
    # Background Removal Visualization
    ##############################
    firstFrameNumber = vpf.findFirstFrame(video_strip)

    first_frame = video_strip[firstFrameNumber]

    ######################################
    # Simple background Removal Visualization
    ######################################
    # for i in range(nframes):
    #     frame = video_strip[i]
    #     foreground = vpf.removeBackground(frame, first_frame)
    #     cv2.imshow('Foreground', foreground)
    #     cv2.imshow('frame', frame)
    #     if cv2.waitKey(60) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()


    drawing = False
    points = []

    def draw_mask(event, x, y, flags, param):
        global drawing, points, mask

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            points = [(x, y)]

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            points.append((x, y))
            cv2.line(mask, points[-2], points[-1], 255, thickness=2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

            if len(points) > 2:
                contour = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [contour], 255)

            points = []

    frame = video_strip[nframes // 2]

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    cv2.namedWindow("Draw Mask")
    cv2.setMouseCallback("Draw Mask", draw_mask)

    while True:
        # Ensure overlay is 3-channel BGR (frame may be grayscale)
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            overlay = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            overlay = frame.copy()

        # Apply red overlay safely (works even if mask has no 255 pixels)
        mask_bool3 = (mask == 255)[:, :, None]
        overlay = np.where(mask_bool3, np.array([0, 0, 255], dtype=overlay.dtype), overlay)

        cv2.imshow("Draw Mask", overlay)

        key = cv2.waitKey(40) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # reset mask
            mask[:] = 0

    cv2.destroyAllWindows()
    cv2.imwrite("mask.png", mask)


    ##############################
    # Filter Visualization
    ###############################
    vpf.applyCLAHE(video_strip)
    for i in range(nframes):
        frame = video_strip[i]
        cv2.imshow('CLAHE filter', frame)
        key = cv2.waitKey(40) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)
    cv2.destroyAllWindows() 

    # vpf.removeBackgroundSimple(video_strip, first_frame, threshold=10)
    # for i in range(nframes):
    #     frame = video_strip[i]
    #     cv2.imshow('Simple Background Removal', frame)
    #     key = cv2.waitKey(40) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    background_mask = vpf.createBackgroundMask(first_frame, threshold=10) # Chamber walls have an intensity of about 3
    otsu_video = vpf.OtsuThreshold(video_strip, background_mask)
    # vpf.invertVideo(video_strip)
    for i in range(nframes):
        frame = otsu_video[i]
        cv2.imshow('Otsu Threshold', frame)
        key = cv2.waitKey(40) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)
    cv2.destroyAllWindows()

    # vpf.applyLaplacianFilter(video_strip)
    # for i in range(nframes):
    #     frame = video_strip[i]
    #     cv2.imshow('Laplacian filter', frame)
    #     key = cv2.waitKey(40) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)
    # cv2.destroyAllWindows() 

    # vpf.applyDoGfilter(video_strip)
    # for i in range(nframes):
    #     frame = video_strip[i]
    #     cv2.imshow('DoG filter', frame)
    #     key = cv2.waitKey(40) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    # vpf.adaptiveGaussianThreshold(video_strip)
    # for i in range(nframes):
    #     frame = video_strip[i]
    #     cv2.imshow('Adaptive Gaussian Threshold', frame)
    #     key = cv2.waitKey(40) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    # vpf.chanVeseSegmentation(video_strip)
    # for i in range(nframes):
    #     frame = video_strip[i]
    #     cv2.imshow('Chan-Vese Segmentation', frame)
    #     key = cv2.waitKey(40) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    # vpf.temporalMedianFilter(video_strip, firstFrameNumber)
    # for i in range(nframes):
    #     frame = video_strip[i]
    #     cv2.imshow('Temporal Median Filter', frame)
    #     key = cv2.waitKey(40) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    # mask = vpf.adaptive_background_subtraction(video_strip)
    # for i in range(nframes):
    #     frame = mask[i]
    #     cv2.imshow('Temporal Median Filter', frame)
    #     cv2.imshow('Original', video_strip[i])
    #     key = cv2.waitKey(40) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)
    # cv2.destroyAllWindows()



    ##############################
    # Optical Flow Visualization
    ##############################
    # TODO: rewrite as a function
    intensity_values = []       # store average intensities
    first_frame = video_strip[firstFrameNumber]
    prev_frame = first_frame

    cluster_masks = np.zeros((nframes, height, width), dtype=np.uint8)

    for i in range(firstFrameNumber, nframes):
        frame = video_strip[i]

        # --- Compute DeepFlow optical flow ---
        flow = of.opticalFlowFarnebackCalculation(prev_frame, frame) # Farneback 0.3 threshold

        # Compute magnitude (motion strength) and angle (not needed here)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # threshold movement
        mask = (mag > 0.4).astype(np.uint8) * 255

        # clean up
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

        # Use clustering to get signal outlines
        cluster_mask = create_cluster_mask(mask, cluster_distance=50, alpha=40)
        clustered_overlay = overlay_cluster_outline(frame, cluster_mask)

        # Compute mean intensity inside the mask
        mean_intensity = cv2.mean(frame, cluster_mask)
        intensity_values.append(mean_intensity)

        # Display results
        # cv2.imshow('filled mask', filled_mask)
        cv2.imshow('Clustered Overlay', clustered_overlay)
        cv2.imshow("mask", mask)
        cv2.imshow('Original', frame)

        key = cv2.waitKey(40) & 0xFF

        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)

        prev_frame = frame

        cluster_masks[i] = cluster_mask

        print(f"Processed frame {i+1}/{nframes}")

    cv2.destroyAllWindows()

    # GOAL: if both otsu and cluster masks agree, keep the region
    # try to remake with non binary masks?

    


    # --- Combine Otsu masks and cluster masks with adjustable weights ---
    # Change these weights to tune contribution from each method
    w_otsu = 0.2      # weight for Otsu mask (0.0 - 1.0)
    w_cluster = 0.6   # weight for cluster mask (0.0 - 1.0)
    w_freehand = 0.2  # weight for freehand mask drawn by user (0.0 - 1.0)
    threshold = 1   # threshold on weighted sum (0.0 - 1.0) — adjust to control how strict combination is

    # Prepare combined masks array
    combined_masks = np.zeros_like(cluster_masks, dtype=np.uint8)
    otsu_optical = np.zeros_like(cluster_masks, dtype=np.uint8)

    # Load freehand mask created earlier by the user (expects single-channel binary image "mask.png")
    freehand_mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
    if freehand_mask is None:
        print("Warning: 'mask.png' not found — proceeding without freehand mask")
        freehand_mask_f = np.zeros((height, width), dtype=np.float32)
    else:
        # Resize to match frames if necessary, keep nearest neighbour to preserve binary nature
        if freehand_mask.shape != (height, width):
            freehand_mask = cv2.resize(freehand_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        freehand_mask_f = (freehand_mask > 0).astype(np.float32)  # 0.0 or 1.0

    for idx in range(nframes):
        otsu_mask = otsu_video[idx].astype(np.float32) / 255.0
        cluster_mask = cluster_masks[idx].astype(np.float32) / 255.0
        freehand = freehand_mask_f  # same for all frames, shape (height, width)

        # If a mask is empty for this frame, treat it as a full-frame mask
        # (counts as if the whole frame is the mask)
        if np.count_nonzero(otsu_mask) == 0:
            otsu_mask = np.ones_like(otsu_mask, dtype=np.float32)
        if np.count_nonzero(freehand) == 0:
            freehand = np.ones_like(freehand, dtype=np.float32)
        # Uncomment for shadowgraph 
        # if np.count_nonzero(cluster_mask) == 0:  
        #     cluster_mask = np.ones_like(cluster_mask, dtype=np.float32)

        # Normalize weights (avoid division by zero)
        total_w = w_otsu + w_cluster + w_freehand
        if total_w <= 0:
            norm_otsu = norm_cluster = norm_freehand = 1.0/3.0
        else:
            norm_otsu = w_otsu / total_w
            norm_cluster = w_cluster / total_w
            norm_freehand = w_freehand / total_w

        # Weighted sum and binarize
        weighted = (otsu_mask * norm_otsu) + (cluster_mask * norm_cluster) + (freehand * norm_freehand)
        combined = (weighted >= threshold).astype(np.uint8) * 255

        # Optional small cleanup
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        combined = cv2.dilate(combined, np.ones((5,5), np.uint8), iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

        # combined_cluster_mask = create_cluster_mask(combined, cluster_distance=50, alpha=40) # Further clustering on combined mask, may need tuning, or skip

        otsu_optical[idx] = combined 
        # combined_masks[idx] = combined_cluster_mask



    print(f"Combined masks computed with w_otsu={w_otsu}, w_cluster={w_cluster}, w_freehand={w_freehand}, threshold={threshold}")

    # Show combined masks (press 'q' to quit, 'p' to pause)
    for i in range(firstFrameNumber, nframes):
        frame = video_strip[i]
        # combined = combined_masks[i]
        otsu_optical_mask = otsu_optical[i]

        clustered_overlay = overlay_cluster_outline(frame, otsu_optical_mask) #may not need clustering

        cv2.imshow('Otsu + Optical flow', otsu_optical_mask)
        cv2.imshow('Clustered Overlay on Combined Mask', clustered_overlay)
        # cv2.imshow('Combined Mask', combined)
        cv2.imshow('Original', frame)

        key = cv2.waitKey(200) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)

    cv2.destroyAllWindows()


    # --- Analyze intensity values ---
    # Needs more work, maybe diffent method to find significant changes

    # calculate derivative of intensity values
    intensity_values = np.array(intensity_values)   [:,0]  # Extract first channel if mean returns a tuple
    
    # Apply rolling mean with window size 5
    window_size = 5
    intensity_smoothed = np.convolve(intensity_values, np.ones(window_size)/window_size, mode='valid')
    
    # Compute derivative on smoothed data
    intensity_derivative = np.diff(intensity_smoothed, prepend=intensity_smoothed[0])

    # --- Create shifted x-axis ---
    # Adjust frame_numbers to match the length after rolling mean
    frame_numbers = np.arange(firstFrameNumber, firstFrameNumber + len(intensity_derivative))

    # only consider frames at least 10 after firstFrameNumber
    start_offset = 10
    if len(intensity_derivative) <= start_offset:
        # not enough frames — fall back to full range
        min_idx = int(np.argmin(intensity_derivative))
    else:
        sliced = intensity_derivative[start_offset:]
        rel_min = int(np.argmin(sliced))
        min_idx = rel_min + start_offset

    min_frame = int(frame_numbers[min_idx])
    min_value = float(intensity_derivative[min_idx])
    print(f"Lowest intensity derivative at frame {min_frame} (index {min_idx}) = {min_value:.6f}")

    plt.plot(frame_numbers, intensity_derivative)
    plt.xlabel("Frame Number")
    plt.ylabel("Mean Intensity Inside Region")
    plt.title("Intensity Over Time (Shifted)")
    plt.show()



                                    
            
# import time
# if __name__ == '__main__':
#     from multiprocessing import freeze_support
#     freeze_support()  # Optional: Needed if freezing to an executable

#     start_time = time.time()
#     main()
#     elapsed_time = time.time() - start_time

#     print(f"Sequential main() finished in {elapsed_time:.2f} seconds.")
