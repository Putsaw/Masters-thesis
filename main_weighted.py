from clustering import *
from extrapolation import SprayConeBackfill, extrapolate_cone
from functions import *
from functions_videos import *
from functions_optical_flow import *
from boundary2 import *

import opticalFlow as of
import videoProcessingFunctions as vpf
from std_functions3 import *
from fill import _fill_frame
import tkinter as tk
from tkinter import filedialog
import json
import os


# IDEAS:
    # Have user set nozzle point and cut video automatically after rotation (Take 200 pixels up and down from nozzle point?, set nozzle point as far left point?)


# TODO:
    # combine optical flow with otsu thresholding for better results (done)
    # improve clustering method to avoid holes in masks
        # maybe use morphological operations + fill, to close holes
        # or add some interpolation algorithm to fill gaps in masks over time
    # optimize performance for larger videos (CUDA?)
    # improve GUI for mask drawing and parameter tuning
    # add option to save/load masks (TBD)
    # add some extrapolation algorithm to extend masks before/after detected motion (partially done)
    # Maybe standardize naming for videos to use the best settings automatically (TBD)

# Hide the main tkinter window
root = tk.Tk()
root.withdraw()

# Load saved spray origins
origins_file = 'spray_origins.json'
if os.path.exists(origins_file):
    with open(origins_file, 'r') as f:
        spray_origins = json.load(f)
else:
    spray_origins = {}

################################
# Load Video Files
###############################
all_files = filedialog.askopenfilenames(title="Select one or more files")
for file in all_files:
    print("Processing:", file)
    video = load_cine_video(file)  # Ensure load_cine_video is defined or imported

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
    rotated_video = vpf.createRotatedVideo(video, 60) # Rotate 60 degrees clockwise

    video_strip = vpf.createVideoStrip(rotated_video)

    firstFrameNumber = vpf.findFirstFrame(video_strip)
    first_frame = video_strip[firstFrameNumber]

    # Set spray origin
    if file in spray_origins:
        spray_origin = tuple(spray_origins[file])
        print(f"Reusing spray origin for {file}: {spray_origin}")
    else:
        # UI to select
        class PointHolder:
            def __init__(self):
                self.point = None
        
        holder = PointHolder()
        def select_origin(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                holder.point = (x, y)  # type: ignore
                print(f"Selected spray origin: {holder.point}")
        
        cv2.imshow('Set Spray Origin - Click on the nozzle', video_strip[firstFrameNumber+100]) # Show a frame after firstFrameNumber for context, may need adjustment
        cv2.setMouseCallback('Set Spray Origin - Click on the nozzle', select_origin)
        
        current_frame = firstFrameNumber + 100
        while holder.point is None:
            key = cv2.waitKeyEx(10)
            if key == ord('q'):
                break
            elif key == 2424832:  # left arrow
                current_frame = max(firstFrameNumber, current_frame - 1)
                cv2.imshow('Set Spray Origin - Click on the nozzle', video_strip[current_frame])
            elif key == 2555904:  # right arrow
                current_frame = min(nframes - 1, current_frame + 1)
                cv2.imshow('Set Spray Origin - Click on the nozzle', video_strip[current_frame])
        cv2.destroyWindow('Set Spray Origin - Click on the nozzle')
        
        if holder.point is None:
            spray_origin = (1, height // 2)  # Default
        else:
            spray_origin = holder.point
        
        # Save
        spray_origins[file] = list(spray_origin)
        with open(origins_file, 'w') as f:
            json.dump(spray_origins, f)

    ##############################
    # Freehand Mask Creation
    ##############################
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


    background_mask = vpf.createBackgroundMask(first_frame, threshold=20) # Chamber walls have an intensity of about 3
    # otsu_video = vpf.OtsuThreshold(video_strip, background_mask)
    cv2.imshow("Background Mask", background_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    ##############################
    # Optical Flow Visualization
    ##############################
    use_intensity_only = False  # If True, set w_magnitude=0 and use 80% thresholding instead of 95%

    use_cumulative_as_mask = False  # if True, use cumulative_mask to restrict areas for intensity score
    
    if use_intensity_only:
        print("Using intensity-only mode (no optical flow contribution).")
        mag_array = np.ones_like(video_strip, dtype=np.float32)
    else:
        print("Using combined intensity and optical flow mode.")
        mag_array = of.runOpticalFlowCalculationWeighted(firstFrameNumber, video_strip, method='Farneback')

    
    # GOAL: if both otsu and cluster masks agree, keep the region
    # mag values above 0.4 are considered motion
    # IDEA: cap the mag values to 0.4, anything above is full confidence of motion
    #       also if intensity or mag has high enough confidence separately, keep the region
    #       Find a good method for dynamic thresholding of the combined score
    # NOTE: freehand mask is binary, so areas inside have full confidence, outside have zero confidence
    #       add a system to increase confidence in nearby areas? (gaussian blur?) or increase weight of neighboring pixels. 
    #
    #       add a system to remove the pre-injection frames from consideration? e.g. find a method to detect when motion starts and only consider frames after that point
    #           or create a way to remove pre-injection noise. (done partially by only starting to write masks after firstFrameNumber and 5 consecutive high mag frames)
    #
    #       add either full motion confidence to areas inside dark regions with no detected motion and motion detected around it, or set motion values
    #           under a certain threshold to 1. (will have to cut out areas outside chamber first)
    #
    #       Option to add an automated weight setting system which tries to optimize weights based on some criteria?
    #           Will be a lot of work to implement though.
    #
    #       Add a system to keep areas of high motion during playback to fill in gaps in the detected area when the center of the spray is not detected well.

    # PROBLEMS: Thresholding is tricky, also combining the different confidence values is tricky
    #           Maybe values over 0.4 in mag should not be set to 1.0 but a lower value, to reduce the dominance of mag in the combination?

    #           Optical flow causes a small area around the chamber walls to be detected as motion. Need to find a way to remove that.
    #
    #           Reconsider high motion detection method: currently if any pixel in frame has mag > 0.5 for 5 consecutive frames, masks are written.
    #
    #           Reconsider cone mask, instead maybe use it to cut off areas outside the cone after combining scores?
    #
    #           Rethink dynamic thresholding method, currently uses 95th percentile of combined score, maybe use Otsu or a lower percentile?

    # maybe if magnitude is close to zero then intensity should have more weight? not sure how to implement that nicely though.

    # --- Combine per-pixel intensity, optical-flow magnitude, and freehand mask ---
    # Parameters: weights (normalized internally) and binary threshold on combined score (0.0 - 1.0)

    w_intensity = 0.4   # weight for per-pixel light intensity
    w_magnitude = 0.6  # weight for optical flow magnitude
    w_freehand = 0.3    # weight for freehand mask
    w_cone = 0.2    # weight for cone mask
    intensity_gamma = 3.0  # gamma correction for intensity score to amplify differences in dark areas, higher = more contrast

    
    if use_intensity_only:
        w_magnitude = 0
    if use_cumulative_as_mask:
        w_magnitude = 0
        # w_cone = 0
        w_intensity = 1.0


    # Create cone mask from spray origin (25 degrees upper and lower from horizontal, to the right)
    # Inside cone: 1.0, outside: linearly decrease to 0 over 45 degrees
    cone_mask = np.zeros((height, width), dtype=np.float32)
    origin_x, origin_y = spray_origin
    cone_angle = 25  # degrees
    falloff_angle = 50  # degrees over which it decreases to 0
    for i in range(height):
        for j in range(width):
            dx = j - origin_x
            dy = i - origin_y
            if dx > 0:
                angle = np.degrees(np.arctan2(dy, dx))
                abs_angle = abs(angle)
                if abs_angle <= cone_angle:
                    cone_mask[i, j] = 1.0
                elif abs_angle <= cone_angle + falloff_angle:
                    cone_mask[i, j] = 1.0 - (abs_angle - cone_angle) / falloff_angle
                # else 0.0
    cone_mask_f = cone_mask

    # Prepare combined masks array (final binary masks) and a diagnostic combined score array
    combined_masks = np.zeros_like(video_strip, dtype=np.uint8)
    final_cluster_masks = np.zeros_like(video_strip, dtype=np.uint8)
    intensity_scores = np.zeros_like(video_strip, dtype=np.float32)
    mag_scores = np.zeros_like(video_strip, dtype=np.float32)
    cumulative_masks = np.zeros_like(video_strip, dtype=np.uint8)

    # Load freehand mask created earlier by the user (expects single-channel binary image "mask.png")
    freehand_mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
    if freehand_mask is None:
        print("Warning: 'mask.png' not found — proceeding without freehand mask")
        freehand_mask_f = np.zeros((height, width), dtype=np.float32)
    else:
        # Resize to match frames if necessary, keep nearest neighbour to preserve binary nature
        if freehand_mask.shape != (height, width):
            freehand_mask = cv2.resize(freehand_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        # Normalize to 0.0-1.0
        freehand_mask_f = (freehand_mask > 0).astype(np.float32)

    # Normalize weights
    total_w = w_intensity + w_magnitude + w_freehand + w_cone
    norm_intensity = w_intensity / total_w
    norm_magnitude = w_magnitude / total_w
    norm_freehand = w_freehand / total_w
    norm_cone = w_cone / total_w

    eps = 1e-6
    high_mag_counter = 0
    cumulative_mask = np.zeros((height, width), dtype=np.uint8)
    for idx in range(nframes):
        # --- Intensity: per-frame robust normalization invariant to lighting ---
        frame = video_strip[idx]
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame.copy()
        intensity = frame_gray.astype(np.float32)
        # Use percentile-based clipping (1st/99th) per frame so global brightness/contrast changes are normalized out
        if use_cumulative_as_mask:
            masked_pixels = intensity[cumulative_mask > 0]
            if len(masked_pixels) > 0:
                p_low, p_high = np.percentile(masked_pixels, (1.0, 99.0))
            else:
                p_low, p_high = np.percentile(intensity, (1.0, 99.0))  # fallback if no masked pixels
        else:
            p_low, p_high = np.percentile(intensity, (1.0, 99.0))
        if p_high - p_low > 1e-6:
            intensity_n = (intensity - p_low) / (p_high - p_low)
            intensity_n = np.clip(intensity_n, 0.0, 1.0)
            # invert so darker pixels -> higher score
            intensity_n = 1.0 - intensity_n
            # apply gamma to amplify differences in dark areas
            intensity_n = np.clip(intensity_n ** intensity_gamma, 0.0, 1.0)
        else:
            # fallback to absolute inversion if percentiles degenerate
            intensity_n = 1.0 - (np.clip(intensity, 0.0, 255.0) / 255.0)
            intensity_n = np.clip(intensity_n ** intensity_gamma, 0.0, 1.0)

        # Restrict intensity score to areas within cumulative_mask
        if use_cumulative_as_mask:
            intensity_n[cumulative_mask == 0] = 0

        intensity_scores[idx] = intensity_n

        # --- Optical flow magnitude: cap at mag_clip then normalize to 0..1 (values >= mag_clip -> 1) ---
        mag = mag_array[idx].astype(np.float32)
        mag_clip = 0.4  # absolute motion cutoff: anything higher considered motion and mapped to 1.0
        mag_clipped = np.clip(mag, 0.0, mag_clip)
        mag_n = mag_clipped / (mag_clip + eps)

        mag_scores[idx] = mag_n

        # Accumulate areas with mag_n == 1.0
        new_areas = (mag_n > 0.99).astype(np.uint8) * 255
        cumulative_mask = np.maximum(cumulative_mask, new_areas)
        cumulative_masks[idx] = cumulative_mask.copy() # TODO: Decide what to do with this, add to score does not work

        # Check for high magnitude values to start writing masks
        if idx >= firstFrameNumber:
            if np.any(mag >= 0.5): # threshold to consider "high" motion detected, intended to avoid noise before injection
                high_mag_counter += 1
            else:
                high_mag_counter = 0
        else:
            high_mag_counter = 0
        freehand = freehand_mask_f  # already 0.0 or 1.0

        # --- Cone mask normalized ---
        cone = cone_mask_f  # already 0.0 or 1.0

        # Replace empty freehand (no drawing) with ones so it doesn't zero-out the product
        if np.count_nonzero(freehand) == 0:
            freehand = np.ones_like(freehand, dtype=np.float32)

        # --- Combine: product (agreement) plus a small boost for single-strong signals ---
        comp_int = (intensity_n + eps) ** norm_intensity
        comp_motion = (mag_n + eps) ** norm_magnitude
        comp_free = (freehand + eps) ** norm_freehand
        comp_cone = (cone + eps) ** norm_cone

        # Assume components are already in [0,1]; combine as joint probability
        combined_score = comp_int * comp_motion * comp_free * comp_cone
        # Optional: Normalize to [0,1] if needed
        combined_score = combined_score / np.max(combined_score) if np.max(combined_score) > 0 else combined_score

        # Optional: map combined_score to 0..255 for diagnostics
        combined_255 = np.clip((combined_score * 255.0), 0, 255).astype(np.uint8)

        # Dynamic thresholding: use 95th percentile or Otsu (for intensity-only or cumulative)
        # Original: dynamic_threshold = np.percentile(combined_score, 80)
        if use_intensity_only or use_cumulative_as_mask:
            # Use Otsu's thresholding
            combined_uint8 = (combined_score * 255).astype(np.uint8)
            otsu_thresh, _ = cv2.threshold(combined_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dynamic_threshold = otsu_thresh / 255.0
        else:
            # Use 95th percentile
            dynamic_threshold = np.percentile(combined_score, 95)

        # Threshold to binary mask
        final_mask = (combined_score >= dynamic_threshold).astype(np.uint8) * 255

        # Exclude background areas
        final_mask[background_mask == 0] = 0

        # Small morphological cleanup to remove noise
        # kernel = np.ones((5,5), np.uint8)
        # final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        # final_mask = cv2.dilate(final_mask, kernel, iterations=1)
        # final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

        # Store only if high mag counter >= 5
        if high_mag_counter >= 5:
            combined_masks[idx] = final_mask
        else:
            combined_masks[idx] = np.zeros_like(final_mask)


        # final_cluster_mask = create_cluster_mask(final_mask, cluster_distance=30, alpha=30)
        # final_cluster_masks[idx] = final_cluster_mask

    print(f"Final masks computed with w_intensity={w_intensity}, w_magnitude={w_magnitude}, w_freehand={w_freehand}, w_cone={w_cone}, intensity_gamma={intensity_gamma}, use_cumulative_as_mask={use_cumulative_as_mask}, dynamic thresholding (Otsu if cumulative mask or intensity-only, else 95th percentile)")  

    # Show combined masks (press 'q' to quit, 'p' to pause)
    intensity_values = [] # store average intensities
    for i in range(firstFrameNumber, nframes):
        frame = video_strip[i]
        combined = combined_masks[i]
        # cluster = final_cluster_masks[i]

        # Compute mean intensity inside the mask
        mean_intensity = cv2.mean(frame, combined)
        intensity_values.append(mean_intensity)

        # clustered_overlay = overlay_cluster_outline(frame, otsu_optical_mask) #may not need clustering

        # cv2.imshow('Otsu + Optical flow', otsu_optical_mask)
        cv2.imshow('Combined Mask', combined)
        cv2.imshow('Original', frame)
        # cv2.imshow('Clustered Mask', cluster)
        cv2.imshow('Intensity Score', (intensity_scores[i] * 255).astype(np.uint8))
        cv2.imshow('Magnitude Score', (mag_scores[i] * 255).astype(np.uint8))
        cv2.imshow('Cumulative High Motion Mask', cumulative_masks[i])

        key = cv2.waitKey(100) & 0xFF
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


    ##############################
    # Extrapolation work in progress
    ##############################


    # IDEA: fill the area close to the nozzle, take an area close to the nozzle of the detection area (left most side to middle?)
    #       calculate the angle which it makes up, and fill the resulting cone.
    #       Maybe use a system which detects if the area around the nozzle is hidden, if not then don't run
    # PROBLEMS: detected area needs to be noise free (although probably should be anyway)
    #       Close point calculation will not work... not that it would've worked anyway
    #       Need to add a way to remove noise outside of detection as a pre-process, idea is to use the nozzle as the
    #       point from which 25(less?) degrees above and below and extends to end of frame and keep any detected areas inside that region.
    #       NOTE, keep areas outside of the angle IF they are connected to and area inside the angle. 

    # spray_origin = (1, height // 2) # Known spray origin (x, y), TODO: add a way to set this interactively

    # for i in range(firstFrameNumber, nframes):

    #     detected_mask = otsu_optical[i]

    #     final_mask = extrapolate_cone(detected_mask, spray_origin, min_points=1)

    #     cv2.imshow("Final Mask", final_mask)

    #     key = cv2.waitKey(100) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)


    # Known spray origin (x, y)
    # spray_origin = (1, height // 2)
    # backfiller = SprayConeBackfill(spray_origin)

    # for i in range(firstFrameNumber, nframes):

    #     # detection step
    #     detected_mask = otsu_optical[i]
    #     frame = video_strip[i]

    #     # Backfill missing left-side cone
    #     backfill_mask = backfiller.backfill(detected_mask)

    #     # Merge
    #     final_mask = cv2.bitwise_or(detected_mask, backfill_mask)

    #     # Visualization
    #     vis = frame.copy()
    #     # Ensure vis is 3-channel BGR (frame may be grayscale)
    #     if vis.ndim == 2 or (vis.ndim == 3 and vis.shape[2] == 1):
    #         vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    #     # Create red overlay and apply mask
    #     overlay = np.zeros_like(vis)
    #     overlay[:] = (0, 0, 255)
    #     final_mask = final_mask.astype(np.uint8) * 255

    #     cv2.copyTo(overlay, final_mask, vis)

    #     cv2.circle(vis, spray_origin, 4, (0, 255, 0), -1)

    #     cv2.imshow("Spray Tracking", vis)
    #     cv2.imshow("Detected Mask", detected_mask)
    #     cv2.imshow("Final Mask", final_mask)

    #     key = cv2.waitKey(100) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)


        # vis = frame.copy()
        # # Ensure vis is 3-channel BGR
        # if vis.ndim == 2 or (vis.ndim == 3 and vis.shape[2] == 1):
        #     vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        # # Ensure final_mask matches frame size
        # if final_mask.shape != vis.shape[:2]:
        #     final_mask = cv2.resize(final_mask, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
        # mask_bool = final_mask > 0
        # # Only assign if there are masked pixels to avoid NumPy assignment errors when mask is empty
        # if np.any(mask_bool):
        #     vis[mask_bool] = (0, 0, 255)
        # cv2.circle(vis, spray_origin, 4, (0, 255, 0), -1)



print("Processing complete.")


# import time
# if __name__ == '__main__':
#     from multiprocessing import freeze_support
#     freeze_support()  # Optional: Needed if freezing to an executable

#     start_time = time.time()
#     main()
#     elapsed_time = time.time() - start_time

#     print(f"Sequential main() finished in {elapsed_time:.2f} seconds.")
