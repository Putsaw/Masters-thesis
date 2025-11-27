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

# Pipeline:
    # Compute optical flow between frames
    # Compute magnitude + direction maps
    # Cluster or threshold flow difference
    # Generate a mask
    # Clean mask with morphology

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
    rotated_video = vpf.createRotatedVideo(video, 45)
    video_strip = vpf.createVideoStrip(rotated_video)

    ##############################
    # Background Removal Visualization
    ##############################
    # firstFrameNumber = vpf.findFirstFrame(video_strip)
    firstFrameNumber = vpf.plot_mean_intensity(video_strip)

    first_frame = video_strip[firstFrameNumber]
    for i in range(nframes):
        frame = video_strip[i]
        foreground = vpf.removeBackground(frame, first_frame)
        cv2.imshow('Foreground', foreground)
        cv2.imshow('frame', frame)
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    """
    vpf.removeBackgroundThreshold(video_strip, threshold=30)
    for i in range(nframes):
        cv2.imshow('Original vid', video_strip[i])
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    """

    # ##############################
    # # DeepFlow Optical Flow Visualization
    # ##############################
    intensity_values = []       # store average intensities
    first_frame = video_strip[firstFrameNumber]
    prev_frame = first_frame

    for i in range(firstFrameNumber, nframes):
        frame = video_strip[i]

        # --- Compute DeepFlow optical flow ---
        flow = of.opticalFlowFarnebackCalculation(prev_frame, frame) # Farneback 0.3 threshold

        # Compute magnitude (motion strength) and angle (not needed here)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # threshold movement
        mask = (mag > 0.3).astype(np.uint8) * 255

        # clean up
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

        # Use clustering to get signal outlines
        clustered_overlay = cluster_signals(mask, frame, cluster_distance=50, alpha=40)

        # Compute mean intensity inside the mask
        # filled_mask = fill_largest_cluster(clustered_overlay, frame)
        # mean_intensity = cv2.mean(frame, filled_mask)
        # intensity_values.append(mean_intensity)

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

        print(f"Processed frame {i+1}/{nframes}")

    cv2.destroyAllWindows()

    # # --- Create shifted x-axis ---
    # frame_numbers = np.arange(firstFrameNumber, firstFrameNumber + len(intensity_values))

    # plt.plot(frame_numbers, intensity_values)
    # plt.xlabel("Frame Number")
    # plt.ylabel("Mean Intensity Inside Region")
    # plt.title("Intensity Over Time (Shifted)")
    # plt.show()



                                    
            
# import time
# if __name__ == '__main__':
#     from multiprocessing import freeze_support
#     freeze_support()  # Optional: Needed if freezing to an executable

#     start_time = time.time()
#     main()
#     elapsed_time = time.time() - start_time

#     print(f"Sequential main() finished in {elapsed_time:.2f} seconds.")
