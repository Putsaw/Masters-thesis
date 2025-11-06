from functions import *
from functions_videos import *
from functions_optical_flow import *
from boundary2 import *

from std_functions3 import *
import matplotlib.pyplot as plt
import os
import glob
import tkinter as tk
from tkinter import filedialog
from scipy.spatial import ConvexHull
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

# Hide the main tkinter window
root = tk.Tk()
root.withdraw()

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


    # for i in range(nframes):
        
    #     frame = video[i]
    #     cv2.imshow('Frame', frame)
    #     if cv2.waitKey(60) & 0xFF == ord('q'):
    #         break

    # Angle of Rotation
    rotation = -45

    # Strip cutting
    x_start = 1
    x_end = -1
    y_start = 100
    y_end = -250

    # Define the center of rotation (usually the center of the image)
    center = (width // 2, height // 2)
    # Define the rotation angle in degrees
    angle = 45 # Rotate by 45 degrees
    # Define scaling factor (1.0 means no scaling)
    scale = 1.0
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    firstFrame = video[4]
    bg_subtractor = cv2.createBackgroundSubtractorKNN()
    bg_subtractor.setHistory(30) #set amount of frames to affect the subtraction
    for i in range(nframes):

        frame = video[i]
        mask = bg_subtractor.apply(frame)

        cv2.imshow('subracted image', mask)
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break

        _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        if i == 300:
            # # em clustering requires at least 2 clusters to work, which is probably not good enough
            # # Apply EM clustering using Gaussian Mixture Model
            # coords = np.column_stack(np.where(binary > 0))
            # gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
            # gmm.fit(coords)
            # probs = gmm.predict_proba(coords)
            # max_probs = probs.max(axis=1)


            # # Define a probability threshold to identify noise
            # threshold = 1.0
            # is_noise = max_probs < threshold

            # # Visualize clustered signals vs. noise
            # plt.figure(figsize=(10, 5))

            # # Left plot: clustered signals and noise
            # plt.subplot(1, 2, 1)
            # plt.title("Clustered Signals vs. Noise")
            # plt.scatter(coords[~is_noise][:, 1], coords[~is_noise][:, 0], c='blue', label='Clustered')
            # plt.scatter(coords[is_noise][:, 1], coords[is_noise][:, 0], c='black', marker='x', label='Noise')
            # plt.legend()
            # plt.gca().invert_yaxis()

            # # Right plot: heatmap of max probabilities
            # plt.subplot(1, 2, 2)
            # plt.title("Max Probability Heatmap")
            # plt.scatter(coords[:, 1], coords[:, 0], c=max_probs, cmap='viridis')
            # plt.colorbar(label='Max Probability')
            # plt.gca().invert_yaxis()

            # plt.tight_layout()
            # plt.show()

            
            # Extract coordinates of non-zero pixels (signals)
            coords = np.column_stack(np.where(binary > 0))

            # Apply DBSCAN clustering
            db = DBSCAN(eps=5, min_samples=10, metric='cosine').fit(coords)

            # Get cluster labels (-1 indicates noise)
            labels = db.labels_

            # Number of clusters (excluding noise)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"Detected {n_clusters} clusters")

            # Plot clustered signals and noise
            plt.figure(figsize=(10, 5))

            # Clustered signals
            plt.subplot(1, 2, 1)
            for label in set(labels):
                if label == -1:
                    continue
                cluster_points = coords[labels == label]
                plt.scatter(cluster_points[:, 1], cluster_points[:, 0], label=f'Cluster {label}', s=5)
            plt.title('DBSCAN Clusters')
            plt.gca().invert_yaxis()
            plt.legend()

            # Noise points
            plt.subplot(1, 2, 2)
            noise_points = coords[labels == -1]
            plt.scatter(noise_points[:, 1], noise_points[:, 0], c='black', marker='x', label='Noise')
            plt.title('Detected Noise')
            plt.gca().invert_yaxis()
            plt.legend()

            plt.tight_layout()
            plt.savefig('dbscan_clusters.png')
            plt.show()


        # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Get outline points of contours
        # centers = []
        # for cnt in contours:
        #     outline = cnt.reshape(-1, 2).tolist()  # Convert contour points to a list of [x, y]
        #     centers.append(outline)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get outline points of contours
        outlines = []
        for cnt in contours:
            outline = cnt.reshape(-1, 2).tolist()
            outlines.append(outline)

        # Flatten all outline points into a single list for hull calculation
        all_points = [pt for outline in outlines for pt in outline]

        # Create a blank canvas (same size as binary)
        canvas = np.zeros_like(binary)

        # Optimize line drawing using Convex Hull if enough points
        if len(all_points) >= 3:
            hull = ConvexHull(all_points)
            hull_points = [all_points[i] for i in hull.vertices]
            for i in range(len(hull_points)):
                pt1 = tuple(hull_points[i])
                pt2 = tuple(hull_points[(i + 1) % len(hull_points)])
                cv2.line(canvas, pt1, pt2, 255, thickness=1)
        else:
            # Fallback to pairwise connection for few points
            for i in range(len(all_points)):
                for j in range(i + 1, len(all_points)):
                    cv2.line(canvas, tuple(all_points[i]), tuple(all_points[j]), 255, thickness=1)

        # Fill the enclosed area
        # contours_fill, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(canvas, contours_fill, -1, 255, thickness=cv2.FILLED)

        # Show result
        overlay = cv2.addWeighted(frame, 1, canvas, 1, 0)

        cv2.imshow('connected blob', overlay)
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break

        #frame = cv2.subtract(video[i], firstFrame)

    # Create rotated video strip
    video_strip = []
    for i in range(nframes):
        frame = video[i]
        rotated_image = cv2.warpAffine(frame, rotation_matrix, (width, height))
        #strip = rotated_image[y_start:y_end, x_start:x_end]
        strip = rotated_image
        video_strip.append(strip)

    # Basic thresholding visualization
    # for i in range(nframes):

    #     blurred = cv2.GaussianBlur(video_strip[i], (5, 5), 0)
    #     _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #     cv2.imshow('binary Image', binary)
    #     if cv2.waitKey(60) & 0xFF == ord('q'):
    #         break

    first_frame = video_strip[0]
    if len(first_frame.shape) == 3:
        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = first_frame.copy()

    # Create HSV image for visualization (3 channels)
    hsv = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255  # full saturation for color visualization

    for i in range(nframes):
        frame = video_strip[i]

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Compute dense optical flow using Farnebäck method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, 
                                            None,
                                            0.5,  # pyramid scale
                                            3,    # levels
                                            15,   # window size
                                            3,    # iterations
                                            5,    # poly_n
                                            1.2,  # poly_sigma
                                            0)    # flags

        # Convert flow to polar coordinates (magnitude and angle)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Use hue to encode direction and value to encode magnitude
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        motion_intensity = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        motion_intensity = np.uint8(motion_intensity)

        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Display
        cv2.imshow('Original', frame)
        cv2.imshow('Optical Flow', rgb_flow)
        cv2.imshow('Optical Flow (Motion Intensity)', motion_intensity)

        if cv2.waitKey(60) & 0xFF == ord('q'):
            break

        # Set previous frame to current
        prev_gray = gray


        # blurred = cv2.GaussianBlur(motion_intensity, (5, 5), 0)
        # _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # cv2.imshow('binary Image', binary)
        # if cv2.waitKey(60) & 0xFF == ord('q'):
        #     break

    cv2.destroyAllWindows()

                                    
            
# import time
# if __name__ == '__main__':
#     from multiprocessing import freeze_support
#     freeze_support()  # Optional: Needed if freezing to an executable

#     start_time = time.time()
#     main()
#     elapsed_time = time.time() - start_time

#     print(f"Sequential main() finished in {elapsed_time:.2f} seconds.")
