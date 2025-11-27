def removeBackground(frame, first_frame):
    import cv2
    import numpy as np

    # Convert to grayscale
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    else:
        frame = frame.copy()
        first_frame = first_frame.copy()

    # Compute absolute difference between the frame and the background
    diff = cv2.absdiff(frame, first_frame)

    # Threshold the grayscale difference to create a binary mask
    _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    # Use the mask to extract the foreground from the original frame
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    return foreground

def createRotatedVideo(video, angle):
    import cv2
    import numpy as np

    nframes, height, width = video.shape
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_video = np.zeros_like(video)
    for i in range(nframes):
        rotated_frame = cv2.warpAffine(video[i], rotation_matrix, (width, height))
        rotated_video[i] = rotated_frame

    return rotated_video

def createVideoStrip(video, strip_height = None):
    # Come back later, spray height needs to be known beforehand
    import numpy as np

    nframes, height, width = video.shape
    if strip_height is None:
        strip_height = height

    video_strip = np.zeros((nframes, strip_height, width), dtype=video.dtype)

    for i in range(nframes):
        start_row = (height - strip_height) // 2
        end_row = start_row + strip_height
        video_strip[i] = video[i, start_row:end_row, :]

    return video_strip

def findFirstFrame(video, threshold=10):
    import numpy as np
    nframes = video.shape[0]

    for i in range(1, nframes):
        frame = video[i]
        # Compute mean brightness
        mean_intensity = frame.mean()

        if mean_intensity > threshold:
            return i
        
    return 0  # Default to first frame if no suitable frame is found

def plot_mean_intensity(video, threshold=10):
    import numpy as np
    import matplotlib.pyplot as plt
    nframes = video.shape[0]
    mean_values = []

    for i in range(nframes):
        frame = video[i]
        mean_intensity = frame.mean()
        mean_values.append(mean_intensity)

    # Plot the graph
    plt.plot(mean_values)
    plt.xlabel("Frame number")
    plt.ylabel("Mean intensity")
    plt.title("Mean Frame Intensity Over Time")
    plt.show()

    # (Optional) Return the first frame above threshold
    for i, m in enumerate(mean_values):
        if m > threshold:
            return i

    return 0

def removeBackgroundThreshold(video, threshold=30):
    #Consider making frame specific
    import cv2
    import numpy as np
    
    nframes = video.shape[0]
    for i in range(nframes):
        frame = video[i]
        _, frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
        video[i] = frame

    return video