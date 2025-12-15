################################
# Removes background from a frame using the first frame as reference
################################
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

################################
# Rotates each frame in the video by a specified angle
################################
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

################################
# Creates a video strip by extracting the central strip of each frame
################################
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

###############################
# Takes a video and returns the first frame with mean intensity above a threshold
################################
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

################################
# Applies a binary threshold to each frame in the video, UNUSED
################################
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

################################
# Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to each frame in the video
################################
def applyCLAHE(video, clipLimit=2.0, tileGridSize=(8,8)):
    import cv2
    import numpy as np

    nframes = video.shape[0]
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    for i in range(nframes):
        frame = video[i]
        frame = clahe.apply(frame)
        video[i] = frame

    return video

def applyLaplacianFilter(video):
    import cv2
    import numpy as np

    nframes = video.shape[0]

    for i in range(nframes):
        frame = video[i]
        frame = cv2.Laplacian(frame, cv2.CV_64F)
        frame = cv2.convertScaleAbs(frame)
        video[i] = frame

    return video

def applyDoGfilter(video, ksize1=5, ksize2=9):
    import cv2
    import numpy as np

    nframes = video.shape[0]

    for i in range(nframes):
        frame = video[i]
        blur1 = cv2.GaussianBlur(frame, (ksize1, ksize1), 0)
        blur2 = cv2.GaussianBlur(frame, (ksize2, ksize2), 0)
        dog = cv2.subtract(blur1, blur2)
        dog = cv2.convertScaleAbs(dog)
        video[i] = dog

    return video

def adaptiveGaussianThreshold(video, maxValue=255, blockSize=11, C=2):
    import cv2
    import numpy as np

    nframes = video.shape[0]

    for i in range(nframes):
        frame = video[i]
        frame = cv2.adaptiveThreshold(frame, maxValue, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, blockSize, C)
        video[i] = frame

    return video