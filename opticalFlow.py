def opticalFlowFarnebackCalculation(prev_frame, frame):
    import cv2

    # Convert to grayscale
    if len(frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame.copy()
        gray = frame.copy()

    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, 
                                        None, # type: ignore
                                        0.5,  # pyramid scale
                                        3,    # levels
                                        15,   # window size
                                        3,    # iterations
                                        5,    # poly_n
                                        1.2,  # poly_sigma
                                        0)    # type: ignore # flags

    return flow

def opticalFlowDeepFlowCalculation(prev_frame, frame, deepflow):
    import cv2
    
    # Convert to grayscale
    if len(frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame.copy()
        gray = frame.copy()

    flow = deepflow.calc(prev_gray, gray, None) # type: ignore

    return flow

