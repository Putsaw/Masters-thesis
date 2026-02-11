def set_spray_origin(file, rotated_video, firstFrameNumber, nframes, height):    
    import cv2
    import json    
    import os

    # Load saved spray origins
    origins_file = 'spray_origins.json'
    if os.path.exists(origins_file):
        with open(origins_file, 'r') as f:
            spray_origins = json.load(f)
    else:
        spray_origins = {}

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
        
        cv2.imshow('Set Spray Origin - Click on the nozzle', rotated_video[firstFrameNumber+100]) # Show a frame after firstFrameNumber for context, may need adjustment
        cv2.setMouseCallback('Set Spray Origin - Click on the nozzle', select_origin)
        
        current_frame = firstFrameNumber + 100
        while holder.point is None:
            key = cv2.waitKeyEx(10)
            if key == ord('q'):
                break
            elif key == 2424832:  # left arrow
                current_frame = max(firstFrameNumber, current_frame - 1)
                cv2.imshow('Set Spray Origin - Click on the nozzle', rotated_video[current_frame])
            elif key == 2555904:  # right arrow
                current_frame = min(nframes - 1, current_frame + 1)
                cv2.imshow('Set Spray Origin - Click on the nozzle', rotated_video[current_frame])
        cv2.destroyWindow('Set Spray Origin - Click on the nozzle')
        
        if holder.point is None:
            spray_origin = (1, height // 2)  # Default
        else:
            spray_origin = holder.point
        
        # Save
        spray_origins[file] = list(spray_origin)
        with open(origins_file, 'w') as f:
            json.dump(spray_origins, f)

        print(f"Spray origin for {file}: {spray_origin}")

    return spray_origin

