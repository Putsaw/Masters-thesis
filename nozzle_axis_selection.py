import numpy as np
import matplotlib.pyplot as plt
import cv2
import pycine.file as cine  # You must have pycine installed
from pathlib import Path

# === Configuration ===
cine_file_path = r"G:\2024_Internal_Transient_Sprays\BC20240524_ReactiveSpray_HZ4\Cine\T8\SchCam2.cine"
frame_num = 40

# === Utilities ===
def read_frame(cine_file_path, frame_offset, width, height):
    with open(cine_file_path, "rb") as f:
        f.seek(frame_offset)
        frame_data = np.fromfile(f, dtype=np.uint16, count=width * height).reshape(height, width)
    return frame_data

def load_single_frame(cine_file_path, frame_num):
    header = cine.read_header(cine_file_path)
    width = header['bitmapinfoheader'].biWidth
    height = header['bitmapinfoheader'].biHeight
    frame_offsets = header['pImage']
    total_frames = len(frame_offsets)

    if frame_num >= total_frames:
        raise ValueError(f"Frame number exceeds total frames: {total_frames}")
    
    frame_data = read_frame(cine_file_path, frame_offsets[frame_num], width, height)
    return frame_data

def normalize_frame(frame):
    return (frame.astype(np.float32) / np.iinfo(frame.dtype).max * 255).astype(np.uint8)

# === GUI to Select Origin and Angle ===
def select_origin_and_angle(img):
    fig, ax = plt.subplots()

    ax.imshow(img, cmap='gray')
    ax.invert_yaxis()
    ax.set_title("Click to select the spray origin (use mouse wheel to zoom)")
    plt.axis('on')

    def zoom(event):
        base_scale = 1.2
        # Get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return

        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1
            print(f"Unknown scroll event: {event.button}")

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rely = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * relx, xdata + new_width * (1 - relx)])
        ax.set_ylim([ydata - new_height * rely, ydata + new_height * (1 - rely)])
        ax.figure.canvas.draw()

    # Connect scroll event to zoom handler
    fig.canvas.mpl_connect('scroll_event', zoom)

    # Wait for user to click
    pts = plt.ginput(1)
    if len(pts) < 1:
        print("No point selected.")
        plt.close(fig)
        return None, None

    origin = pts[0]
    x0, y0 = origin
    print(f"Selected origin: x = {x0:.1f}, y = {y0:.1f}")

    # Ask for angle
    angle_deg = float(input("Enter spray angle in degrees (0 = horizontal right, counter-clockwise positive): "))
    angle_rad = np.deg2rad(angle_deg)

    # Line drawing
    length = max(img.shape) * 1.5
    x1 = x0 - length * np.cos(angle_rad)
    y1 = y0 - length * np.sin(angle_rad)
    x2 = x0 + length * np.cos(angle_rad)
    y2 = y0 + length * np.sin(angle_rad)

    # Plot the line and the point
    ax.plot(x0, y0, 'ro', label='Origin')
    ax.plot([x1, x2], [y1, y2], 'r-', label=f'{angle_deg:.1f}°')
    ax.legend()
    ax.set_title("Spray Axis Visualization (zoom with scroll)")
    plt.show()

    return (x0, y0), angle_deg

'''def select_origin_and_angle(img):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title("Click to select the spray origin")
    plt.axis('on')

    # Get single click for origin
    pts = plt.ginput(1)
    if len(pts) < 1:
        print("No point selected.")
        plt.close(fig)
        return None, None

    origin = pts[0]
    x0, y0 = origin
    print(f"Selected origin: x = {x0:.1f}, y = {y0:.1f}")

    # Ask for angle in degrees
    angle_deg = float(input("Enter spray angle in degrees (0 = horizontal right, counter-clockwise positive): "))
    angle_rad = np.deg2rad(angle_deg)

    # Line parameters
    length = max(img.shape) * 1.5
    x1 = x0 - length * np.cos(angle_rad)
    y1 = y0 - length * np.sin(angle_rad)
    x2 = x0 + length * np.cos(angle_rad)
    y2 = y0 + length * np.sin(angle_rad)

    # Draw origin and line
    ax.plot(x0, y0, 'ro', label='Origin')
    ax.plot([x1, x2], [y1, y2], 'r-', label=f'Angle = {angle_deg:.1f}°')
    ax.legend()
    ax.set_title("Spray Axis Visualization")
    plt.show()

    return (x0, y0), angle_deg'''

# === Main ===
if __name__ == "__main__":
    frame = load_single_frame(cine_file_path, frame_num)
    frame_norm = normalize_frame(frame)
    origin, angle_deg = select_origin_and_angle(frame_norm)

    if origin:
        print(f"Spray origin: {origin}")
        print(f"Spray angle (degrees): {angle_deg}")
