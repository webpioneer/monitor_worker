import cv2
from ultralytics import YOLO, solutions
import argparse

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Define points for a line or region of interest in the video frame
line_points = [(20, 400), (1080, 400)]  # Line coordinates

# Specify classes to count, for example: person (0) and car (2)
classes_to_count = [0, 2]  # Class IDs for person and car

# Initialize the Object Counter with visualization options and other parameters
counter = solutions.ObjectCounter(
    view_img=False,  # Disable display during processing
    reg_pts=line_points,  # Region of interest points
    classes_names=model.names,  # Class names from the YOLO model
    draw_tracks=True,  # Draw tracking lines for objects
    line_thickness=2,  # Thickness of the lines drawn
)

def process_stream(stream_url, output_video_path, test_mode=False):
    # Open the video stream
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Error opening video stream: {stream_url}")
        return

    # Get video properties: width, height, and frames per second (fps)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Initialize the video writer to save the output video
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"Starting processing for stream: {stream_url}")

    # Process video frames in a loop
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print(f"Video frame is empty or video processing completed for stream: {stream_url}")
            break

        # Perform object tracking on the current frame, filtering by specified classes
        tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)

        # Use the Object Counter to count objects in the frame and get the annotated image
        im0 = counter.start_counting(im0, tracks)

        # Write the annotated frame to the output video
        video_writer.write(im0)

        # If test_mode is enabled, display the video
        if test_mode:
            cv2.imshow("Video Stream", im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture and writer objects
    cap.release()
    video_writer.release()
    if test_mode:
        cv2.destroyAllWindows()
    print(f"Completed processing for stream: {stream_url}")

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection and Counting")
    parser.add_argument('-t', '--test', action='store_true', help="Enable test mode to display video")
    args = parser.parse_args()

    # List of video stream URLs to process
    video_streams = [
        "http://219.101.88.67:60001/nphMotionJpeg?Resolution=640x480&Quality=Clarity",
        # Add more streams as needed
    ]

    # Process each video stream in the list
    for stream_url in video_streams:
        output_video_path = f"output_{stream_url.split('/')[-1]}.avi"  # Generate output path based on stream URL
        process_stream(stream_url, output_video_path, test_mode=args.test)
