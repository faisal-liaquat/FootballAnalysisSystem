import cv2

#function to read a video from a source file and return its frames
def read_video(video_path):
    #initialize video capture object with the given video file path
    cap = cv2.VideoCapture(video_path)
    frames = []  #list to store video frames

    #loop to read frames from the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    return frames  # Return the list of frames

#function to save a list of frames as an output video
def save_video(output_video_frames, output_video_path):
    #video format to save it
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #initialize the VideoWriter object with output path, codec, FPS, and frame size
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))

    #loop through each frame in the list and write it to the output video
    for frame in output_video_frames:
        out.write(frame)

    out.release()  #release the VideoWriter object


