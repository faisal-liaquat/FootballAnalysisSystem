from ultralytics import YOLO

#loading our yolo model that we trained on the roboflow datset
model = YOLO('models/best.pt')

#predicting objects in a video file ('input_videos/08fd33_0.mp4') and save the results
results = model.predict('input_videos/08fd33_0.mp4', save=True)

#print the first set of results
print(results[0])
print("=================================")

#iterating over the bounding boxes detected in the first frame of the video
#and printing each box's information (e.g., coordinates, confidence, class)
for box in results[0].boxes:
    print(box)
