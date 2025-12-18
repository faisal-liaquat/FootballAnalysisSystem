from ultralytics import YOLO
import supervision as sv  #import supervision library for handling detections and tracking
import pickle
import os
import pandas as pd
import cv2
import numpy as np
import sys


sys.path.append('../')

#import utility functions for bounding box manipulations
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        #initialize the YOLO model using the provided model path
        self.model = YOLO(model_path)
        #initialize the ByteTrack tracker for tracking ids and stuff
        self.tracker = sv.ByteTrack()

    #method to add position (center or foot) to each tracked object's bounding box
    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)  #get center position for the ball
                    else:
                        position = get_foot_position(bbox)  #get foot position for players and referees
                    tracks[object][frame_num][track_id]['position'] = position  #add position to track info

    #method to interpolate missing ball positions in the frames
    def interpolate_ball_positions(self, ball_positions):
        #extract bounding boxes of the ball from each frame
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        #convert to DataFrame for interpolation
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        #interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        #convert back to list format
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    #method to detect objects in video frames
    def detect_frames(self, frames):
        batch_size = 20  #batch size
        detections = []  #list to store detections

        #processing frames in batches
        for i in range(0, len(frames), batch_size):
            #using the YOLO model to predict detections for the current batch of frames
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            #add the batch detections to the overall detections list
            detections += detections_batch

        return detections  #return all the detections

    #getting object tracks from frames
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        #load pre-calculated tracks from a stub file if available
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        #get detections for the given frames
        detections = self.detect_frames(frames)

        #dictionaries to store bounding boxes for each frame
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            #extract class names from the detection
            cls_names = detection.names
            #create a reverse mapping of class names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            print(cls_names)

            #cnvert detections to a format that the supervision library can work with
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #convert goalkeeper detections to player objects
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            #update tracker with detections
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            #dictionaries for current frame
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            #process tracked objects and store bounding boxes
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            #process detections to store ball bounding boxes
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        #save the tracks
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    #draw an ellipse around a tracked object
    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])  #get the bottom y-coordinate of the bounding box
        x_center, _ = get_center_of_bbox(bbox)  #get the center x-coordinate of the bounding box
        width = get_bbox_width(bbox)  #get the width of the bounding box

        #drawing an ellipse around the object's bounding box
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        #draw a rectangle to display the track ID
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            #draw the rectangle
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            #draw the track ID inside the rectangle
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    #draw a triangle above a tracked object's bounding box (pointer)
    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])  #top y-coordinate of the bounding box
        x, _ = get_center_of_bbox(bbox)  #get the center x-coordinate of the bounding box

        #define the points of the triangle
        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        #draw the triangle
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    #draw ball control information on the frame
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        #draw a semi-transparent rectangle to display ball control information
        overlay = frame.copy()  #create a copy of the current frame for overlay
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)  #draw a white rectangle on the overlay
        alpha = 0.4  #set the transparency level
        #blend the overlay with the original frame to create a semi-transparent effect
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        #get the ball control data up to the current frame
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        #calculate the number of frames each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        #calculate the ball control percentage for each team
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        #display the ball control percentage for Team 1 and 2 on the frame
        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 3)

        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 3)


        return frame

    #draw annotations on the video frames
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []  #list to store annotated frames

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            #draw players on the frame
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))  #default color is red
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):  #draw triangle if player has the ball
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            #draw referees on the frame
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255), track_id=None)

            #draw the ball on the frame
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            #draw team ball control (possession) information on the frame
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)  # Add the annotated frame to the output list

        return output_video_frames  #return the list of annotated frames
