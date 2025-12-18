from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import numpy as np
import cv2

def main():
    #reading the video frames from a file
    video_frames = read_video('input_videos\\121364_8.mp4')

    #the object tracker with a pre-trained model
    tracker = Tracker('models/best.pt')

    #get object tracks from the video frames, optionally reading from a stub for faster processing
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    #add position information to the tracks (this might include bounding boxes, positions, etc.)
    tracker.add_position_to_tracks(tracks)

    #initialize the camera movement estimator with the first frame of the video
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    #calculate the camera movement for each frame, optionally reading from a stub
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    #adjust the positions of the tracked objects to account for camera movement
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    #initialize the view transformer for transforming positions to a top-down view
    view_transformer = ViewTransformer()
    #transform the tracked positions to the top-down view
    view_transformer.add_transformed_position_to_tracks(tracks)

    #interpolate the positions of the ball between frames where it might be missing
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    #initialize the speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    #add speed and distance information to the tracks
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video5.avi')


if __name__ == '__main__':
    main()

