import numpy as np
import cv2


class ViewTransformer():
    def __init__(self):
        #define the dimensions of the ptich in meters
        court_width = 68  #width of the pitch in meters
        court_length = 23.32  #length of the football pitch in meters

        #vertices of the trapezoid in the image (camera view)
        #these need to be calculated based on the actual camera view
        self.pixel_vertices = np.array([[110, 1035],  #nottom-left
                                        [265, 275],   #top-left
                                        [910, 260],   #top-right
                                        [1640, 915]]) #bottom-right

        #vertices of the desired rectangular view (top-down view)
        self.target_vertices = np.array([
            [0, court_width],          #bottom-left
            [0, 0],                    #top-left
            [court_length, 0],         #top-right
            [court_length, court_width] #bottom-right
        ])

        #convert the vertices to float32 type for the perspective transformation
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        #cmpute the perspective transform matrix
        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        #convert the point to integer and check if it is inside the polygon
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None  #return None if the point is outside the polygon

        #reshape the point for perspective transformation
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        #apply the perspective transformation
        transform_point = cv2.perspectiveTransform(reshaped_point, self.persepctive_trasnformer)
        #reshape the transformed point back to original shape
        return transform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        #iterate through all tracked objects and their tracks
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    #get the adjusted position of the tracked object
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    #transform the position to the top-down view
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    #add the transformed position to the track information
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
