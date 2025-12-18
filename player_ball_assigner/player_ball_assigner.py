import sys

# Add the parent directory to the system path to import modules from the utils folder
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        # Set the maximum distance a player can be from the ball to be considered as having the ball
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        # Get the center position of the ball's bounding box
        ball_position = get_center_of_bbox(ball_bbox)

        # Initialize variables to track the closest player
        minimum_distance = 99999
        assigned_player = -1

        # Iterate through each player
        for player_id, player in players.items():
            player_bbox = player['bbox']

            # Measure the distance from the ball to the left side of the player's bounding box
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            # Measure the distance from the ball to the right side of the player's bounding box
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            # Determine the minimum distance from the ball to the player's bounding box
            distance = min(distance_left, distance_right)

            # Check if the player is within the maximum distance to have the ball
            if distance < self.max_player_ball_distance:
                # Update the closest player if the distance is less than the current minimum distance
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        # Return the ID of the player closest to the ball, or -1 if no player is close enough
        return assigned_player
