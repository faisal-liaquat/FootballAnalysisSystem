from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        #dictionaries to store team colors and player-team assignments
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        #reshape the image to a 2D array of pixels for clustering
        image_2d = image.reshape(-1, 3)

        #perform K-means clustering with 2 clusters (assuming 2 teams)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        #extract the portion of the frame within the bounding box
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        #take the top half of the bounding box image (usually where the jersey is)
        top_half_image = image[0:int(image.shape[0] / 2), :]

        #get the clustering model for the top half image
        kmeans = self.get_clustering_model(top_half_image)

        #get the cluster labels for each pixel
        labels = kmeans.labels_

        #reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        #determine the cluster for the player's jersey by examining the corners of the image
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster  #assuming only 2 clusters, the other cluster is the player's

        #get the color of the player's cluster
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        #get the color for each detected player
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        #cluster the player colors into two teams
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        #assign the team colors based on the cluster centers
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        #check if the player's team has already been determined
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        #get the player's color
        player_color = self.get_player_color(frame, player_bbox)

        #predict the team for the player based on their color
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1  #adjust team_id to start from 1 instead of 0

        #special case for player ID 92 (assume they belong to team 1) #handling the goalkeepr case for posession
        if player_id == 92:
            team_id = 2

        #store the player's team in the dictionary
        self.player_team_dict[player_id] = team_id

        return team_id
