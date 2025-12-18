## FootballAnalysisSystem
This project utilizes YOLOv8 for player detection, means to cluster players into two teams, and ByteTracker for player tracking in football matches. The combination of these technologies enables comprehensive analysis of player movements and calculate team ball control during game. 

## Introduction 
Football Analysis Project aims to provide insightful analysis of football matches by leveraging state-of-the-art computer vision techniques. The project focuses on three main components:

- **Player Detection**: YOLOv8, a real-time object detection system, is employed to detect players in football videos.

- **Player Tracking**: Supervision ByteTracker is utilized for player tracking throughout the duration of football matches.

- **Team Clustering**: Players detected by YOLOv8 are clustered into two teams using K-means clustering algorithm. This step enables the segmentation of players based on their shirt color using two method foreground and center-box.

- **Calculate Ball Control**: Calculate the ball control of each team by calculating the time the ball is in the possession of each team.

## Clustering Method 
- **Foreground method** the foreground method is based on the top-half image of the player. The top-half image is obtained by cropping the bounding box of the player. The top-half image clustering to two clusters using K-means clustering algorithm. To get the player shirt color and background color which is the most dominant color in the image. 
<img width="1890" height="913" alt="image" src="https://github.com/user-attachments/assets/f267aa24-5559-4b20-940e-b667e70a0511" />

- **Center-box method** The center-box method is based on the center box of the player. The center box is obtained by cropping the center box of the player. The center box clustering to two clusters using K-means clustering algorithm. To get the player shirt color which is usually in the center box of player bounding box.
<img width="1890" height="913" alt="image" src="https://github.com/user-attachments/assets/cc228a03-1c38-4788-893d-88c2d96accda" />

- **Siglip method (NEW)** The siglip method is based on the visual embedding of the player. The visual embedding is obtained by passing the player image through a pre-trained model. The visual embedding clustering to two clusters using K-means clustering algorithm. To get the player team assignment.

## Dependencies 
Before running the project, ensure you have the following dependencies installed:

- numpy
- opencv_python
- pandas
- scikit_learn
- supervision
- ultralytics
- torch
- transformers
- umap_learn

## Installation
To install and set up the project, follow these steps:
- **Clone the Repository**: Clone this repository to your local machine using the following command:
     git clone https://github.com/faisal-liaquat/FootballAnalysisSystem
- **Install Dependencies**
- Download Pre-trained Weights: Download the pre-trained weights for YOLOv8X from the following link: https://drive.google.com/file/d/1-0adL_JHRt7h93qWNxgq2NXYW-FLJjij/view

