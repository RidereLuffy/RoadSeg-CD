"""
Codes are modified based on https://github.com/anilbatra2185/road_connectivity
"""

import numpy as np
import cv2
from data_utils import affinity_utils

satellite_image = cv2.imread("dataset/train/sat.jpg")
road_mask = cv2.imread("dataset/train/mask.png",0)
road_mask = road_mask.astype(np.float)
# Smooth the road graph with tolerance=4 and get keypoints of road segments
keypoints = affinity_utils.getKeypoints(np.copy(road_mask), smooth_dist=4)
bin_size = 5
# generate orienation mask in euclidean and polar domain
vecmap_euclidean,orienation_angles = affinity_utils.getVectorMapsAngles(road_mask,keypoints,theta=10,bin_size=bin_size)#theta is related to the road width