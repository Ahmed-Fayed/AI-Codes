# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 13:14:39 2022

@author: ahmed
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import matplotlib.patches as patches
import cv2
import os
from tqdm import tqdm




isClosed = True
color = (255, 0, 0)
thickness = 4



x = 400
y = 500

width = 1400
height = 1400

verts = [[x, y], # left, top
         [x, height], # left, buttom
         [x + width, height], # right, buttom
         [x + width, y]] # right, top


# Get Poly Path
poly_path = mplPath.Path(verts)



src_videos_dir = ''
dest_dir = ''

for video_name in os.listdir(src_videos_dir):
    video_dest_dir = os.path.join(src_videos_dir, video_name)
    
    # Assert if dir is already exist or no
    if not (os.path.isdir(video_dest_dir)):
        os.mkdir(video_dest_dir)
    
    # Create a VideoCapture object and read from input file
    # If the input is taken from the camera, pass 0 instead of the video file name.
    cap = cv2.VideoCapture('B8A44F1EF69A_18-12-2021_12-58-34 PM.asf')
    # cap.set(propId=cv2.CAP_PROP_FOURCC, value=6)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            
            # Draw Polygon on frame
            frame = cv2.polylines(frame, [np.array(verts).reshape(-1, 1, 2)], 
                          isClosed, color, thickness)
            # Display the resulting frame
            frame = cv2.resize(frame, (800, 500))
            cv2.imshow('Frame',frame)
    
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
    
        # Break the loop
        else: 
            break
    
    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()
    













