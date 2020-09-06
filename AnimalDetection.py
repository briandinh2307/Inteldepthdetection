
import pyrealsense2 as rs
import numpy as np
import cv2
import math
import time
import IntelCamera
import DepthProcessing
import os
from sys import argv
import json

assert len(argv) == 2, 'Need a path to the JSON detection requirement file.'
conf = json.load(open(argv[1]))



Intel_depth_camera = IntelCamera.IntelCamera()
Intel_depth_camera.enable_infrared()
Intel_depth_camera.prepare_streams()
Intel_depth_camera.enable_color_depth()
detection = DepthProcessing.DepthProcessing(Intel_depth_camera)

Intel_depth_camera.set_laser_power(20)

lastUploaded = time.strftime('%Y%m%d%H%M%S')

def saveImage(image_array):
    global lastUploaded
    timestamp = time.strftime('%Y%m%d%H%M%S')
    if(int(timestamp) - int(lastUploaded) >= 3):
        md = time.strftime('%m%d')
        for count, blob in enumerate(image_array):
            if blob.flag == 0:
                continue
            path_dir_cropped = 'Data/Cropped/' + str(md) + 'th'
            path_dir_resized = 'Data/Resized/' + str(md) + 'th'
            path_dir_infrared = 'Data/Infrared/' + str(md) + 'th'

            if os.path.isdir(path_dir_resized) is False:
                os.makedirs(path_dir_resized)
                os.makedirs(path_dir_infrared)
                    
            cv2.imwrite(path_dir_resized + '/' + str(timestamp) + 'th.' + str(count) + '.png', blob.cr_re_binary)
            if (blob.cr_re_infrared is not None):
                cv2.imwrite(path_dir_infrared + '/' + str(timestamp) + 'th.' + str(count) + '.png', blob.cr_re_infrared)
            lastUploaded = timestamp

def main():
    try:
        while True:
            # print('check fps:', time.strftime("%M%S"))
            Intel_depth_camera.run_camera()
        
            infrared_zero,_ = Intel_depth_camera.get_infrared_image()
            raw_depth = Intel_depth_camera.get_raw_depth()
            color_depth = Intel_depth_camera.get_color_depth()
            processed_vid = detection.motionDetection(raw_depth, width=conf['min_width'], height=conf['min_height'], area=conf['min_area'], min_motion=conf['min_motion_frames'])
            detection.processInfrared(infrared_zero)
            # detection.infraredPredictionHandler()
            
            # print(len(detection.store_blobs))
            saveImage(detection.store_blobs)
            for blob in detection.store_blobs:
                if blob.flag == 1: 
                    [x, y, w, h] = blob.cnt_parameter
                    width = blob.real_width
                    height = blob.real_height
                    cv2.rectangle(processed_vid, (x, y), (x+w, y+h), 255, 1)
                    cv2.putText(processed_vid, str(width), (x+w//2, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                    cv2.putText(processed_vid, str(height), (x-80, y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                    if (blob.prediction_result != None):
                        cv2.putText(processed_vid, str(blob.prediction_result), (x + w + 10, y + h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
            cv2.imshow('ColorDepth', color_depth)
            cv2.imshow('Motion', processed_vid)
            cv2.imshow('Infrared', infrared_zero)
            if (cv2.waitKey(25) == ord('q')):
                break

    finally:
        Intel_depth_camera.stop_camera()
        cv2.destroyAllWindows()
            
if __name__ == "__main__":
    main()

