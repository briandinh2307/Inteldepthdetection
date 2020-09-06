import numpy as np
import cv2
import math
import IntelCamera
# import PredictionModel

class Blob:
    """Store all parameters of each individual object"""

    def __init__(self):
        self.binary_object = None           # binary image of of the ROI
        self.infrared_object = None
        self.ID = None                      # N/A
        self.prediction_result = None              # N/A
        self.cont_frame = None              # number of continous frames the object appeared
        self.updated = None                 # check if object appear in current frame
        self.last_prediction_time = None    # N/A
        self.temp_contour = None            # object number in the current frame
        self.cnt_parameter = None           # the bounding box parameter of the object [x, y, w, h]
        self.real_area = None               # Centimeter square
        self.real_height = None             # Centimeter
        self.real_width = None              # Centimeter
        self.distance = None                # distance in centimeter
        self.flag = 0                       # Object available for save/display
        self.cropped_object = np.zeros((480, 640), dtype=np.uint8)  # ROI put in the 640x480 window
        self.cr_re_binary = None
        self.cr_re_infrared = None

    def checkBelong(self, new_image):
        """Check if new object appeared in the previous frames"""
        return np.any(np.logical_and(self.cropped_object, new_image))

    def saveObject(self, binary_object, bounding_box, distance, area, width, height, contour_th):
        """Save object parameters"""

        [x, y, w, h] = bounding_box
        self.cnt_parameter = bounding_box
        self.distance = distance
        self.binary_object = binary_object
        self.cropped_object = np.zeros((480, 640), dtype=np.uint8)
        self.cropped_object[y:y+h, x:x+w] = binary_object
        self.real_area = area
        self.real_width = width
        self.real_height = height
        self.temp_contour = contour_th

class DepthProcessing:
    """Do background subtraction based on background depth. 
       Crop and store each object separately. Perform simple noise removal"""

    def __init__(self, camera):
        
        self.Intel_camera = camera
        self.warmup_frames = 20
        self.raw_depth_diff = 300           # Threshold between the background and current depth frame
        self.min_motion_frames = 10         # Number of continuous frames object must appear in the scene
        self.threshold_blobs_area = 1200    # Pixel area
        self.threshold_real_area = 150.0    # Centimeter square
        self.threshold_real_width = 15      # Centimeter
        self.threshold_real_height = 15     # Centimeter
        self.background_frame = None        # background
        self.processed_frame = np.zeros((480, 640), dtype=np.uint8)
        self.store_blobs = []               # store all satisfied blobs as a list
        self.temp_store_blobs = []          # temporarily store objects in the current frame
        self.current_frame = None           # save current frame

    def saveBackground(self):
        """Wait for the first 20 frames to warm up and store the 21st frame as background
           Perform average background for every new frame"""

        if (self.warmup_frames > 0):
            self.warmup_frames -= 1
            return False
        else:
            if (self.background_frame is None):
                self.background_frame = self.current_frame.copy().astype('float')
            else:
                cv2.accumulateWeighted(self.current_frame, self.background_frame, 0.00003)
            return True

    def framesDifferencing(self):
        """Take the difference between the current frame and background to subtract background
           Return 8-bit binary frame image"""
        
        diff_frame = cv2.absdiff(self.current_frame, abs(self.background_frame).astype(np.uint16))
        diff_frame = cv2.threshold(diff_frame, self.raw_depth_diff, 255, cv2.THRESH_BINARY)[1]
        
        # Object too close to the camera will have depth = 0
        mask_data = np.where(self.current_frame <= 0, False, True)
        diff_frame[mask_data == False] = 0
        return diff_frame.astype(np.uint8)

    def saveBlob(self, ROI, bounding_cnt, distance, area, width, height, contour_th):
            
        b = Blob()
        b.saveObject(ROI, bounding_cnt, distance, area, width, height, contour_th)
        self.temp_store_blobs.append(b)

    def smallBlobsRemoval(self, frame_w_blobs, time):
        """Remove noise. Small blobs are removed"""

        contours, hierarchy = cv2.findContours(frame_w_blobs, cv2.RETR_EXTERNAL, \
                                               cv2.CHAIN_APPROX_SIMPLE)
        
        for i in range(len(contours)):
            cnt = contours[i]

            # Consider number of pixels to remove small blobs
            if cv2.contourArea(cnt) <= self.threshold_blobs_area:
                cv2.drawContours(frame_w_blobs, [cnt], -1, 0, -1, 1)
            else:
                # find centroid of the object
                M = cv2.moments(cnt)
                cX = int(M["m10"]/M["m00"])
                cY = int(M["m01"]/M["m00"])
                
                # Blobs must be a hold instead of fragment
                check_centroid = np.sum(frame_w_blobs[cY-50:cY+50, cX-50:cX+50])
                if (check_centroid == 0):
                    cv2.drawContours(frame_w_blobs, [cnt], -1, 0, -1, 1)
                    continue
                
                # Calculate the dimension of the object's pixels.
                # dist_to_centroid = self.Intel_camera.get_distance(cX, cY) * 100
                dist_to_centroid = self.current_frame[cY, cX] * self.Intel_camera.depth_scale * 100
                cm_per_pixel = (math.tan(self.Intel_camera.depth_ver_FOV/2) \
                                *2*dist_to_centroid) / 480.0

                # Clean edge from surrounding noise
                [x, y, w, h] = cv2.boundingRect(cnt)
                ROI = frame_w_blobs[y:y+h, x:x+w]
                clean_edge = ROI/255 * self.current_frame[y:y+h, x:x+w]
                ROI = np.where(clean_edge > self.current_frame[cY, cX] + 300, 0, ROI)
                ROI = np.where(clean_edge < self.current_frame[cY, cX] - 300, 0, ROI)

                # Check for number of white pixels in the blob
                num_of_pixel = cv2.countNonZero(ROI)

                # Find area according to the number of white pixels
                real_area = num_of_pixel * cm_per_pixel * cm_per_pixel
                real_width = round(w * cm_per_pixel, 2)
                real_height = round(h * cm_per_pixel, 2)

                # Consider the approximate real_area, width and height to remove small blobs
                if (real_area <= self.threshold_real_area or 
                    real_width <= self.threshold_real_width or  
                    real_height <= self.threshold_real_height):

                    cv2.drawContours(frame_w_blobs, [cnt], -1, 0, -1, 1)
                    continue
                
                frame_w_blobs[y:y+h, x:x+w] = ROI

                # Save the satisfied object
                if (time == 0):
                    self.saveBlob(ROI, [x, y, w, h], dist_to_centroid, real_area, real_width, real_height, i)
    
        return frame_w_blobs


    def noiseRemoval(self, noisy_frame, iteration = 2):
        """Remove noise"""

        self.temp_store_blobs = []
        noisy_frame = cv2.medianBlur(noisy_frame, 5)
        
        # Only consider objects within 5 meters
        clipping_distance = 4 / self.Intel_camera.depth_scale
        noisy_frame = np.where(self.current_frame > clipping_distance, 0, noisy_frame)
        
        # Perform at least 2 iteration to have exact bounding box
        while iteration > 0:
            iteration -= 1      
            blobs_cleaned = self.smallBlobsRemoval(noisy_frame, iteration)

        binary_frame = cv2.morphologyEx(blobs_cleaned, cv2.MORPH_CLOSE, (7,7))    
        return binary_frame  
                    
    def min_frame_detection(self):
        """Check if object continiously appear in the frame.
           If satisfied, store all parameters"""

        result_1 = []
        result_2 = []
        temp_contour_checking = []

        for new in self.temp_store_blobs:
            found = 0               # Check if the new object appeared in the previous frame
            for old in self.store_blobs:

                find_prev_obj = old.checkBelong(new.cropped_object)
                if (find_prev_obj == True):     # Object appeared in the previous frame
                    found = 1
                    # Check for how long object has appeared in the scene
                    if (old.cont_frame >= self.min_motion_frames):  
                        old.flag = 1        # Flag to confirm object can be output for display or saving
                    else:
                        old.cont_frame += 1
                    old.updated = 1         # previous object has appear in the current frame
                    old.saveObject(new.binary_object, new.cnt_parameter, new.distance, new.real_area, \
                                   new.real_width, new.real_height, new.temp_contour)
                    # result_1.append(old)
                # elif (old.updated == 1):
                    # result_1.append(old)
                # else:
                    # old.updated = 0
                    # result_1.append(old)
                result_1.append(old)

            if (found == 0):
                new.cont_frame = 0
                new.updated = 1
                result_1.append(new)
        
        if (len(result_1) > 0):
            for x in result_1:
                if (x.updated == 1):
                    x.updated = 0
                    # Moving objects may cause duplicate. Delete duplicates
                    if (x.temp_contour not in temp_contour_checking):
                        temp_contour_checking.append(x.temp_contour)
                        result_2.append(x)

        self.store_blobs = result_2
        return self.store_blobs

    def objectResizing(self, type = 'depth'):
        
        assert type == 'depth' or type == 'infrared', 'Can only resize "depth" or "infrared"'
        image_width = 640
        image_height = 480

        for blob in self.store_blobs:
            
            if blob.flag == 1:
                if (type == 'depth'):
                    image_to_resize = blob.binary_object
                elif (type == 'infrared'):
                    image_to_resize = blob.infrared_object
                scale_percent = blob.distance / 200.0 
                resized_width = int(image_to_resize.shape[1] * scale_percent)
                resized_height = int(image_to_resize.shape[0] * scale_percent)

                if (resized_width < image_width and resized_height < image_height):
                    resized_image = cv2.resize(image_to_resize, (resized_width, resized_height))
                    startx = image_width//2 - resized_width//2
                    starty = image_height//2 - resized_height//2
                    result = np.ones((image_height, image_width), dtype=np.uint8) * 100
                    result[starty: starty+resized_height, startx: startx+resized_width] = resized_image   
        
                if (type == 'depth'):
                    blob.cr_re_binary = result
                elif (type == 'infrared'):
                    blob.cr_re_infrared = result

    def depthResizingHandler(self):
        self.objectResizing(type = 'depth')
        
    def infraredResizingHandler(self):
        self.objectResizing(type = 'infrared')

    def motionDetection(self, raw_depth, width = 15, height = 15, area = 150.0, min_motion = 20):
        """Main function for detection. 
           Save background -> subtract background on new frames -> Remove noise"""

        self.threshold_real_width = width
        self.threshold_real_height = height
        self.threshold_real_area = area
        self.min_motion_frames = min_motion
        self.current_frame = raw_depth
        # Save background
        saved = self.saveBackground()
        if (saved == False):
            return self.processed_frame
        
        # Perform background subtraciton
        background_differencing = self.framesDifferencing()

        # Remove noise
        self.processed_frame = self.noiseRemoval(background_differencing, iteration = 2)
        
        # Check consistent object
        self.min_frame_detection()

        # Resize objects according to shape
        self.depthResizingHandler()
        return self.processed_frame

    def processInfrared(self, infrared):
        for blob in self.store_blobs:
            if blob.flag == 1:
                [x, y, w, h] = blob.cnt_parameter
                cropped_infrared = infrared[y:y+h, x:x+w]
                blob.infrared_object = np.where(blob.binary_object > 0, cropped_infrared, 0)
        self.infraredResizingHandler()

    def prediction(self, type = 'infrared'):

        assert type == 'depth' or type == 'infrared', 'Can only predict "depth" or "infrared" image'
        for blob in self.store_blobs:

            if blob.flag == 1:
                if (type == 'depth'):
                    image_to_predict = blob.cr_re_binary
                elif (type == 'infrared'):
                    image_to_predict = blob.cr_re_infrared
                
                kind, time = PredictionModel.predict(image_to_predict, blob.last_prediction_time)
                if (kind is not None):
                    blob.prediction_result = kind
                    blob.last_prediction_time = time

    def infraredPredictionHandler(self):
        self.prediction(type = 'infrared')

    def depthPredictionHandler(self):
        self.prediction(type = 'depth')
