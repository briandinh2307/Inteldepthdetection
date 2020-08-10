import pyrealsense2 as rs
from pyrealsense2 import stream
import numpy as np
import cv2
import datetime
import math

class IntelCamera:
    """Create and run Intel camera. First call prepare_streams() to set up
       then put the run_camera() in while loop. To get the data for display,
       call get_ () functions"""

    def __init__(self, depth_enable=1, infrared_enable=0, color_enable=0):
        """By default, only the depth stream is enabled"""

        self.depth_enable = depth_enable
        self.infrared_enable = infrared_enable
        self.color_enable = color_enable
        self.colorize_enable = 0
        self.laser_pwr = 50
        self.stream_width = 640
        self.stream_height = 480
        self.fps = 30

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.depth_ver_FOV = (57*math.pi) / 180.0

    def prepare_streams(self):
        """Set up the camera. Enable the required streams"""
        
        if (self.depth_enable):
            self.enable_depth()

        if (self.infrared_enable):
            self.enable_infrared()

        if(self.color_enable):
            self.enable_color()
            
        self.pipeline_profile = self.pipeline.start(self.config)
        self.device = self.pipeline_profile.get_device()
        self.depth_scale = self.device.first_depth_sensor().get_depth_scale() 

    def enable_color_depth(self, color_value = 0):
        """Colorize depth image for vision. Must be called to activate"""

        assert self.depth_enable, 'Depth stream is not enabled'
        self.colorize_enable = 1
        self.colorizer = rs.colorizer(color_value)

    def set_laser_power(self, laser_pwr=50):
        """Adjust laser power. Range ~0-360. Cannot run on rasp pi 3 with USB 2.0"""

        self.laser_pwr = laser_pwr
        depth_sensor = self.device.query_sensors()[0]
        depth_sensor.set_option(rs.option.laser_power, laser_pwr)
        return self.laser_pwr

    def enable_infrared(self):
        """Enable infrared streams. Both left and right sensors"""

        self.config.enable_stream(stream.infrared, 1, self.stream_width, self.stream_height, \
                                  rs.format.y8, self.fps)
        self.config.enable_stream(stream.infrared, 2, self.stream_width, self.stream_height, \
                                  rs.format.y8, self.fps)
        self.infrared_enable = 1

    def enable_depth(self):
        """Enable depth stream. Depth stream is enabled by default"""

        self.config.enable_stream(stream.depth, self.stream_width, self.stream_height, \
                                  rs.format.z16, self.fps)
        self.depth_enable = 1

    def enable_color(self):
        """Enable the color stream. The sensor to the right of the D435i"""

        self.config.enable_stream(stream.color, self.stream_width, self.stream_height, \
                                  rs.format.bgr8, self.fps)
        self.color_enable = 1
        
    def get_frames(self):
        self.frames = self.pipeline.wait_for_frames()

        if (self.depth_enable):
            self.depth_frame = self.frames.get_depth_frame()
            if (self.colorize_enable):
                self.depth_color_frame = self.colorizer.colorize(self.depth_frame)

        if (self.infrared_enable):
            self.infrared_frame_zero = self.frames.get_infrared_frame(1)
            self.infrared_frame_one = self.frames.get_infrared_frame(2)

        if (self.color_enable):
            self.color_frame = self.frames.get_color_frame()

    def run_camera(self):
        """Call in while loop to continuously feed frames"""
        
        self.get_frames()
        if (self.depth_enable):
            self.depth_data = np.asanyarray(self.depth_frame.get_data())
            if (self.colorize_enable):
                self.depth_color_image = np.asanyarray(self.depth_color_frame.get_data())

        if (self.infrared_enable):
            self.infrared_image_zero = np.asanyarray(self.infrared_frame_zero.get_data())
            self.infrared_image_one = np.asanyarray(self.infrared_frame_one.get_data())

        if (self.color_enable):
            self.color_image = np.asanyarray(self.color_frame.get_data())

    def stop_camera(self):
        """Stop the camera before exiting the program"""
        self.pipeline.stop()

    def get_distance(self, x, y):
        """Provide the depth in meters at the given pixel"""

        assert self.depth_enable, 'Enable depth stream to measure distance'
        return self.depth_frame.get_distance(x, y)

    def get_raw_depth(self):
        """Return raw depth values at every pixel. Data type is unsigned 16-bit. Resolution is 640x480"""

        assert self.depth_enable, 'Depth stream is not enabled'
        return self.depth_data

    def get_color_depth(self):
        """Return depth colormap images. Data type is 8-bit rgb. Resolution is 640x480"""

        assert self.depth_enable, 'Depth stream is not enabled'
        assert self.colorize_enable, 'Depth colormap is not enabled'
        return self.depth_color_image

    def get_infrared_image(self):
        """Return infrared images of left and right infrared sensors. Resolution is 640x480"""

        assert self.infrared_enable, 'Infrared streams are not enabled'
        return (self.infrared_image_zero, self.infrared_image_one)

    def get_color_image(self):
        """Return color images. Resolution is 640x480"""

        assert self.color_enable, 'Color stream is not enabled'
        return self.color_image
