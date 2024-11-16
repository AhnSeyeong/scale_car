#!/usr/bin/env python3
import rospy
import cv2
import math
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from std_msgs.msg import String

class perception:
    def __init__(self):
        rospy.init_node('perception')
        self.br = CvBridge()
        
        # Subscribers and Publishers
        self.image_sub = rospy.Subscriber('/usb_cam/image_rect_color/compressed', CompressedImage, self.image_callback)
        self.scan_pub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.mission_flag_pub = rospy.Publisher('/mission_flag', String, queue_size=5)
        self.debug_publisher1 = rospy.Publisher('/debug_image1', Image, queue_size=10)
        
        # ROI and Perspective Transformation Parameters
        self.roi_x_l = rospy.get_param('~roi_x_l', 0)
        self.roi_x_h = rospy.get_param('~roi_x_h', 640)
        self.roi_y_l = rospy.get_param('~roi_y_l', 300)
        self.roi_y_h = rospy.get_param('~roi_y_h', 480)

        self.src_points = np.float32([
            [self.roi_x_l, self.roi_y_l],
            [self.roi_x_h, self.roi_y_l],
            [self.roi_x_l, self.roi_y_h],
            [self.roi_x_h, self.roi_y_h]
        ])
        self.dst_points = np.float32([
            [self.roi_x_l, self.roi_y_l],
            [self.roi_x_h, self.roi_y_l],
            [self.roi_x_l + 210, self.roi_y_h],
            [self.roi_x_h - 210, self.roi_y_h]
        ])
        self.matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        
        # Obstacle and Line Detection Parameters
        self.obstacle_flag = False
        self.area_threshold = 350
        self.cooldown = 10
        self.last_count_time = 0
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])

        self.lower_red = np.array([0, 100, 100])
        self.upper_red = np.array([10, 255, 255])
        self.kidzone_flag = False
        self.cur_kidzone  = False

        self.line_cnt = 0
        self.pub_flag = False

    def scan_callback(self, data):
        ranges = data.ranges
        angles = [data.angle_min + i * data.angle_increment for i in range(len(ranges))]
        angle_min = -math.radians(176)
        angle_max = math.radians(176)
        desired_ranges = []
        for i, angle in enumerate(angles):
            if angle <= angle_min or angle >= angle_max:
                if ranges[i] <= 1.0:
                    desired_ranges.append(ranges[i])

        # rospy.loginfo(f"{desired_ranges}")            

        # Set obstacle flag if more than 2 obstacles are detected
        self.obstacle_flag = len(desired_ranges) > 2


    def image_callback(self, data):
        # Image processing
        self.image_ = self.br.compressed_imgmsg_to_cv2(data, 'bgr8')
        self.warp_ = cv2.warpPerspective(self.image_, self.matrix, (self.image_.shape[1], self.image_.shape[0]))
        self.roi_ = self.warp_[self.roi_y_l:self.roi_y_h, self.roi_x_l:self.roi_x_h]
        
        # Check for yellow lane and publish debug image
        self.check_line(self.roi_)
        if self.line_cnt == 1:
            self.check_kidzone(self.roi_)
        self.debug_publisher1.publish(self.br.cv2_to_imgmsg(self.roi_, 'bgr8'))

    def check_line(self, img):
        # Color masking and contour detection
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, self.lower_yellow, self.upper_yellow)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check area of detected contours
        current_time = rospy.get_time()
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.area_threshold and (current_time - self.last_count_time) >= self.cooldown:
                self.line_cnt += 1
                self.last_count_time = current_time
                self.pub_flag = True
                rospy.loginfo("Yellow lane detected!")
                break

    def check_kidzone(self, img):
        # Color masking and contour detection
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, self.lower_red, self.upper_red)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check area of detected contours
        current_time = rospy.get_time()
	
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.area_threshold:
                self.kidzone_flag = True
                self.last_count_time = current_time
                rospy.loginfo("KidZone detected!")
                break
        
   

    def publish_mission_flag(self):
        # Main flag publishing logic
        rospy.loginfo(f"{self.obstacle_flag} {self.line_cnt} {self.kidzone_flag}")
        if self.obstacle_flag:
            self.mission_flag_pub.publish('static_obstacle')
            self.obstacle_flag = False
        else:
            if self.line_cnt == 1 and self.pub_flag:
                self.mission_flag_pub.publish('dynamic_obstacle')
                self.pub_flag = False
            elif self.line_cnt == 1 and self.kidzone_flag:
                self.mission_flag_pub.publish('kid_zone')
                self.cur_kidzone = True
            elif self.line_cnt == 2 and self.pub_flag:
                self.mission_flag_pub.publish('roundabout')
                self.pub_flag = False
            elif self.line_cnt == 3 and self.pub_flag:
                self.mission_flag_pub.publish('roundabout_end')
                self.cooldown = 3
                self.pub_flag = False
            elif self.line_cnt == 4 and self.pub_flag:
                self.mission_flag_pub.publish('mission_end')
                self.pub_flag = False

            if self.kidzone_flag == False and self.cur_kidzone:
                self.mission_flag_pub.publish('kids_zone_end')

if __name__ == "__main__":
    try:
        node = perception()
        rate = rospy.Rate(15)  # Loop rate for flag publishing
        while not rospy.is_shutdown():
            node.publish_mission_flag()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
