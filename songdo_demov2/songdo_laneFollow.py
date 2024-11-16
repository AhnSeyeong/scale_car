#!/usr/bin/env python3

import rospy
import cv2
import math
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from ackermann_msgs.msg import AckermannDriveStamped

class DetectLine:
    def __init__(self):
        self.br = CvBridge()
        self.image_sub = rospy.Subscriber('/usb_cam/image_rect_color/compressed', CompressedImage, self.image_callback)
        self.ack_pub = rospy.Publisher('/high_level/ackermann_cmd_mux/input/nav_0', AckermannDriveStamped, queue_size=10)
        self.debug_publisher1 = rospy.Publisher('debugging_image1', Image, queue_size=10)
        self.debug_publisher2 = rospy.Publisher('debugging_image2', Image, queue_size=10)

        self.roi_x_l = rospy.get_param('~roi_x_l', 0)
        self.roi_x_h = rospy.get_param('~roi_x_h', 640)
        self.roi_y_l = rospy.get_param('~roi_y_l', 300)
        self.roi_y_h = rospy.get_param('~roi_y_h', 480)

        self.guide_point_distance_x = rospy.get_param('~guide_point_distance_x', 0.3)
        #Color masking parameter
        self.white_lane_low = np.array([0, 0, 228])  #0, 0, 170
        self.white_lane_high = np.array([180, 50, 252])    #180, 45, 255

        self.current_center = 320   

        self.clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4,4))
        self.kernel = np.ones((3, 3), np.uint8)  # Morpology operator parameter
   
        self.debug_sequence = rospy.get_param('~debug_image_num', 1)  # Parameter for debugging (0: ROI, 1: Masking, 2: Whole image)
        self.min_contour_area = rospy.get_param('~min_contour_area', 150)  # Minimum area threshold of detecting contour
        self.max_contour_area = rospy.get_param('~max_contour_area', 8000)  # Minimum area threshold of detecting contour
        self.min_distance_threshold = rospy.get_param('~min_distance_threshold', 95)  # Minimum distance threshold between contour points
        self.line_distance_threshold = rospy.get_param('~line_distance_threshold', 250)  # Distance threshold between two closest lines

        self.src_points = np.float32([
            [self.roi_x_l, self.roi_y_l],  # Top-left corner
            [self.roi_x_h, self.roi_y_l],  # Top-right corner
            [self.roi_x_l, self.roi_y_h],  # Bottom-left corner
            [self.roi_x_h, self.roi_y_h]   # Bottom-right corner
        ])

        self.dst_points = np.float32([
            [self.roi_x_l  , self.roi_y_l],     # Top-left corner
            [self.roi_x_h  , self.roi_y_l],     # Top-right corner
            [self.roi_x_l + 210, self.roi_y_h], # Bottom-left corner
            [self.roi_x_h - 210, self.roi_y_h]  # Bottom-right corner
        ])

        self.brightness_avg = None
        self.alpha = 0.0001
        self.matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.inv_matrix = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
        self.img_ = None


    def image_callback(self, msg):
        self.image_ = self.br.compressed_imgmsg_to_cv2(msg, 'bgr8')  # img size: 480 * 640     

    def calculate_guiding_position(self, guide_center, deaccel_flag):
        dy = guide_center - 320
        dx = self.roi_.shape[0] + 10
        theta_rad = np.arctan2(dy,dx)
        theta = math.degrees(np.arctan2(dy,dx))
        print(f"steer : {round(theta,2)}")
                
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.drive.steering_angle = -theta_rad 
        if deaccel_flag :
            msg.drive.speed = 0.28
        else:
            msg.drive.speed = 0.4
        
        self.ack_pub.publish(msg)

    def run(self, deaccel_flag):
        self.warp_ = cv2.warpPerspective(self.image_, self.matrix, (self.image_.shape[1], self.image_.shape[0]))

        self.roi_ = self.warp_[self.roi_y_l:self.roi_y_h, self.roi_x_l:self.roi_x_h]

        self.roi_ = cv2.GaussianBlur(self.roi_, (5, 5), 0)

        hsv = cv2.cvtColor(self.roi_, cv2.COLOR_BGR2HSV)

        # h, s, v = cv2.split(hsv)
        # cur_avg_brightness = np.mean(v)
        # # v_clahe = self.clahe.apply(v)   

        # if self.brightness_avg is None:
        #     self.brightness_avg = cur_avg_brightness
        # else:
        #     self.brightness_avg = self.alpha * cur_avg_brightness + (1 - self.alpha) * self.brightness_avg
        
        # adjustment_factor = self.brightness_avg / cur_avg_brightness  

        # v = cv2.convertScaleAbs(v, alpha=adjustment_factor)

        # hsv_clahe = cv2.merge((h, s, v))

        self.mask_white = cv2.inRange(hsv, self.white_lane_low, self.white_lane_high)

        self.mask_white = cv2.morphologyEx(self.mask_white, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(self.mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        #lines initialize 
        left_lines = [] 
        right_lines = []
        stop_lines = []
        img_center = np.array([self.roi_.shape[1] // 2, self.roi_.shape[0] // 2])  # x , y

        print("-----------------")

        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_contour_area or area > self.max_contour_area:
                continue  
            
            # print(area)
            
            # Approximate contour to a polygon
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx_1 = cv2.approxPolyDP(contour, epsilon, True)
            approx_1 = np.squeeze(approx_1)
            approx_2 = np.roll(approx_1, shift=1, axis=0)
            distance = np.linalg.norm(approx_1 - approx_2, axis=1)

            print(f"distance : {distance}")

            if np.max(distance) < self.min_distance_threshold:
                continue  

            [vx, vy, x0, y0] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            
            slope = vy/ vx
            if abs(slope) < 0.3:
                stop_lines.append([x0,y0,slope])
                continue
            
            topy = 0  # Where the line crosses the top edge (y=0)
            bottomy = self.roi_.shape[0]  # Where the line crosses the bottom edge (y=height)

            # Calculate the corresponding x-values for y=0 (top) and y=height (bottom)
            topx = int((topy - y0) * vx / vy + x0)  # Top edge
            bottomx = int((bottomy - y0) * vx / vy + x0)  # Bottom edge
            centerx = (topx + bottomx) // 2


            cv2.drawContours(self.roi_, [approx_1], 0, (0, 255, 0), 3)
            for point in approx_1:
                x, y = point
                cv2.circle(self.roi_, (x, y), 5, (255, 0, 0), -1)
            cv2.line(self.roi_, (topx, topy), (bottomx, bottomy), (0, 0, 255), 4)

            if bottomx < img_center[0]:
                left_lines.append(centerx)
            else:
                right_lines.append(centerx)
        
        left_lines.sort()
        right_lines.sort()
        # print(left_lines)
        # print(right_lines)
        print(f"stop_lines : {stop_lines}")
        
        guide_center = []
        idx_counter = 0
        current_idx = 0
        # Seperate Lane 
        if len(left_lines) + len(right_lines) >= 2:

            if len(left_lines) > 1:
                for i in range(len(left_lines)-1):
                    diff = left_lines[i+1] - left_lines[i]
                    print(f"left center : {diff}")
                    if diff > self.line_distance_threshold:
                        print("left center add!")
                        guide_center.append((left_lines[i+1] + left_lines[i])//2)
                        idx_counter += 1

            if len(left_lines) > 0 and len(right_lines) > 0:
                diff = right_lines[0] - left_lines[-1]
                if diff > self.line_distance_threshold:
                    print("middle center add!")
                    current_idx = idx_counter
                    guide_center.append((left_lines[-1] + right_lines[0]) // 2) 
                    idx_counter += 1  

            if len(right_lines) > 1:
                for j in range(len(right_lines)-1):
                    diff = right_lines[j+1] - right_lines[j]
                    print(f"right center : {diff}")
                    if diff > self.line_distance_threshold:
                        print("right center add!")
                        guide_center.append((right_lines[-1] + right_lines[0]) // 2) 
                        idx_counter += 1

        elif len(left_lines) + len(right_lines) == 1:

            if len(left_lines) > 0 :
                guide_center.append(left_lines[0] + 170)
                idx_counter += 1

            if len(right_lines) > 0 :
                guide_center.append(right_lines[0] - 170)
                idx_counter += 1

        print(f"guide center : {guide_center}")
        print(f"current index : {current_idx}")
        if idx_counter != 0:
            self.current_center = guide_center[current_idx]
            print(f"current center : {self.current_center}")
            self.calculate_guiding_position(self.current_center, deaccel_flag)
        else:
            self.calculate_guiding_position(self.current_center, deaccel_flag)

        for center in guide_center:
            if center - self.current_center == 0:
                cv2.circle(self.roi_, (center, self.roi_.shape[0] // 2), 10, (0, 255, 255), -1)  
            elif center - self.current_center > 0:
                cv2.circle(self.roi_, (center, self.roi_.shape[0] // 2), 10, (0, 255, 0), -1)  
            elif center - self.current_center < 0:
                cv2.circle(self.roi_, (center, self.roi_.shape[0] // 2), 10, (0, 0, 255), -1) 
        
        self.warp_[self.roi_y_l:self.roi_y_h, self.roi_x_l:self.roi_x_h] = self.roi_
        self.warp_ = cv2.warpPerspective(self.warp_, self.inv_matrix, (self.image_.shape[1], self.image_.shape[0]))
        self.image_[self.roi_y_l:self.roi_y_h, self.roi_x_l:self.roi_x_h] = self.warp_[self.roi_y_l:self.roi_y_h, self.roi_x_l:self.roi_x_h]

        # Debug image publishing
        if self.debug_sequence == 0:
            self.debug_publisher.publish(self.br.cv2_to_imgmsg(self.roi_, 'bgr8'))
        elif self.debug_sequence == 1:
            self.debug_publisher1.publish(self.br.cv2_to_imgmsg(self.mask_white, 'mono8'))
            self.debug_publisher2.publish(self.br.cv2_to_imgmsg(self.roi_, 'bgr8'))
        else:
            self.debug_publisher.publish(self.br.cv2_to_imgmsg(self.image_, 'bgr8'))      

if __name__ == '__main__':
    try:
        rospy.init_node('detect_line')
        detect_line = DetectLine()
        detect_line.run()

    except rospy.ROSInterruptException:
        pass
