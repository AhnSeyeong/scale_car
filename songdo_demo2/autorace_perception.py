1#!/usr/bin/env python3 
import rospy
import cv2
import math
import numpy as np
from sklearn.cluster import DBSCAN
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from std_msgs.msg import String
from ar_track_alvar_msgs.msg import AlvarMarkers


class perception:
    def __init__(self):
        rospy.init_node('perception')
        self.br = CvBridge()
        
        # Subscribers and Publishers
        self.image_sub = rospy.Subscriber('/usb_cam/image_rect_color/compressed', CompressedImage, self.image_callback)
        self.scan_pub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.ar_sub = rospy.Subscriber('/ar_pose_marker', AlvarMarkers, self.marker_callback)
        self.mission_flag_pub = rospy.Publisher('/mission_flag', String, queue_size=5)
        self.debug_publisher1 = rospy.Publisher('/debug_image1', Image, queue_size=10)
        
        # ROI and Perspective Transformation Parameters
        self.roi_x_l = rospy.get_param('~roi_x_l', 0)
        self.roi_x_h = rospy.get_param('~roi_x_h', 640)
        self.roi_y_l = rospy.get_param('~roi_y_l', 330)
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
            [self.roi_x_l + 200, self.roi_y_h],
            [self.roi_x_h - 200, self.roi_y_h]
        ])
        self.matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        
        # Obstacle and Line Detection Parameters
        self.obstacle_flag = False
        self.line_threshold = 50
        self.kidszone_threshold = 1000
        self.cooldown = 18
        self.last_count_time = 0
        self.mask = None

        # YELLOW HSV RANGE
        self.lower_yellow = np.array([20, 18, 190])
        self.upper_yellow = np.array([55, 255, 255])

        # 첫 번째 빨간색 범위
        self.lower_red1 = np.array([0, 50, 50])
        self.upper_red1 = np.array([15, 255, 255])

        # 두 번째 빨간색 범위
        self.lower_red2 = np.array([160, 50, 50])
        self.upper_red2 = np.array([180, 255, 255])
        
        self.pub_flag = False

        self.kidzone_flag = False
        self.cone_detected = False
        self.tunnel_flag = False
        self.first = True
        

        self.cur_mission = 0
        self.line_cnt = 0

        rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")

    def marker_callback(self, data):
        if data.markers:
            self.current_marker_id = data.markers[0].id
            rospy.loginfo(f"Detected AR Marker ID: {self.current_marker_id}")
            self.pub_flag = True

    def scan_callback(self, data):
        angles = np.linspace(data.angle_min, data.angle_max, len(data.ranges))
        ranges = np.array(data.ranges, dtype=np.float32)

        if self.cur_mission == 2:
            self.cone_check(ranges, angles)
        elif self.cur_mission == 5:
            self.tunnel_check(ranges)
        elif self.cur_mission == 3:
            if rospy.Time.now().to_sec() - self.start_time >= 12.0:
                self.obstacle_check(ranges, angles)

    def cone_check(self, ranges, angles):
        angle_min = -math.radians(150)
        angle_max = math.radians(150)
        valid_angles = (angles <= angle_min) | (angles >= angle_max)
        close_objects = ranges <= 1.0
        filtered_ranges = ranges[valid_angles & close_objects]
        filtered_angles = angles[valid_angles & close_objects]
        
        x = filtered_ranges * np.cos(filtered_angles)
        y = filtered_ranges * np.sin(filtered_angles)
        points = np.column_stack((x, y))
        
        if len(points) < 6:
            return
        db = DBSCAN(eps=0.1, min_samples=2).fit(points)
        labels = db.labels_
        
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue  
            
            cluster_points = points[labels == label]
            cluster_size = len(cluster_points)
            
            if cluster_size > 6:
                distances = np.linalg.norm(cluster_points, axis=1)
                if np.min(distances) <= 0.7:
                    self.cone_detected = True
                    self.pub_flag = True
                    
                    break
    
    def tunnel_check(self, range):
        target_angle_1_rad = np.radians(110)
        target_angle_2_rad = np.radians(-110)

        num_points = len(range)
        angle_increment = 2*np.pi / len(range)
        idx_1 = int((target_angle_1_rad + np.pi) / (angle_increment + 1e-6))
        idx_2 = int((target_angle_2_rad + np.pi) / (angle_increment + 1e-6))

        idx_1 = max(0, min(idx_1, num_points - 1))
        idx_2 = max(0, min(idx_2, num_points - 1))

        distance_1 = range[idx_1]
        distance_2 = range[idx_2]

        if distance_1 > 100 or distance_2 > 100:
            return

        distance_between_points = np.sqrt(
            (distance_1 * np.cos(target_angle_1_rad) - distance_2 * np.cos(target_angle_2_rad))**2 +
            (distance_1 * np.sin(target_angle_1_rad) - distance_2 * np.sin(target_angle_2_rad))**2
        )

        if distance_between_points < 0.6:
            self.tunnel_flag = True
            self.pub_flag = True
        else:
            self.tunnel_flag = False

    def obstacle_check(self, range, angle):
        angle_min = -math.radians(160)
        angle_max = math.radians(160)
        valid_angles = (angle <= angle_min) | (angle >= angle_max)  
        close_objects = range <= 0.8
        desired_ranges = range[valid_angles & close_objects]
        if len(desired_ranges) > 2:
            self.obstacle_flag = True
            self.pub_flag = True
            rospy.loginfo("Crossing Gate detected!")

    def image_callback(self, data):
        self.image_ = self.br.compressed_imgmsg_to_cv2(data, 'bgr8')
        self.warp_ = cv2.warpPerspective(self.image_, self.matrix, (self.image_.shape[1], self.image_.shape[0]))
        self.roi_ = self.warp_[self.roi_y_l:self.roi_y_h, self.roi_x_l:self.roi_x_h]

        if self.cur_mission == 1:
            self.mask = self.check_kidzone(self.roi_)
        elif self.cur_mission == 0 or self.cur_mission == 4 or self.cur_mission == 6 or self.cur_mission == 7:
            self.mask = self.check_line(self.roi_)
        if self.mask is not None:
            self.debug_publisher1.publish(self.br.cv2_to_imgmsg(self.mask, 'mono8'))
    
    def check_line(self, img):
        # Color masking and contour detection
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, self.lower_yellow, self.upper_yellow)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check area of detected contours
        current_time = rospy.get_time()
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.line_threshold and (current_time - self.last_count_time) >= self.cooldown:
                self.line_cnt += 1
                self.last_count_time = current_time
                self.pub_flag = True
                rospy.loginfo("lane detected!")
                break
        return mask

    def check_kidzone(self, img):
        # Color masking and contour detection
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_image, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv_image, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check area of detected contours
        red_detected = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.kidszone_threshold:
                red_detected = True
                break

        # Handle flag transitions
        if red_detected and not self.kidzone_flag:
            self.kidzone_flag = True
            self.pub_flag = True  # Ensure flag triggers a message
            rospy.loginfo("KidZone detected!")
        elif not red_detected and self.kidzone_flag:
            self.kidzone_flag = False
            self.pub_flag = True  # Ensure flag triggers a message
            rospy.loginfo("KidZone ended!")

        return mask
        
    def publish_mission_flag(self):
            if self.cur_mission == 0 and self.line_cnt == 1 and self.pub_flag:
                self.mission_flag_pub.publish('dynamic_obstacle')
                self.pub_flag = False
                self.cur_mission += 1
                rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")
            elif self.cur_mission == 1 and self.pub_flag:
                if self.kidzone_flag:
                    self.mission_flag_pub.publish('kids_zone')
                else:
                    self.mission_flag_pub.publish('kids_zone_end')
                    self.cur_mission += 1
                self.pub_flag = False
                rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")
            elif self.cur_mission == 2 and self.cone_detected and self.pub_flag:
                    self.mission_flag_pub.publish('rubber_cone')
                    rospy.loginfo(f"Rubber Cone!")
                    self.cur_mission += 1
                    self.pub_flag = False
                    self.start_time = rospy.Time.now().to_sec()
                    rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")
            elif self.cur_mission == 3 and  self.obstacle_flag and self.pub_flag:
                self.mission_flag_pub.publish('gate')
                self.obstacle_flag = False
                self.cur_mission += 1
                self.pub_flag = False
                rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")
            elif self.cur_mission == 4 and self.line_cnt == 2 and self.pub_flag:
                self.mission_flag_pub.publish('roundabout')
                self.cur_mission += 1
                self.pub_flag = False
                rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")
            elif self.cur_mission == 5 and self.tunnel_flag and self.pub_flag:
                self.mission_flag_pub.publish('tunnel')
                self.cur_mission += 1
                self.pub_flag = False
                rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")
            elif self.cur_mission == 6 and self.line_cnt == 3 and self.pub_flag:
                if self.first:
                    rospy.loginfo(f"CrossRoad")
                    self.mission_flag_pub.publish('crossroad')
                    self.first = False
                    self.current_marker_id = 4
                    self.marker_check = rospy.Time.now().to_sec()
                    self.cooldown = 2.5
                if self.current_marker_id == 0 and rospy.Time.now().to_sec() - self.marker_check >= 1.0:
                    self.mission_flag_pub.publish('crossroadA')
                    rospy.loginfo(f"CrossRoad A")
                    self.current_marker_id = None
                    self.cur_mission += 1
                    self.pub_flag = False
                    rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")
                elif self.current_marker_id == 4 and rospy.Time.now().to_sec() - self.marker_check >= 1.0:
                    self.mission_flag_pub.publish('crossroadB')
                    rospy.loginfo(f"CrossRoad B")
                    self.cur_mission += 1
                    self.current_marker_id = None
                    self.pub_flag = False
                    rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")
            elif self.cur_mission >= 7 and self.line_cnt >= 3 and self.pub_flag:
                rospy.sleep(0.6)
                self.mission_flag_pub.publish('parking')
                rospy.loginfo(f"Parking")
                self.cur_mission += 1
                self.pub_flag = False
                rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")

                    

if __name__ == "__main__":
    try:
        node = perception()
        rate = rospy.Rate(10)  # Loop rate for flag publishing
        while not rospy.is_shutdown():
            node.publish_mission_flag()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
