#!/usr/bin/env python3

import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped
from sklearn.cluster import DBSCAN
from std_msgs.msg import String

class ConeFollower:
    def __init__(self):
        rospy.init_node('cone_follower', anonymous=True)
        
        # Constants
        self.ROI = {'x_min': -2.0, 'x_max': 0.5, 'y_min': -0.5, 'y_max': 0.5}
        self.CONTROL_RATE = 50  # Hz
        self.BASE_SPEED = 1.0
        self.WHEELBASE = 0.325
        self.LOOK_AHEAD = 0.7
        self.MAX_STEER = np.radians(30)
        
        # Path related variables
        self.path = None
        self.scan_data = None
        self.path_offset = None  # 콘 위치에 따른 오프셋 저장
        
        # Publishers
        self.drive_pub = rospy.Publisher('/high_level/ackermann_cmd_mux/input/nav_0', 
                                       AckermannDriveStamped, queue_size=1)
        self.viz_pub = rospy.Publisher('/visualization_markers', 
                                     MarkerArray, queue_size=1)
        self.status_pub = rospy.Publisher('/mission_status', 
                                        String, queue_size=1)
        
        # Subscriber
        rospy.Subscriber("/scan", LaserScan, self.scan_callback, queue_size=1)
        
        # Main control loop
        rospy.Timer(rospy.Duration(1.0/self.CONTROL_RATE), self.control_loop)
        
        # Mission monitoring
        self.no_cone_count = 0
        self.end_threshold = 10000
        
    def scan_callback(self, scan):
        self.scan_data = scan
            
    def get_points_in_roi(self, scan):
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        ranges = np.array(scan.ranges, dtype=np.float32)
        
        # 무효한 값 처리
        mask = np.isfinite(ranges) & (ranges < scan.range_max)
        valid_ranges = ranges[mask]
        valid_angles = angles[mask]
        
        if len(valid_ranges) == 0:
            return np.array([])
        
        # 좌표 변환
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        
        # ROI 필터링
        roi_mask = ((x >= self.ROI['x_min']) & 
                    (x <= self.ROI['x_max']) & 
                    (y >= self.ROI['y_min']) & 
                    (y <= self.ROI['y_max']))
        
        return np.column_stack((x[roi_mask], y[roi_mask]))
            
    def detect_cones(self, points):
        if len(points) < 3:
            return None, None
            
        # 첫 번째 DBSCAN - 개별 콘 감지
        db = DBSCAN(eps=0.1, min_samples=2).fit(points)
        labels = db.labels_
        
        centroids = []
        for label in set(labels) - {-1}:
            cluster = points[labels == label]
            centroids.append(np.mean(cluster, axis=0))
            
        if not centroids:
            return None, None
            
        centroids = np.array(centroids)
        
        # 두 번째 DBSCAN - 좌/우 구분
        db_lr = DBSCAN(eps=0.52, min_samples=2).fit(centroids)
        labels_lr = db_lr.labels_
        
        # 가장 큰 클러스터 찾기
        unique_labels = set(labels_lr) - {-1}
        if not unique_labels:
            return None, None
            
        cluster_sizes = {label: np.sum(labels_lr == label) for label in unique_labels}
        largest_label = max(cluster_sizes, key=cluster_sizes.get)
        largest_cluster = centroids[labels_lr == largest_label]
        
        # 콘이 로봇 기준 왼쪽/오른쪽인지 판단
        closest_cone_idx = np.argmin(np.linalg.norm(largest_cluster, axis=1))
        is_right = largest_cluster[closest_cone_idx][1] > 0
        
        return largest_cluster, is_right
        
    def generate_path(self, cones, is_right):
        if len(cones) < 2:
            return None
            
        # x좌표 기준 정렬
        sorted_cones = cones[np.argsort(cones[:, 0])]
        
        # 오프셋 거리 계산 (콘 사이 거리의 60%)
        dist = np.linalg.norm(sorted_cones[1] - sorted_cones[0])
        offset_dist = 0.4    
        
        # 경로 포인트 생성
        path_points = []
        for i in range(len(sorted_cones) - 1):
            current = sorted_cones[i]
            next_cone = sorted_cones[i + 1]
            
            # 방향 벡터와 수직 벡터 계산
            direction = next_cone - current
            direction = direction / np.linalg.norm(direction)
            normal = np.array([-direction[1], direction[0]])
            
            if is_right:
                normal = -normal
                
            # 오프셋 적용
            offset_point = current + normal * offset_dist
            path_points.append(offset_point)
            
        # 마지막 포인트 추가
        path_points.append(sorted_cones[-1] + normal * offset_dist)
        
        path = np.array(path_points)
        
        # 스무딩
        smoothed = path.copy()
        for _ in range(3):
            smoothed[1:-1] = 0.5 * smoothed[1:-1] + 0.25 * (smoothed[:-2] + smoothed[2:])
            
        return smoothed
        
    def calculate_curvature(self, path):
        curvature = np.zeros(len(path))
        for i in range(1, len(path) - 1):
            prev, curr, next_pt = path[i-1], path[i], path[i+1]
            dx1, dy1 = curr - prev
            dx2, dy2 = next_pt - curr
            angle_diff = np.arctan2(dy2, dx2) - np.arctan2(dy1, dx1)
            curvature[i] = abs(angle_diff)
        return curvature
        

    def pure_pursuit_control(self, path):
        if len(path) < 2:
            return 0, 0
            
        curvature = self.calculate_curvature(path)
        current_curvature = np.mean(curvature[:3])  # 현재 위치 근처의 곡률
        
        # 곡률이 클수록 Look-ahead distance를 줄임
        dynamic_lookahead = self.LOOK_AHEAD * (1 / (1 + current_curvature * 2))
        dynamic_lookahead = np.clip(dynamic_lookahead, 0.5, self.LOOK_AHEAD)
        
        # 목표점 선택
        distances = np.linalg.norm(path, axis=1)
        target_idx = np.argmin(np.abs(distances - dynamic_lookahead))
        target = path[target_idx]
        
        # 제어점 시각화 추가
        self.visualize_target(target)
        
        # 조향각 계산
        alpha = math.atan2(target[1], target[0])
        steering = math.atan2(2 * self.WHEELBASE * math.sin(alpha), self.LOOK_AHEAD)
        steering = np.clip(steering, -self.MAX_STEER, self.MAX_STEER)
        
        # 곡률 기반 속도 제어
        curvature = self.calculate_curvature(path)
        current_curvature = curvature[target_idx]
        max_curvature = np.max(curvature)
        
        curvature_factor = 1.0 - (current_curvature / (max_curvature + 1e-6)) * 0.7
        steering_factor = 1.0 - (abs(steering) / self.MAX_STEER) * 0.5
        speed = self.BASE_SPEED * min(curvature_factor, steering_factor)
        speed = np.clip(speed, 1.0, self.BASE_SPEED)
        
        return -steering, speed
        
    def control_loop(self, event):
        if self.scan_data is None:
            return
            
        points = self.get_points_in_roi(self.scan_data)
        cones, is_right = self.detect_cones(points)
        
        if cones is None:
            self.no_cone_count += 1
            if self.no_cone_count >= self.end_threshold:
                self.end_mission()
            return
            
        self.no_cone_count = 0
        
        path = self.generate_path(cones, is_right)
        if path is not None:
            steering, speed = self.pure_pursuit_control(path)
            
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = rospy.Time.now()
            drive_msg.drive.steering_angle = steering
            drive_msg.drive.speed = 0.5
            self.drive_pub.publish(drive_msg)
            
            self.visualize(cones, path)
            
    def visualize(self, cones, path):
        marker_array = MarkerArray()
        
        # 콘 마커
        cone_marker = Marker()
        cone_marker.header.frame_id = "laser"
        cone_marker.ns = "cones"
        cone_marker.type = Marker.POINTS
        cone_marker.scale.x = cone_marker.scale.y = 0.1
        cone_marker.color.r = 1.0
        cone_marker.color.a = 1.0
        
        for cone in cones:
            p = Point()
            p.x, p.y = cone
            cone_marker.points.append(p)
        
        marker_array.markers.append(cone_marker)
        
        # 제어점 마커
        control_marker = Marker()
        control_marker.header.frame_id = "laser"
        control_marker.ns = "control_points"
        control_marker.type = Marker.POINTS 
        control_marker.scale.x = control_marker.scale.y = 0.15 # 제어점은 조금 더 크게
        control_marker.color.g = 1.0
        control_marker.color.a = 1.0
        
        for point in path:
            p = Point()
            p.x, p.y = point
            control_marker.points.append(p)
            
        marker_array.markers.append(control_marker)
        
        # ID 할당
        for id, marker in enumerate(marker_array.markers):
            marker.id = id
            marker.header.stamp = rospy.Time.now()
            marker.lifetime = rospy.Duration(1.0/self.CONTROL_RATE)
        
        self.viz_pub.publish(marker_array)
        
    def visualize_target(self, target):
        marker = Marker()
        marker.header.frame_id = "laser"
        marker.ns = "target_point"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = target[0]
        marker.pose.position.y = target[1]
        marker.scale.x = marker.scale.y = marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.header.stamp = rospy.Time.now()
        marker.lifetime = rospy.Duration(1.0/self.CONTROL_RATE)
        
        self.target_pub = rospy.Publisher('/target_marker', Marker, queue_size=1)  # 클래스 생성자에 추가
        self.target_pub.publish(marker)
 
            
    def end_mission(self):
        rospy.loginfo("Mission complete - No cones detected")
        self.status_pub.publish(String("rubbercone_stop"))
        rospy.signal_shutdown("Mission complete")

def main():
    try:
        follower = ConeFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
