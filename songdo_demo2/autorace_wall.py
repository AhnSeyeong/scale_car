#!/usr/bin/env python3

import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray


class CenterlineFollow:
    def __init__(self):
        # ROI 설정
        self.ROI_X_MIN = 0.0
        self.ROI_X_MAX = 0.4
        self.ROI_Y_MIN = -0.4
        self.ROI_Y_MAX = 0.4

        # PID Constants
        self.KP = 3.2
        self.KD = 0.008
        self.KI = 0.001

        # Variables
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = 0.0
        self.scan_data = None

        # ROS Node 초기화
        # Subscriber와 Publisher 설정
        rospy.Subscriber("/scan", LaserScan, self.callback_scan)
        self.pub_drive = rospy.Publisher("/high_level/ackermann_cmd_mux/input/nav_0", AckermannDriveStamped, queue_size=1)
        # self.pub_markers = rospy.Publisher("/roi_markers", MarkerArray, queue_size=1)


    def callback_scan(self, scan):
        """
        스캔 데이터를 저장하는 콜백 함수.
        """
        self.scan_data = scan

    def process_scan_data(self):
        """
        저장된 LiDAR 데이터를 처리하고 PID 제어를 수행.
        """
        # LiDAR 데이터 처리
        ranges = np.array(self.scan_data.ranges)
        angles = np.linspace(self.scan_data.angle_min + math.pi, self.scan_data.angle_max + math.pi, len(ranges))

        # Polar 좌표계를 Cartesian 좌표계로 변환하고, ROI 내의 포인트를 필터링
        points = np.array([[r * math.cos(angle), r * math.sin(angle)]
                          for r, angle in zip(ranges, angles) if r < self.scan_data.range_max])

        # 좌우 ROI 필터링: y축을 기준으로 좌우 구분
        left_points = np.array([point for point in points if self.ROI_Y_MIN <= point[1] <= 0 and self.ROI_X_MIN <= point[0] <= self.ROI_X_MAX])
        right_points = np.array([point for point in points if 0 <= point[1] <= self.ROI_Y_MAX and self.ROI_X_MIN <= point[0] <= self.ROI_X_MAX])

        if len(left_points) == 0 or len(right_points) == 0:
            return "done"
        
        # 마커 시각화
        # self.visualize_markers(left_points, right_points)

        # 센터라인 중심 계산 및 PID 제어
        if len(left_points) > 0 and len(right_points) > 0:
            left_center = np.mean(left_points, axis=0)
            right_center = np.mean(right_points, axis=0)
            center_x = (left_center[0] + right_center[0]) / 2
            center_y = (left_center[1] + right_center[1]) / 2

            # 중앙에서 벗어난 정도를 에러로 계산 (y축 편차 사용)
            error = center_y
            current_time = rospy.Time.now().to_sec()
            delta_time = current_time - self.last_time if self.last_time != 0 else 1e-6  # 초기화 방지
            self.integral += error * delta_time
            derivative = (error - self.prev_error) / delta_time if delta_time > 0 else 0

            # PID 제어를 통한 조향각 계산
            steering_angle = -(self.KP * error + self.KD * derivative + self.KI * self.integral)

            # 조향각에 따른 속도 조정
            if abs(steering_angle) > 20.0 * (math.pi / 180.0):  # 조향각이 20도 이상인 경우
                speed = 0.3
            elif abs(steering_angle) > 10.0 * (math.pi / 180.0):  # 조향각이 10도 이상인 경우
                speed = 0.3
            else:  # 직진에 가까운 경우
                speed = 0.3

        # AckermannDrive 메시지 설정 및 퍼블리시
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = -steering_angle
        msg.drive.speed = 0.35
        self.pub_drive.publish(msg)

        # 이전 값 업데이트
        self.prev_error = error
        self.last_time = current_time

    def visualize_markers(self, left_points, right_points):
        """
        좌측과 우측 ROI 내의 포인트들과 노란색 센터라인(center line) 마커 생성.
        """
        marker_array = MarkerArray()

        # 좌측 포인트 마커 추가 (초록색)
        for i, point in enumerate(left_points):
            left_marker = self.create_marker(i, point[0], point[1], 0.0, 1.0, 0.0)  # 초록색
            left_marker.lifetime = rospy.Duration(0.1)  # 0.1초 동안 표시
            marker_array.markers.append(left_marker)

        # 우측 포인트 마커 추가 (파란색)
        for i, point in enumerate(right_points, start=len(left_points)):
            right_marker = self.create_marker(i, point[0], point[1], 0.0, 0.0, 1.0)  # 파란색
            right_marker.lifetime = rospy.Duration(0.1)  # 0.1초 동안 표시
            marker_array.markers.append(right_marker)

        # 센터라인(center line) 마커 추가 (노란색)
        if len(left_points) > 0 and len(right_points) > 0:
            left_center = np.mean(left_points, axis=0)
            right_center = np.mean(right_points, axis=0)
            center_x = (left_center[0] + right_center[0]) / 2
            center_y = (left_center[1] + right_center[1]) / 2

            center_marker = self.create_marker(len(left_points) + len(right_points), center_x, center_y, 1.0, 1.0, 0.0)  # 노란색
            center_marker.lifetime = rospy.Duration(0.1)  # 0.1초 동안 표시
            marker_array.markers.append(center_marker)

        # MarkerArray 발행 (마커 표시)
        self.pub_markers.publish(marker_array)

    def create_marker(self, marker_id, x, y, r, g, b):
        """
        RViz에서 시각화를 위한 단일 마커 생성.
        """
        marker = Marker()
        marker.header.frame_id = "laser"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # 마커 크기
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = 1.0  # 투명도
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0
        marker.id = marker_id

        return marker

    def run(self):
        """
        주 루프 실행.
        """
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            self.process_scan_data()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("centerline_follow", anonymous=True)
    node = CenterlineFollow()
    node.run()
