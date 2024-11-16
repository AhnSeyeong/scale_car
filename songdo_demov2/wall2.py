#!/usr/bin/env python3

import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray

# ROI 설정
ROI_X_MIN = -1
ROI_X_MAX = 1
ROI_Y_MIN = -1
ROI_Y_MAX = 1

# PID Constants
KP = 3.2
KD = 0.008
KI = 0.001

# Publisher
pub_drive = None
pub_markers = None

# Variables
prev_error = integral = last_time = 0.0

def callback_scan(scan):
    global prev_error, integral, last_time

    # LiDAR 데이터 처리
    ranges = np.array(scan.ranges)
    angles = np.linspace(scan.angle_min + math.pi, scan.angle_max + math.pi, len(ranges))

    # Polar 좌표계를 Cartesian 좌표계로 변환하고, ROI 내의 포인트를 필터링
    points = np.array([[r * math.cos(angle), r * math.sin(angle)]
                      for r, angle in zip(ranges, angles) if r < scan.range_max])

    # 좌우 ROI 필터링: y축을 기준으로 좌우 구분
    left_points = np.array([point for point in points if ROI_Y_MIN <= point[1] <= 0 and ROI_X_MIN <= point[0] <= ROI_X_MAX])
    right_points = np.array([point for point in points if 0 <= point[1] <= ROI_Y_MAX and ROI_X_MIN <= point[0] <= ROI_X_MAX])

    # 마커 시각화
    visualize_markers(left_points, right_points)

    # 센터라인 중심 계산 및 PID 제어
    if len(left_points) > 0 and len(right_points) > 0:
        left_center = np.mean(left_points, axis=0)
        right_center = np.mean(right_points, axis=0)
        center_x = (left_center[0] + right_center[0]) / 2
        center_y = (left_center[1] + right_center[1]) / 2

        # 중앙에서 벗어난 정도를 에러로 계산 (y축 편차 사용)
        error = center_y
        current_time = rospy.Time.now().to_sec()
        delta_time = current_time - last_time if last_time != 0 else 1e-6  # 초기화 방지
        integral += error * delta_time
        derivative = (error - prev_error) / delta_time if delta_time > 0 else 0

        # PID 제어를 통한 조향각 계산
        steering_angle = -(KP * error + KD * derivative + KI * integral)

        # 조향각에 따른 속도 조정
        if abs(steering_angle) > 20.0 * (math.pi / 180.0):  # 조향각이 20도 이상인 경우
            speed = 0.3
        elif abs(steering_angle) > 10.0 * (math.pi / 180.0):  # 조향각이 10도 이상인 경우
            speed = 0.5
        else:  # 직진에 가까운 경우
            speed = 0.7

        # AckermannDrive 메시지 설정 및 퍼블리시
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = -steering_angle
        msg.drive.speed = 0.4
        pub_drive.publish(msg)
        print(msg)

        # 이전 값 업데이트
        prev_error = error
        last_time = current_time

def visualize_markers(left_points, right_points):
    """
    좌측과 우측 ROI 내의 포인트들을 RViz에서 시각화하기 위한 마커 생성.
    """
    marker_array = MarkerArray()

    # 좌측 ROI 마커 (초록색)
    for i, point in enumerate(left_points):
        marker = create_marker(i, point[0], point[1], 0.0, 1.0, 0.0)  # 초록색
        marker_array.markers.append(marker)

    # 우측 ROI 마커 (파란색)
    for i, point in enumerate(right_points, start=len(left_points)):
        marker = create_marker(i, point[0], point[1], 0.0, 0.0, 1.0)  # 파란색
        marker_array.markers.append(marker)

    # 센터라인(center line) 마커 계산 (노란색)
    if len(left_points) > 0 and len(right_points) > 0:
        left_center = np.mean(left_points, axis=0)
        right_center = np.mean(right_points, axis=0)
        center_x = (left_center[0] + right_center[0]) / 2
        center_y = (left_center[1] + right_center[1]) / 2
        center_marker = create_marker(len(left_points) + len(right_points), center_x, center_y, 1.0, 1.0, 0.0)  # 노란색
        marker_array.markers.append(center_marker)

    # 마커 발행
    pub_markers.publish(marker_array)

def create_marker(marker_id, x, y, r, g, b):
    """
    RViz에서 시각화를 위한 단일 마커 생성.
    """
    marker = Marker()
    marker.header.frame_id = "laser"
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

def main():
    global pub_drive, pub_markers, last_time
    rospy.init_node("centerline_follow", anonymous=True)

    # Subscriber와 Publisher 설정
    sub = rospy.Subscriber("/scan", LaserScan, callback_scan)
    pub_drive = rospy.Publisher("/high_level/ackermann_cmd_mux/input/nav_0", AckermannDriveStamped, queue_size=1)
    pub_markers = rospy.Publisher("/roi_markers", MarkerArray, queue_size=1)

    last_time = rospy.Time.now().to_sec()
    rospy.spin()

if __name__ == "__main__":
    main()
