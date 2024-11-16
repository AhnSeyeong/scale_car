#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion
import math

# 콜백 함수
def imu_callback(msg):
    # IMU 메시지에서 쿼터니언 추출
    quaternion = (
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z,
        msg.orientation.w
    )

    # 쿼터니언을 RPY(roll, pitch, yaw)로 변환
    roll, pitch, yaw = euler_from_quaternion(quaternion)

    # RPY 출력
    rospy.loginfo("Roll: %.3f, Pitch: %.3f, Yaw: %.3f", math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

def imu_listener():
    # 노드 초기화
    rospy.init_node('imu_listener', anonymous=True)

    # /imu 토픽 구독
    rospy.Subscriber("/imu", Imu, imu_callback)

    # ROS 메시지가 처리될 때까지 대기
    rospy.spin()

if __name__ == '__main__':
    try:
        imu_listener()
    except rospy.ROSInterruptException:
        pass
