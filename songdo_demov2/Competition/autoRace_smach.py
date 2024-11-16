#!/usr/bin/env python3
import rospy
import smach
import smach_ros
import songdo_laneFollow
import math
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan


class MissionStart(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['start_mission'])
        
    def execute(self, userdata):
        rospy.loginfo('State: MISSION START')
        rospy.sleep(1)  # Wait for 1 second
        return 'start_mission'

class LaneFollow(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['static_obstacle', 'dynamic_obstacle', 
                                             'rubber_cone', 'kids_zone', 'mission_end'])
        self.mission_flag_sub = rospy.Subscriber('/mission_flag', String, self.flag_callback)
        self.mission_flag = None  # Default mission flag
        self.line_follow = songdo_laneFollow.DetectLine()
        self.rate = rospy.Rate(20)
        self.deaccel_flag = False

    def flag_callback(self, msg):
        if msg is not None:
            self.mission_flag = msg.data
        else:
            self.mission_flag = None

    def execute(self, userdata):
        rospy.loginfo('State: LANE FOLLOW')
        while not rospy.is_shutdown():
            if self.mission_flag == 'static_obstacle':
                self.mission_flag = None
                return 'static_obstacle'
            elif self.mission_flag == 'dynamic_obstacle':
                self.mission_flag = None
                return 'dynamic_obstacle'
            elif self.mission_flag == 'rubber_cone':
                self.mission_flag = None
                return 'rubber_cone'
            elif self.mission_flag == 'kids_zone':
                self.mission_flag = None
                self.deaccel_flag = True
                return 'kids_zone'
            elif self.mission_flag == 'kids_zone_end':
                self.mission_flag = None
                self.deaccel_flag = False
            elif self.mission_flag == 'mission_end':
                self.mission_flag = None
                return 'mission_end'
            else:
                self.line_follow.run(self.deaccel_flag)

                self.rate.sleep()

class StaticObstacleAvoid(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['obstacle_cleared'])
        self.desired_ranges = []  # 초기값 설정
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.angle_min = -math.radians(173)
        self.angle_max = math.radians(173)
        self.data_updated = False  # 데이터 갱신 플래그

    def scan_callback(self, data):
        ranges = data.ranges
        angles = [data.angle_min + i * data.angle_increment for i in range(len(ranges))]

        # 특정 조건을 만족하는 거리 값들을 필터링하여 저장
        self.desired_ranges = [
            ranges[i] for i, angle in enumerate(angles)
            if (angle <= self.angle_min or angle >= self.angle_max) and ranges[i] <= 1.0
        ]
        self.data_updated = True  # 데이터가 갱신되었음을 표시

    def execute(self, userdata):
        rospy.loginfo('State: STATIC OBSTACLE AVOID')
        rate = rospy.Rate(10)  # 0.1초마다 조건 확인

        while not rospy.is_shutdown():
            if self.data_updated:  # 데이터가 갱신되었을 때만 길이를 확인
                print(len(self.desired_ranges))
                if len(self.desired_ranges) <= 2:
                    return 'obstacle_cleared'
                self.data_updated = False  # 데이터 확인 후 플래그 초기화
            rate.sleep()

class DynamicObstacleAvoid(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['obstacle_cleared'])

    def execute(self, userdata):
        rospy.loginfo('State: DYNAMIC OBSTACLE AVOID')
        rospy.sleep(3.5)  # Simulate dynamic obstacle avoidance
        return 'obstacle_cleared'

class RubberConeDriving(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['mission_done'])

    def execute(self, userdata):
        rospy.loginfo('State: RUBBER CONE DRIVING')
        rospy.sleep(10)  # Simulate rubber cone driving
        return 'mission_done'

class KidsZoneDriving(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['mission_done'])
        self.desired_ranges = []
        self.data_updated = False  # 데이터 갱신 플래그
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
    
    def scan_callback(self, data):
        ranges = data.ranges
        angles = [data.angle_min + i * data.angle_increment for i in range(len(ranges))]
        angle_min = -math.radians(90)
        angle_max = math.radians(175)
        
        # 조건에 맞는 거리 데이터 필터링 및 업데이트
        self.desired_ranges = [
            ranges[i] for i, angle in enumerate(angles)
            if (angle <= angle_min or angle >= angle_max) and ranges[i] <= 1.5
        ]
        self.data_updated = True  # 데이터 갱신 표시

    def execute(self, userdata):
        rospy.loginfo('State: KIDS ZONE DRIVING')
        rate = rospy.Rate(2)  # 0.5초마다 조건 확인

        # 데이터가 수신될 때까지 대기
        while not rospy.is_shutdown():
            if self.data_updated:  # 데이터가 업데이트되었을 때만 확인
                rospy.loginfo(f'Desired ranges count: {len(self.desired_ranges)}')  # 디버깅 출력
                if len(self.desired_ranges) <= 2:
                    return 'mission_done'
                self.data_updated = False  # 플래그 초기화
            rate.sleep()

class MissionEnd(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['terminate'])

    def execute(self, userdata):
        rospy.loginfo('State: MISSION END')
        rospy.sleep(10)  # Simulate end process
        return 'terminate'

def main():
    rospy.init_node('autonomous_vehicle_state_machine')

    # Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['MISSION_TERMINATED'])

    # Open the container
    with sm:
        # Add states to the state machine
        smach.StateMachine.add('MISSION_START', MissionStart(), transitions={'start_mission': 'LANE_FOLLOW'})
        smach.StateMachine.add('LANE_FOLLOW', LaneFollow(), transitions={'static_obstacle': 'STATIC_OBSTACLE_AVOID',
                                                                         'dynamic_obstacle': 'DYNAMIC_OBSTACLE_AVOID',
                                                                         'rubber_cone': 'RUBBER_CONE_DRIVING',
                                                                         'kids_zone': 'KIDS_ZONE_DRIVING',
                                                                         'tunnel': 'TUNNEL',
                                                                         'gate': 'CROSSING_GATE'
                                                                         'crossroad': 'CROSSROAD_DRIVING'
                                                                         'parking': 'PARKING'})
        smach.StateMachine.add('RUBBER_CONE_DRIVING', RubberConeDriving(), transitions={'mission_done': 'LANE_FOLLOW'})
        smach.StateMachine.add('KIDS_ZONE_DRIVING', KidsZoneDriving(), transitions={'mission_done': 'LANE_FOLLOW'})
        smach.StateMachine.add('STATIC_OBSTACLE_AVOID', StaticObstacleAvoid(), transitions={'obstacle_cleared': 'LANE_FOLLOW'})
        smach.StateMachine.add('DYNAMIC_OBSTACLE_AVOID', DynamicObstacleAvoid(), transitions={'obstacle_cleared': 'LANE_FOLLOW'})
        smach.StateMachine.add('CROSSING_GATE', DynamicObstacleAvoid(), transitions={'obstacle_cleared': 'LANE_FOLLOW'})
        smach.StateMachine.add('TUNNEL_DRIVING', KidsZoneDriving(), transitions={'mission_done': 'LANE_FOLLOW'})
        smach.StateMachine.add('CROSSROAD_DRIVING', KidsZoneDriving(), transitions={'mission_done': 'LANE_FOLLOW'})
        smach.StateMachine.add('PARKING', KidsZoneDriving(), transitions={'mission_done': 'MISSION_END'})
        smach.StateMachine.add('MISSION_END', MissionEnd(), transitions={'terminate': 'MISSION_TERMINATED'})

    # Create and start the introspection server
    sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
    sis.start()

    # Execute the state machine
    outcome = sm.execute()

    # Wait for ctrl-c to stop the application
    rospy.spin()
    sis.stop()

if __name__ == '__main__':
    main()
