## 이 코드는 오직 데모 시연만을 위해 작성된 코드입니다.

이 코드들은 모두 WeGo Robotics사의 제품인 Webot2.0 환경에서 개발이 되었습니다.

![Webot2.0](https://github.com/AhnSeyeong/scale_car/blob/main/contents/webot.png)


따라서 이 코드들을 다른 플랫폼에서 적용하려면 플랫폼에 장착된 센서에 맞춰 해당 코드의 알고리즘을 사용하여야 합니다.

이 코드에서는 두 개의 autorace_smach.py 파일과 autorace_perception.py 파일로 동작합니다. 나머지 코드들은 autorace_smach.py의 모듈로써 동작합니다.

### +songdo_laneFollow.py

이 코드는 차선 추종을 위한 코드입니다. 카메라 상에서 양 차선을 인식해서 차선의 중앙 지점을 추종하도록 합니다.

차선이 하나만 잡히는 경우에도 좌,우측을 구분하여 중앙점을 추종하도록 합니다.


### +autorace_wall.py

이 코드는 터널 주행을 위한 코드입니다. 터널 내부에서 양쪽 벽간의 거리가 동일하도록 PID 제어를 이용하여 주행합니다.


### +autorace_cone.py

이 코드는 라바콘 코스 주행을 위한 코드입니다. 좌측과 우측 라바콘을 클러스터링하여 분류하고 좌, 우측 중 라바콘이 많은 곳과 일정한 offset을 유지하도록 하여 주행합니다.

급격한 코너나 좌우 라바콘 간의 간격이 불규칙하여도 동작하도록 설계되었습니다.
