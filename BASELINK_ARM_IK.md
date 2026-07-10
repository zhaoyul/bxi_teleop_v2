# Base-Link 双臂 IK 接入说明

这个包用于 `teleop` / `withoutarm.onnx` 模式：身体、腿、腰继续由模型控制，左右手臂由外部输入的 `base_link` 坐标系目标位姿控制。

## 接口

输入目标：

```text
arm_ik/left_target   geometry_msgs/PoseStamped
arm_ik/right_target  geometry_msgs/PoseStamped
```

要求 `header.frame_id` 为 `base_link`。`pose.position` 是目标 TCP 位置，`pose.orientation` 是目标 TCP 姿态。

输出到现有控制链：

```text
pico_control_joint_commands  sensor_msgs/JointState
pico/left_grip               std_msgs/Float32
pico/right_grip              std_msgs/Float32
```

`pico/left_grip` 和 `pico/right_grip` 会发布 `1.0`，用于触发现有 `TeleopState` 接管手臂关节。

## 启动

```bash
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
cd bxi_arm_v2
bash build.sh
source install/setup.bash

ros2 launch bxi_example_py_elf3 example_demo.launch.py
ros2 launch elf3_arm_bringup elf3_arm_bringup_baselink.launch.py
```

如果同时启动了原来的 `elf3_arm_ikpy_control_pico`，会和本节点争抢 `pico_control_joint_commands`。做 base_link 目标点控制时，不要启动 Pico IK 节点。

## 发送测试目标

```bash
ros2 topic pub --once /arm_ik/left_target geometry_msgs/msg/PoseStamped "{
  header: {frame_id: 'base_link'},
  pose: {
    position: {x: 0.25, y: 0.32, z: 0.45},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}"
```

## RViz 可视化

```bash
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
cd /home/parallels/bxi_arm_v2
source install/setup.bash
ros2 launch elf3_arm_ik_baselink baselink_arm_ik_rviz.launch.py
```

另开终端发布目标点，RViz 中可看到手臂姿态变化。

### 复位与握手起手式

从任意位置回到待机零位：

```bash
ros2 service call /arm_ik/go_home std_srvs/srv/Trigger {}
```

从当前位置进入握手起手式：

```bash
ros2 service call /arm_ik/go_handshake_ready std_srvs/srv/Trigger {}
```

完整循环演示“任意位置 -> 复位 -> 任意位置 -> 握手起手式”：

```bash
ros2 run elf3_arm_ik_baselink demo_reset_scenarios
```

默认会反复播放，适合客户评审时放在 RViz 里连续展示。RViz 会同步显示当前步骤和本次姿态/轨迹请求返回耗时，例如：

```text
L4 2/4:Reset-Home calc=1.0ms
L4 4/4:Reset-Hand calc=1.1ms
```

只跑一轮用于测试：

```bash
ros2 run elf3_arm_ik_baselink demo_reset_scenarios --once
```

演示内容：

- 大幅任意姿态 A -> 回零。
- 大幅任意姿态 B -> 左臂归零、右臂进入握手起手式。
- 字幕显示当前循环轮次、阶段和服务触发到轨迹启动返回耗时。

## 当前进展

已完成：

- 基于 ELF3 URDF 的双臂 6D TCP IK 节点。
- 基于 `base_link` 的左右手目标输入接口。
- 输出 `pico_control_joint_commands`，可对接现有手臂命令链。
- 回零服务 `/arm_ik/go_home`。
- 握手起手式服务 `/arm_ik/go_handshake_ready`：左臂回零，右臂伸出。
- 大幅演示姿态服务 `/arm_ik/go_demo_pose_a`、`/arm_ik/go_demo_pose_b`。
- RViz 展示、机器人模型、同步字幕、循环演示和耗时显示。

待真实联调：

- 接入 `withoutarm.onnx` state：身体、腿、腰由模型控制，手臂由本 IK 节点覆盖。
- 在真实控制链路中确认最终关节命令合并点。
- 将当前 demo 层的服务返回耗时细分为纯 IK 求解耗时、轨迹生成耗时和控制链路延迟。

## 当前保护

- 目标必须在配置工作空间内。
- 左右 TCP 目标距离过近时保持上一帧命令。
- 每周期关节变化有限幅。
- 关节命令做一阶低通平滑。
- 目标超时后保持上一帧命令。
