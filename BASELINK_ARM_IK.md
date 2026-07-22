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

结构化求解状态：

```text
arm_ik/status  std_msgs/String (JSON)
```

状态包含成功标志、失败原因、位置误差、姿态误差、纯 IK 耗时、校验耗时、
雅可比条件数和关节限位裕量。失败结果不会覆盖上一条已验证的关节命令。

正式同步 IK 服务：

```text
arm_ik/solve  elf3_arm_ik_interfaces/srv/SolveArmIK
```

标准轨迹规划与执行 Action：

```text
arm_ik/plan_trajectory  elf3_arm_ik_interfaces/action/PlanArmTrajectory
```

查看完整请求和响应字段：

```bash
ros2 interface show elf3_arm_ik_interfaces/srv/SolveArmIK
ros2 interface show elf3_arm_ik_interfaces/action/PlanArmTrajectory
```

Action 支持只生成 `trajectory_msgs/JointTrajectory` 或直接执行。执行模式支持
Cancel；取消时优先读取最新实测关节状态，并可自动规划回 `home` 或
`handshake_ready`。轨迹自动满足请求中的最大速度、最大加速度和控制周期。

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

## 验证

可复现的精度与耗时基准：

```bash
python3 install/elf3_arm_ik_baselink/share/elf3_arm_ik_baselink/tools/benchmark_ik.py \
  --iterations 20
```

需要先启动 IK 节点的 ROS2 端到端冒烟测试：

```bash
python3 install/elf3_arm_ik_baselink/share/elf3_arm_ik_baselink/tools/ros_ik_smoke_test.py
```

正式 Service 和 Action 接口测试：

```bash
python3 install/elf3_arm_ik_baselink/share/elf3_arm_ik_baselink/tools/ros_solve_service_test.py
python3 install/elf3_arm_ik_baselink/share/elf3_arm_ik_baselink/tools/ros_trajectory_action_test.py
```

2026-07-22 Ubuntu VM 基准结果：连续跟踪 80 个样本，P95 `7.883ms`，
最大 `10.332ms`，低于 20ms 的比例为 `100%`。较难姿态从零初值冷启动仍有
`32.935ms`，因此当前 20ms 结论只适用于以当前关节状态为初值的连续控制场景。

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

正式交付差距和逐项关闭计划维护在：

```text
IK_DELIVERY_TODO.md
```

已完成：

- 基于 ELF3 URDF 的双臂 6D TCP IK 节点。
- 基于 `base_link` 的左右手目标输入接口，包含左右肩相对基座的安装偏移。
- FK 回算校验位置/姿态误差、关节限位裕量和近奇异状态。
- 不可达、近奇异、贴近关节限位或非有限输入会被拒绝并保持上一安全命令。
- 发布真实 IK 耗时和结构化求解状态。
- 正式 `SolveArmIK` Service，失败时不返回未经验证的关节解。
- 正式 `PlanArmTrajectory` Action，输出标准 `JointTrajectory`。
- 轨迹速度、加速度、控制周期约束和执行中 Cancel。
- Cancel 后从最新实测或估计停止点自动规划回安全位。
- 输出 `pico_control_joint_commands`，可对接现有手臂命令链。
- 回零服务 `/arm_ik/go_home`。
- 握手起手式服务 `/arm_ik/go_handshake_ready`：左臂回零，右臂伸出。
- 大幅演示姿态服务 `/arm_ik/go_demo_pose_a`、`/arm_ik/go_demo_pose_b`。
- RViz 展示、机器人模型、同步字幕、循环演示和耗时显示。
- 基准 ELF3 模型配置：`src/elf3_arm_ik_baselink/config/models/elf3.yaml`。

待真实联调：

- 接入 `withoutarm.onnx` state：身体、腿、腰由模型控制，手臂由本 IK 节点覆盖。
- 在真实控制链路中确认最终关节命令合并点。
- 把结构化 IK 状态接入 RViz 面板。
- 结合碰撞代理体对整条轨迹逐点检查。

## 当前保护

- 目标必须在配置工作空间内。
- IK 结果必须通过 FK 误差、关节限位裕量和雅可比条件数校验。
- 左右 TCP 目标距离过近时保持上一帧命令。
- 每周期关节变化有限幅。
- 关节命令做一阶低通平滑。
- 目标超时后保持上一帧命令。
