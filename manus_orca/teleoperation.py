#!/usr/bin/env python3
#
# ================================================================
#  Function: hand.usd → URDF → Pink IK, multi-end-effector (10) → 20 DOF right hand control
# ================================================================

import asyncio
import tempfile
import torch
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState

from isaaclab.app import AppLauncher
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers.utils import convert_usd_to_urdf
from isaaclab.controllers.pink_ik_cfg import PinkIKControllerCfg
from isaaclab.controllers.pink_ik import PinkIKController
from isaaclab.utils import configclass
from pink.tasks import FrameTask
import pinocchio as pin

# ---------------------------- Configuration ----------------------------
USD_HAND_PATH = "PATH/TO/hand.usd"  # ✏ Replace with your hand.usd path
TIP_LINK_NAMES = [
    # ✏ Fill in the 10 end-effector prim names in the order matching PoseArray
    "right_thumb_mcp_jointbody", "right_thumb_pip_jointbody",
    "right_index_mcp_jointbody", "right_index_pip_jointbody",
    "right_middle_mcp_jointbody", "right_middle_pip_jointbody",
    "right_ring_mcp_jointbody",  "right_ring_pip_jointbody",
    "right_pinky_mcp_jointbody", "right_pinky_pip_jointbody",
]
ROS_TOPIC_IN  = "/glove/r_short"            # ✏ Input PoseArray topic from glove
ROS_TOPIC_OUT = "/leaphand_node/cmd_allegro" # ✏ Output JointState topic
SCALE = 1.6                                   # Scale from glove to model

# -------------------------- Scene Configuration --------------------------
@configclass
class HandSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )
    hand = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Hand",
        spawn=sim_utils.UsdFileCfg(usd_path=USD_HAND_PATH, scale=(1, 1, 1)),
    )

# ---------------------------- ROS 2 Node ----------------------------
class GloveListener(Node):
    """Receive PoseArray from glove and store as torch.Tensor[10,3]"""
    def __init__(self, device):
        super().__init__("glove_listener_right")
        self.sub = self.create_subscription(PoseArray, ROS_TOPIC_IN, self.cb_pose, 10)
        self.pub = self.create_publisher(JointState, ROS_TOPIC_OUT, 10)
        self.buf = torch.zeros((len(TIP_LINK_NAMES), 3), device=device)  # [10,3]
        self.new_msg = False

    def cb_pose(self, msg: PoseArray):
        for i, pose in enumerate(msg.poses[:len(TIP_LINK_NAMES)]):
            self.buf[i] = torch.tensor(
                [pose.position.x, pose.position.y, -pose.position.z],  # Flip Z to match physics
                device=self.buf.device
            ) * SCALE
        self.new_msg = True

# ---------------------------- Main Coroutine ----------------------------
async def main_async():
    # 1. Convert USD to URDF automatically
    temp_dir = tempfile.gettempdir()
    urdf_path, mesh_path = convert_usd_to_urdf(
        usd_path=USD_HAND_PATH,
        output_path=temp_dir,
        force_conversion=True,
    )

    # 2. Launch Isaac Sim
    app_launcher = AppLauncher(headless=False)
    simulation_app = app_launcher.app
    sim_cfg = sim_utils.SimulationCfg(dt=1/120.)
    sim = sim_utils.SimulationContext(sim_cfg)
    device = sim.device

    # 3. Load scene
    scene = InteractiveScene(HandSceneCfg(num_envs=1))
    hand_view = scene["hand"]
    sim.reset()
    sim.set_camera_view([1.2, 1.2, 1.2], [0, 0, 0])

    # 4. Configure Pink IK tasks
    tip_tasks = [
        FrameTask(frame=name, position_cost=1.0, orientation_cost=0.2, lm_damping=10)
        for name in TIP_LINK_NAMES
    ]
    ik_cfg = PinkIKControllerCfg(
        urdf_path=urdf_path,
        mesh_path=mesh_path,
        joint_names=hand_view.prim_path_to_joint_names(),
        articulation_name="hand",
        base_link_name="right_wrist_jointbody",
        num_hand_joints=0,
        variable_input_tasks=tip_tasks,
        fixed_input_tasks=[],
        show_ik_warnings=True,
    )
    ik = PinkIKController(cfg=ik_cfg, device=device)

    # 5. End-effector and joint indices
    ee_ids = hand_view.find_bodies(TIP_LINK_NAMES)
    ee_ids_t = torch.tensor(ee_ids, device=device)
    joint_ids = hand_view.dof_indices

    # 6. Initialize ROS 2 node
    rclpy.init()
    glove_node = GloveListener(device=device)
    executor = rclpy.get_global_executor()

    print("[INFO] Ready → Pink IK (10 EE → 20 DOF)")

    # 7. Main loop
    while simulation_app.is_running():
        executor.spin_once(glove_node, timeout_sec=0.0)

        if glove_node.new_msg:
            glove_node.new_msg = False
            targets = glove_node.buf.reshape(1, -1)  # [1,30] = 10 x 3

            for task, pos in zip(ik.cfg.variable_input_tasks, targets.view(-1, 3)):
                T = pin.SE3.Identity()
                T.translation = pin.utils.toNumpy(pos)
                task.set_target(T)

            # Get current state
            jac = hand_view.root_physx_view.get_jacobians()[:, ee_ids_t, :, joint_ids]
            body_state = hand_view.data.body_state_w[:, ee_ids_t, 0:7]
            root_state = hand_view.data.root_state_w[:, 0:7]
            q_cur = hand_view.data.joint_pos[:, joint_ids]

            # Compute desired joint positions
            q_des = ik.compute(
                curr_joint_pos=q_cur.squeeze(0).cpu().numpy(),
                dt=sim_cfg.dt,
            )

            # Apply joint targets
            hand_view.set_joint_position_target(q_des, joint_ids=joint_ids)

            # Publish ROS joint state
            js = JointState()
            js.position = q_des.flatten().tolist()
            glove_node.pub.publish(js)

        # Simulation step
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)

    # 8. Shutdown
    glove_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()

# ---------------------------- Entry Point ----------------------------
if __name__ == "__main__":
    asyncio.run(main_async())