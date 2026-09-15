import numpy as np
import threading
import time
import asyncio
from typing import Any
from scipy.spatial.transform import Rotation as R

from lerobot.teleoperators.teleoperator import Teleoperator
from .config_so101_vuer_teleop import So101VuerTeleopConfig

from vuer import Vuer, VuerSession
from vuer.schemas import Hands, MotionControllers, ImageBackground, Scene, CoordsMarker
import cv2
import base64
from .clutch import ClutchState, clutch_step
from .filters import OneEuroState, one_euro_step, quat_continuous
from .ik import ARM_JOINTS, Q_NOMINAL, ArmIK
from .xr_input import XRSample, parse_controllers, parse_hands

# One-euro filter tuning for the incoming hand pose.
POS_MIN_CUTOFF, POS_BETA = 1.0, 5.0   # position: Hz, per (m/s)
ROT_MIN_CUTOFF, ROT_BETA = 1.0, 0.5   # quaternion components: Hz, per (1/s)
# Vuer streams at ~30 Hz; no sample for this long means tracking was lost.
STALE_AFTER_S = 0.15


class So101VuerTeleop(Teleoperator):
    config_class = So101VuerTeleopConfig
    name = "so101_vuer"

    def __init__(self, config: So101VuerTeleopConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._clock = time.monotonic

        # Threading mechanisms
        self._vuer_thread = None
        self._lock = threading.Lock()

        # XR input (written by the Vuer thread, read by get_action; guarded by _lock)
        self._sample: XRSample | None = None
        self._sample_t: float | None = None
        self._pos_filter = OneEuroState()
        self._rot_filter = OneEuroState()

        # Arm state (main control loop only). Q_NOMINAL until seed_from_robot() supplies the real pose.
        self.ik: ArmIK | None = None
        self._clutch = ClutchState()
        self._active_source: str | None = None
        self._q = Q_NOMINAL.copy()
        self._gripper_open = 0.0

        # --- NEW: Visualizer State ---
        self._viz_pos = np.array([0.0, 0.0, 0.0])
        self._viz_rot = np.array([0.0, 0.0, 0.0])
        # -----------------------------

        self._latest_frame_b64 = None
        self._latest_frame = None  # patched: was never initialised -> AttributeError with no camera

    def seed_from_robot(self, observation: dict) -> None:
        """Start from the arm's measured pose so the first actions hold it still."""
        q = np.radians([float(observation[f"{joint}.pos"]) for joint in ARM_JOINTS])
        gripper = float(observation["gripper.pos"])
        if not (np.all(np.isfinite(q)) and np.isfinite(gripper)):
            raise ValueError(f"Robot observation has non-finite joint values: {observation}")
        self._q = q
        self._gripper_open = gripper / 100.0

    def _on_sample(self, sample: XRSample, t: float) -> None:
        """Filter a raw XR sample and publish it for the control loop."""
        with self._lock:
            restart = (
                self._sample is None or t - self._sample_t > STALE_AFTER_S or sample.source != self._sample.source
            )  # after a tracking gap or a hand/controller switch, never blend with the old pose
            pos_state = OneEuroState() if restart else self._pos_filter
            rot_state = OneEuroState() if restart else self._rot_filter
            self._pos_filter, pos = one_euro_step(pos_state, sample.pose[:3, 3], t, POS_MIN_CUTOFF, POS_BETA)
            quat = quat_continuous(rot_state.x, R.from_matrix(sample.pose[:3, :3]).as_quat())
            self._rot_filter, quat = one_euro_step(rot_state, quat, t, ROT_MIN_CUTOFF, ROT_BETA)
            pose = np.eye(4)
            pose[:3, :3] = R.from_quat(quat).as_matrix()
            pose[:3, 3] = pos
            self._sample = XRSample(pose=pose, clutch=sample.clutch, gripper_open=sample.gripper_open, source=sample.source)
            self._sample_t = t
            self._viz_pos = sample.pose[:3, 3].copy()
            self._viz_rot = R.from_matrix(sample.pose[:3, :3]).as_euler("xyz")

    def _vuer_worker(self):
        """Background thread for the Vuer asyncio event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        app = Vuer(host=self.config.vuer_host, cert=self.config.vuer_cert, key=self.config.vuer_key)

        @app.add_handler("HAND_MOVE")
        async def on_hand_move(event, session):
            sample = parse_hands(event.value, self.config.user_hand)
            if sample is not None:
                self._on_sample(sample, self._clock())

        @app.add_handler("CONTROLLER_MOVE")
        async def on_controller_move(event, session):
            sample = parse_controllers(event.value, self.config.user_hand)
            if sample is not None:
                self._on_sample(sample, self._clock())

        @app.spawn(start=True)
        async def main(session: VuerSession):
            session.set(Scene())
            session.upsert(Hands(stream=True, key="hands", showLeft=True, showRight=True), to="bgChildren")
            session.upsert(MotionControllers(stream=True, key="motionControllers", left=True, right=True), to="bgChildren")
            last_sent_b64 = None
            while self._is_connected:
                with self._lock:
                    current_img = self._latest_frame
                    # Safely copy the gizmo state
                    viz_pos = self._viz_pos.copy()
                    viz_rot = self._viz_rot.copy()

                # --- NEW: Render the Vector Gizmo ---
                session.upsert(
                    CoordsMarker(
                        position=viz_pos.tolist(),
                        rotation=viz_rot.tolist(),
                        scale=0.15, # Sets the vectors to be 15cm long
                        key="ik_gizmo"
                    ),
                    to="bgChildren"
                )

                if current_img is not None:
                    session.upsert(
                        ImageBackground(
                            current_img,
                            format="jpeg",
                            quality=50,
                            fixed=True,
                            # Pushes the screen further away from your face (80cm)
                            distanceToCamera=1,
                            key="camera_feed",
                            # X=0 (centered), 3 m ahead, slightly below a standing eye level
                            position=[0, 1.0, -3],
                        ),
                        to="bgChildren"
                    )

                await asyncio.sleep(0.033)

        print("VR Server Started. Waiting for headset connection...")
        app.run()

    def _camera_worker(self):
        """Autonomously searches for the active robot in memory to copy its camera feed, bypassing LeRobot's closed CLI loop."""
        import gc
        from lerobot.robots.robot import Robot as BaseRobot
        active_robot = None

        while self._is_connected:
            try:
                if active_robot is None:
                    # Dynamically find the active robot initialized by the CLI
                    for obj in gc.get_objects():
                        if isinstance(obj, BaseRobot) and getattr(obj, "is_connected", False):
                            active_robot = obj
                            break

                if active_robot is not None:
                    # patched: read camera objects directly instead of get_observation(), which
                    # also reads the motor bus and collides with the main loop on a real serial
                    # arm ("Port is in use!"). With no cameras there is nothing to stream: stop.
                    cameras = getattr(active_robot, "cameras", None) or {}
                    if not cameras:
                        return
                    selected_img = None
                    for cam in cameras.values():
                        val = cam.async_read()
                        if isinstance(val, np.ndarray) and val.ndim == 3:
                            selected_img = val
                            break

                    if selected_img is not None:
                        # Convert to uint8 if necessary
                        if selected_img.dtype != np.uint8:
                            selected_img = (np.clip(selected_img, 0, 1) * 255).astype(np.uint8)

                        # Deep copy and resize to save bandwidth (fixes the SegFault and Network Choking)
                        selected_img = cv2.resize(selected_img.copy(), (320, 240))

                        with self._lock:
                            # Pass the raw numpy array directly! No Base64 or OpenCV conversion needed.
                            self._latest_frame = selected_img
            except Exception:
                pass
            time.sleep(0.033) # ~30 FPS polling

    def _init_kinematics(self) -> None:
        self.ik = ArmIK(self.config.mjcf_path)

    def connect(self) -> None:
        self._init_kinematics()
        self._is_connected = True

        self._vuer_thread = threading.Thread(target=self._vuer_worker, daemon=True)
        self._cam_thread = threading.Thread(target=self._camera_worker, daemon=True)

        self._vuer_thread.start()
        self._cam_thread.start()

    def disconnect(self) -> None:
        self._is_connected = False
        # Note: Vuer apps are notoriously difficult to kill gracefully from a thread,
        # but the daemon=True flag ensures it dies when the main LeRobot script exits.

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def get_action(self) -> dict:
        with self._lock:
            sample, sample_t = self._sample, self._sample_t

        if sample is None or self._clock() - sample_t > STALE_AFTER_S:
            # Tracking lost (or never started): release, so a reappearing hand can never jump the arm.
            self._clutch, self._active_source = ClutchState(), None
        else:
            # Hand <-> controller switch: release for one tick. A first appearance is not a switch; the
            # stale branch above already released the clutch and _on_sample restarted the filters.
            switched = self._active_source is not None and sample.source != self._active_source
            self._active_source = sample.source
            current = self.ik.fk(self._q)  # anchor to what the arm can actually reach
            self._clutch, target = clutch_step(
                self._clutch, sample.clutch and not switched, sample.pose, current, self.config.motion_scale
            )
            if target is not None:
                q = self.ik.solve(self._q, target)
                self._q = q if np.all(np.isfinite(q)) else self._q
            self._gripper_open = sample.gripper_open

        action = {f"{joint}.pos": float(np.degrees(angle)) for joint, angle in zip(ARM_JOINTS, self._q)}
        return {**action, "gripper.pos": self._gripper_open * 100.0}

    @property
    def action_features(self) -> dict:
        return {
            "shoulder_pan.pos": float, "shoulder_lift.pos": float, "elbow_flex.pos": float,
            "wrist_flex.pos": float, "wrist_roll.pos": float, "gripper.pos": float,
        }

    @property
    def feedback_features(self) -> dict: return {}
    @property
    def is_calibrated(self) -> bool: return True
    def calibrate(self) -> None: pass
    def configure(self) -> None: pass

    def send_feedback(self, feedback: dict[str, Any]) -> None: pass
