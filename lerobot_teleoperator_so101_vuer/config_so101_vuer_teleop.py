from dataclasses import dataclass
from lerobot.teleoperators.config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("so101_vuer")
@dataclass
class So101VuerTeleopConfig(TeleoperatorConfig):
    mjcf_path: str = ""  # MuJoCo model used for IK, e.g. <robot-arm>/sim/so101/scene.xml
    user_hand: str = "right"  # "left" or "right"
    motion_scale: float = 0.5  # metres of gripper travel per metre of hand travel
    vuer_host: str = "0.0.0.0"
    vuer_cert: str = "./cert.pem"
    vuer_key: str = "./key.pem"

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        if self.user_hand not in ("left", "right"):
            raise ValueError(f"user_hand must be 'left' or 'right', got {self.user_hand!r}")
        if not self.motion_scale > 0.0:
            raise ValueError(f"motion_scale must be positive, got {self.motion_scale!r}")
