"""Keep this arm's safety envelope alive inside LeRobot's own entry points.

`lerobot-record` and `lerobot-rollout` build and connect the robot themselves, so
anything this repo does to the arm before handing off is either undone or never done:

* `SOFollower.connect()` ends in `configure()`, which writes `P_Coefficient=16` to every
  motor. Gains applied before the hand-off are overwritten; only `Acceleration` survives.
* `SOFollower.connect()` defaults to `calibrate=True`, so a calibration mismatch stops an
  autonomous run dead on an interactive prompt.
* Nothing seeds the teleoperator from the arm's measured pose, so it would start at its
  nominal posture and drive the arm there on the first recorded tick, before the operator
  has clutched in.

Wrapping the classes is the only hook that lands *after* LeRobot's own writes. Each
installer patches a class in this process only and is safe to call more than once.
"""
import logging
from functools import wraps

from run_quest import apply_gains

logger = logging.getLogger(__name__)

# Marker set on every wrapper we install, so installing twice is a no-op.
_PATCHED = "_so101_pipeline_patch"

# The follower arm the teleoperator should read its starting pose from. LeRobot connects
# the robot before the teleoperator, so the connect hook records it for the seeding hook.
_connected_robot = None


def note_connected_robot(robot) -> None:
    """Record the follower arm that a later teleoperator seeding should read its pose from."""
    global _connected_robot
    _connected_robot = robot


def install_gains() -> None:
    """Re-apply this arm's servo gains at the end of every `SOFollower.configure()`.

    LeRobot calls `configure()` from `connect()`, so the gains land inside whichever
    process owns the robot, after LeRobot has written its own `P_Coefficient=16`.
    """
    from lerobot.robots.so_follower.so_follower import SOFollower

    if getattr(SOFollower.configure, _PATCHED, False):
        return

    original = SOFollower.configure

    @wraps(original)
    def configure_then_apply_gains(self) -> None:
        original(self)
        apply_gains(self)

    setattr(configure_then_apply_gains, _PATCHED, True)
    SOFollower.configure = configure_then_apply_gains


def install_noninteractive_connect() -> None:
    """Connect the follower without LeRobot's calibration prompt, and remember it.

    An autonomous run has nobody at the keyboard to answer `calibrate()`'s prompt, so we
    connect the way park.py does. A mismatch is then reported loudly instead of silently:
    the joint angles would be wrong, and that is a reason to stop, not to guess.
    """
    from lerobot.robots.so_follower.so_follower import SOFollower

    if getattr(SOFollower.connect, _PATCHED, False):
        return

    original = SOFollower.connect

    @wraps(original)
    def connect_without_prompting(self, calibrate: bool = False) -> None:
        original(self, calibrate=False)
        if not self.is_calibrated:
            logger.error(
                "The motors' calibration does not match '%s'. Joint angles will be wrong. "
                "Stop the run and re-calibrate this arm before using it.",
                self.calibration_fpath,
            )
        note_connected_robot(self)

    setattr(connect_without_prompting, _PATCHED, True)
    SOFollower.connect = connect_without_prompting


def install_teleop_seeding() -> None:
    """Seed the Quest teleoperator from the arm's measured pose as the teleoperator connects.

    This is what `run_quest.py` does explicitly; `lerobot-record` has no such step, and
    without it the first action commands the teleoperator's nominal posture.
    """
    from lerobot_teleoperator_so101_vuer.so101_vuer_teleop import So101VuerTeleop

    if getattr(So101VuerTeleop.connect, _PATCHED, False):
        return

    original = So101VuerTeleop.connect

    @wraps(original)
    def connect_then_seed(self) -> None:
        original(self)
        if _connected_robot is None:
            raise RuntimeError(
                "The teleoperator connected before the follower arm, so it cannot be seeded "
                "from the measured pose. Install the connect hook first, or the arm would be "
                "driven to the teleoperator's nominal posture on the first tick."
            )
        self.seed_from_robot(_connected_robot.get_observation())
        logger.info("teleop seeded from the arm's measured pose")

    setattr(connect_then_seed, _PATCHED, True)
    So101VuerTeleop.connect = connect_then_seed
