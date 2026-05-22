"""事件检测器统一注册与导出"""
from events.back_facing  import BackFacingDetector
from events.gazing       import GazingDetector
from events.eye_state    import EyeStateDetector
from events.waving       import WavingDetector
from events.fall_down    import FallDownDetector
from events.talking      import TalkingDetector
from events.pick_up      import PickUpDetector
from events.door_operate import DoorOperateDetector
from events.sit_stand    import SitStandDetector
from events.fire_smoke   import FireSmokeDetector


def build_all_detectors() -> list:
    """返回所有事件检测器实例列表（有序）"""
    return [
        # 人相关
        BackFacingDetector(),
        GazingDetector(),
        # EyeStateDetector(),
        WavingDetector(),
        # FallDownDetector(),
        # TalkingDetector(),
        # # 人与环境
        # PickUpDetector(),
        # DoorOperateDetector(),
        # SitStandDetector(),
        # # 环境
        # FireSmokeDetector(),
    ]
