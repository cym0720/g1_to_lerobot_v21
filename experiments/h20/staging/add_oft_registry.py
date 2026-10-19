import importlib.util
from pathlib import Path

DC = Path("/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014/baselines/starvla_local/examples/realRobots/PickAppleACT/train_files/data_registry/data_config.py")
txt = DC.read_text()

addition = '''
# --- OFT 20261016 registry (horizon 32, unfrozen VLM, base cached dataset) ---
_OFT_VIDEO_KEYS = [
    "video.cam_high_rgb",
    "video.cam_left_wrist_rgb",
    "video.cam_right_wrist_rgb",
]
_OFT_STATE_KEYS = ["state.robot"]
_OFT_ACTION_KEYS = ["action.robot"]
_OFT_LANGUAGE_KEYS = ["annotation.human.task_description"]
_OFT_HORIZON = 32


class PickAppleG1OFTConfig:
    """G1 20D state/action contract for QwenOFT horizon-32."""

    embodiment_tag = EmbodimentTag.AGIBOT_G1
    video_keys = _OFT_VIDEO_KEYS
    state_keys = _OFT_STATE_KEYS
    action_keys = _OFT_ACTION_KEYS
    language_keys = _OFT_LANGUAGE_KEYS
    state_key_dims = {"state.robot": 20}
    action_key_dims = {"action.robot": 20}
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(_OFT_HORIZON))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(apply_to=self.state_keys, normalization_modes={"state.robot": "q99"}),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(apply_to=self.action_keys, normalization_modes={"action.robot": "q99"}),
            ]
        )


ROBOT_TYPE_CONFIG_MAP["g1_pick_apple_oft"] = PickAppleG1OFTConfig()
DATASET_NAMED_MIXTURES["g1_pick_apple_20261016_train_h32"] = [("train16", 1.0, "g1_pick_apple_oft")]
'''

if "g1_pick_apple_oft" not in txt:
    DC.write_text(txt.rstrip() + "\n" + addition)
    print("PATCHED added OFT registry")
else:
    print("already had OFT registry")

spec = importlib.util.spec_from_file_location("_reg_check", DC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for k in ("g1_pick_apple_oft",):
    print("VERIFY robot", k, "present" if k in mod.ROBOT_TYPE_CONFIG_MAP else "MISSING")
for k in ("g1_pick_apple_20261016_train_h32", "g1_pick_apple_1015_train36_c16"):
    print("VERIFY mix", k, "present" if k in mod.DATASET_NAMED_MIXTURES else "MISSING")
cfg = mod.ROBOT_TYPE_CONFIG_MAP.get("g1_pick_apple_oft")
if cfg:
    print("OFT action_indices len =", len(cfg.action_indices))
