"""Build a new 100/6 split; preserve previous datasets and checkpoint."""
import json
import time
from pathlib import Path
from prepare_merged_starvla_1006 import build

def main():
    root = Path(__file__).resolve().parents[2] / 'output/pick_apple_merged_20261006'
    output = root / 'training_100_6'
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    val = [5, 17, 29, 41, 53, 94]
    train = [i for i in range(106) if i not in val]
    assert len(train) == 100 and len(val) == 6 and not set(train) & set(val)
    build(root / 'merged_base', output / 'train100', train)
    build(root / 'merged_base', output / 'val6', val)
    split = dict(train_merged_episode_ids=train, val_merged_episode_ids=val,
                 old_train=86, new_train=14, old_val=5, new_val=1,
                 elapsed_seconds=time.monotonic()-started,
                 existing_20000_step_model='trained on previous train96; not retrained on this split')
    (output / 'split.json').write_text(json.dumps(split, indent=2))
    (output / 'README.md').write_text('Pick apple: train100 (86 deploy_g1 + 14 1004); val6 (5 deploy_g1 + 1 1004).\n'
        'Whole-episode disjoint split. Three 224x224 cameras, state/action20, chunk16 compatible.\n'
        'Continuous labels: state[t+3], clamped at episode end. Grippers: original same-frame action >=0.5.\n'
        'Body output: index18=lift, index19=waist pitch. Swap only when required by robot interface.\n'
        'Statistics recomputed separately for each split; validation must use training normalization.\n'
        'Existing 20000-step checkpoint was trained on previous train96.\n')

if __name__ == '__main__':
    main()
