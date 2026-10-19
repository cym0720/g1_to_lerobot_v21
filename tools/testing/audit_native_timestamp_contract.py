#!/usr/bin/env python3
"""CPU read-only checks of the current native 20D/8D timestamp contract."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
import train_g1_native_lerobot as baseline

ROOT = baseline.REPO / "output/lerobot_act_control_20260929"


def main():
    torch.set_num_threads(2)
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.act.configuration_act import ACTConfig
    reports = {}
    for name in ("native_train", "right_arm_train"):
        root = ROOT/name/"dataset"
        info = json.loads((root/"meta/info.json").read_text())
        fps = info["fps"]
        assert fps == 30
        rows, offset, offsets, lengths = [], 0, {}, {}
        tables = {}
        for path in sorted((root/"data").rglob("*.parquet")):
            table = pq.read_table(path)
            episode = int(table["episode_index"][0].as_py())
            frame = table["frame_index"].to_numpy()
            timestamp = table["timestamp"].to_numpy()
            expected = (np.arange(len(table),dtype=np.float64)/fps).astype(np.float32)
            np.testing.assert_array_equal(frame,np.arange(len(table)))
            np.testing.assert_array_equal(table["index"].to_numpy(),offset+np.arange(len(table)))
            np.testing.assert_array_equal(timestamp,expected)
            assert str(table.schema.field("timestamp").type) == "float"
            assert len(table)<2 or np.all(np.diff(timestamp)>0)
            mathematical_error = float(np.max(np.abs(timestamp.astype(np.float64)-np.arange(len(table))/fps)))
            rows.append(dict(episode=episode,frames=len(table),dtype="float32",units="seconds",
                first_timestamp=float(timestamp[0]),last_timestamp=float(timestamp[-1]),
                exact_float32_frame_over_fps=True,max_float32_rounding_error_seconds=mathematical_error))
            offsets[episode],lengths[episode]=offset,len(table)
            tables[episode]=table
            offset += len(table)
        assert offset == info["total_frames"]
        assert len(rows) == info["total_episodes"]
        cfg=TrainPipelineConfig(dataset=DatasetConfig(repo_id="local/g1_0923_pick_apple",root=str(root),
            video_backend="pyav",use_imagenet_stats=True),
            policy=ACTConfig(chunk_size=16,n_action_steps=16,device="cpu",use_amp=False))
        dataset=make_dataset(cfg)
        dataset.image_transforms=baseline.Letterbox()
        assert dataset.delta_indices["action"] == list(range(16)),dataset.delta_indices
        np.testing.assert_allclose(dataset.delta_timestamps["action"],np.arange(16)/fps,rtol=0,atol=1e-12)
        samples=[]
        for episode in (0,45,90):
            table=tables[episode]
            action=np.asarray(table["action"].to_pylist(),dtype=np.float32)
            state=np.asarray(table["observation.state"].to_pylist(),dtype=np.float32)
            for frame in (0,1,lengths[episode]-16,lengths[episode]-1):
                item=dataset[offsets[episode]+frame]
                indices=np.minimum(frame+np.arange(16),lengths[episode]-1)
                np.testing.assert_array_equal(item["action"].numpy(),action[indices])
                np.testing.assert_array_equal(item["observation.state"].numpy(),state[frame])
                np.testing.assert_array_equal(item["action_is_pad"].numpy(),frame+np.arange(16)>=lengths[episode])
                assert float(item["timestamp"]) == float(np.float32(frame/fps))
                samples.append(dict(episode=episode,frame=frame,future_indices=indices.tolist(),
                    timestamp_seconds=float(item["timestamp"]),correct_action_and_padding=True))
        reports[name]=dict(episodes=rows,total_frames=offset,fps=fps,
            timestamp_feature=info["features"]["timestamp"],
            native_delta_indices=dataset.delta_indices["action"],
            native_delta_seconds=dataset.delta_timestamps["action"],samples=samples,
            native_video_tolerance_seconds=dataset.tolerance_s)
        (ROOT/"timestamp_contract_audit.json").write_text(json.dumps(dict(phase="sampling",runs=reports),indent=2))
        print("TIMESTAMP_CONTRACT_RUN_OK",name,len(rows),offset,flush=True)
    result=dict(phase="complete",runs=reports,
        conclusion="Both current native datasets conform: float32 seconds, per-episode zero origin, frame/fps at 30 FPS, action offsets 0..15, correct end padding. No millisecond/epoch timestamps or extra 16-step shift.",
        limitation="Format/index checks do not independently prove physical camera synchronization or actuator command/feedback latency. Refer to raw source audit for recorded stream clock alignment; do not automatically shift labels.")
    (ROOT/"timestamp_contract_audit.json").write_text(json.dumps(result,indent=2))
    print("TIMESTAMP_CONTRACT_AUDIT_COMPLETE",flush=True)


if __name__=="__main__":main()
