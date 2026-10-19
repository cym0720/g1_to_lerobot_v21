import json
import numpy as np
from pathlib import Path

ep = Path("/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014/runs/DP_1008_h20_aug_h16_20k_retry1/gt/step_020000/episode_000000")
z = np.load(ep / "arrays.npz")
report = json.load(open(ep.parent / "report.json"))
metrics = json.load(open(ep / "metrics.json"))

gt = z["gt_action"].astype(np.float64)
pred = z["nonoverlap_chunk_execution"].astype(np.float64)
print("gt frames:", gt.shape)
print("--- per-dim gt min/max vs report q01/q99 (dims 8..15) ---")
q01 = report["normalization"]["action_q01"]
q99 = report["normalization"]["action_q99"]
for d in range(8, 16):
    print("dim%d gt[%.4f,%.4f] q01=%.4f q99=%.4f" % (d, gt[:, d].min(), gt[:, d].max(), q01[d], q99[d]))

diff = np.abs(pred[:, 8:14] - gt[:, 8:14])
per_joint = diff.mean(axis=0)
print("--- joint MAE (dims8-14, all frames) ---")
print(np.round(per_joint, 6))
print("mean=%.6f median=%.6f" % (per_joint.mean(), np.median(per_joint)))
print("rmse_mean=%.6f" % np.sqrt((diff ** 2).mean()))

sec = metrics.get("metrics", {}).get("nonoverlap_chunk_execution", {})
print("--- episode metrics.json nonoverlap section ---")
print(json.dumps(sec, ensure_ascii=False)[:900])

ep0 = report["episodes"][0] if isinstance(report["episodes"], list) else list(report["episodes"].values())[0]
rep_sec = ep0.get("metrics", {}).get("nonoverlap_chunk_execution", {})
print("--- report episode nonoverlap keys ---")
print(json.dumps({k: rep_sec[k] for k in rep_sec if "mae" in k or "rmse" in k}, ensure_ascii=False)[:500])

# variant: exclude first chunk
d2 = np.abs(pred[16:, 8:14] - gt[16:, 8:14]).mean()
print("mae excl first chunk=%.6f" % d2)
# variant: mean of per-chunk means
starts = z["chunk_starts"]
chunk_maes = []
for i, s in enumerate(starts):
    e = starts[i + 1] if i + 1 < len(starts) else gt.shape[0]
    chunk_maes.append(np.abs(pred[s:e, 8:14] - gt[s:e, 8:14]).mean())
print("mean of per-chunk MAEs=%.6f" % np.mean(chunk_maes))
