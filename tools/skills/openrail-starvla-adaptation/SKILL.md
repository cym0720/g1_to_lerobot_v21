---
name: openrail-starvla-adaptation
description: Adapt a trained StarVLA checkpoint bundle to Open-RAIL GPU serving and client configuration, then diagnose live inference or robot connection failures. Use for model-to-Open-RAIL integration; not for training or dataset conversion.
---

# Open-RAIL StarVLA adaptation

Adapt the selected checkpoint to the installed Open-RAIL runtime without changing model weights or statistics. Inspect the live host, server process, client process, newest relevant Codex session, package files, and Open-RAIL config loader before editing. Treat old session notes as hypotheses; confirm every path, port, and process against the current host.

## Files and responsibilities

### Model deployment bundle: inspect, preserve

For a typical model directory `/data/open-rail/data/models/<bundle>/<profile>/`, verify these files and their matching checkpoint step:

- `checkpoints/steps_<N>_pytorch_model.pt`: selected weights. Never rewrite as part of client adaptation.
- `config.yaml`: training/model architecture configuration. Use it to verify camera keys, model dimensions, and normalization expectations; do not casually edit it to fix a client or network problem.
- `dataset_statistics.json`: normalization statistics used by the model. Preserve and verify it accompanies the checkpoint.
- `client.yaml`: package-side reference for expected robot type, camera mapping, state/action order, and client settings. It is not necessarily directly accepted by `run_web_client.py`.
- `serve.sh`: model-side GPU server launcher; check the step allowlist and camera arguments before use.
- `README.txt`, `preflight.json`, `SHA256SUMS`, and any training/data audit receipts: inspect as provenance and verify checksums after delivery.

Do not change weights, training config, or normalization statistics to solve Open-RAIL UI timeouts or robot network failures.

### Open-RAIL runtime client config: make an isolated copy

- Open-RAIL client configs live under `/data/open-rail/conf/`.
- `run_web_client.py --conf` accepts a config filename resolved as `ROOT/conf/<filename>`; passing the model bundle's absolute `client.yaml` path does not load that file.
- Create a new descriptive config such as `starvla_<bundle>_<step>.yaml`, based on a known compatible Open-RAIL profile (for example the installed `starvla_fisheye.yaml`). Do not overwrite the default or an existing profile. Back up before any intentional in-place correction.
- Compare the copy against both the model bundle's `client.yaml` and the existing profile. Make only evidenced changes. Verify:
  - `robots.type` and G1 robot subtype;
  - camera names and reference camera (`head`, `head_left_fisheye`, `head_right_fisheye` as appropriate), plus `color_order`;
  - state/action `io_fields`, dimensions, joint order, and action layout against model metadata;
  - controller mode, arm/gripper policies, and preprocessing expected by the deployed checkpoint;
  - `vla_zmq.ip` and port match the GPU server listener.
- Use a dedicated `vla_zmq.infer_timeout` with enough margin for observed real inference latency. Historical 10.5 s inference exceeded the previous 10,000 ms threshold; 30,000 ms was used in the 2026-10-28 FT105 adaptation. Measure the current model before choosing a different value. Do not treat a longer timeout as a fix for disconnected ZMQ, missing observations, or unreachable robot DDS.
- Validate YAML parsing, expected camera/state/action fields, selected timeout, and SHA256. Record the source config and exact changed keys.

### Open-RAIL code and server process: normally inspect, do not patch

- `/data/open-rail/run_web_client.py`: entry point; confirms config resolution and Uvicorn behavior.
- `/data/open-rail/web_client/server.py`: status/config/start APIs and lifecycle behavior.
- `/data/open-rail/client/core/vla_client.py`: ZMQ request timeout and observe/inference/control loops.
- `/data/open-rail/client/robots/robot_proxy.py` and the selected robot implementation: robot process, DDS and camera/control wiring.
- `/data/open-rail/scripts/run_starvla_server.sh`: GPU environment setup and server launcher.

Do not edit these sources for a model-specific adaptation unless a reproduced defect requires a code fix. Prefer the bundle's `serve.sh <step>` and existing Open-RAIL launch script. Confirm the server's actual checkpoint path, camera keys, PID, GPU/runtime status, and listening port before starting a second server.

## Validation and diagnosis

1. **Provenance:** check checkpoint step, bundle files, model I/O metadata, and checksums.
2. **Config:** validate the isolated config and compare camera/state/action contracts by names and dimensions, not just vector length.
3. **GPU model server:** issue one bounded synthetic ZMQ request directly to the server, with no robot object or command publisher. Verify finite response shape and latency. Label it as synthetic server validation; it does not prove real-camera inference or robot movement.
4. **Client UI/API:** start only one `run_web_client.py`. Read `/api/client/config/path` and `/api/client/status`; record `running`, `observe_running`, `inference_running`, `control_running`, `infer_count`, `avg_infer_time`, `inference_error_count`, and `zmq_connected`. Distinguish browser/WebSocket preview disconnects from inference timeouts and reset HTTP completion.
5. **Robot network:** before client runtime start, inspect the configured locator IP, `ip -br addr`, `ip route`, route selection, and bounded reachability. Errors such as `Network not ready`, missing PTP, or `Destination Net Unreachable` are robot transport failures; increasing model inference timeout cannot repair them. Stop/leave the failed initialization safely and report the missing network path. Do not change robot locator IP or host routes by guessing.
6. **One client only:** ensure no other client owns robot/DDS/control connections. A UI process can exist while runtime status is still `running=false` and `zmq_connected=false`.

## Live robot start gate

Open-RAIL's `/api/client/start` calls the VLA client's `start()`, which starts observation, inference, and control together. It can send model actions to the physical robot; it is not a harmless connectivity probe. Before calling it, obtain direct user confirmation that an operator is watching, the motion area is clear, and the emergency stop is available. If any condition is absent, limit work to config/server/API checks and provide the exact next step.

After an authorized start, monitor status and logs immediately. Confirm fresh observations, successful inference replies, and control state; do not equate HTTP 200, process existence, port listening, or a synthetic GPU probe with robot motion. If startup repeatedly reports robot network failure, stop retrying and report the interface/route evidence.

## 2026-10-28 FT105 adaptation record

- GPU server used `.../pickapple_fisheye_ft105_agi10k_bs32_20261028/fisheye_8d/checkpoints/steps_10000_pytorch_model.pt`, camera keys `cam.head cam.hand_left cam.hand_right`, listening on 5566.
- Created `/data/open-rail/conf/starvla_fisheye_ft105_10k.yaml` from the existing `starvla_fisheye.yaml`; changed only `vla_zmq.infer_timeout` from 10000 to 30000 ms. Preserved the installed synchronous G1 client profile. No model weights, statistics, Open-RAIL source files, or existing config were changed.
- The GPU server passed direct synthetic ZMQ probes (action shape `(64, 20)`, around 0.8–1.1 seconds); this was not a real-camera/robot acceptance test.
- Client process started with the new config. Starting the runtime through `/api/client/start` created a G1 worker but failed because AGI had no route to locator `192.168.100.21`: `Network not ready`, `No valid PTP device found`, and ping returned `Destination Net Unreachable` through `10.42.5.200`. No Observe, Inference, Control, or ZMQ-connected state was reached. The robot was therefore not shown to move.
- In later work, rediscover the current client and server processes and current network before retrying; do not assume this host/network snapshot remains true.
