import json, sys
p = sys.argv[1]
r = json.load(open(p))
print("dimension_contract:")
print(json.dumps(r.get("dimension_contract"), indent=1, ensure_ascii=False)[:1200])
print("videos:", json.dumps(r.get("videos"))[:200])
e = r["episodes"][0] if isinstance(r.get("episodes"), list) else list(r["episodes"].values())[0]
print("episode keys:", list(e.keys()))
print("metrics:", json.dumps(e.get("metrics"), ensure_ascii=False)[:600])
print("chunk_dynamics keys:", list((e.get("chunk_dynamics") or {}).keys()))
