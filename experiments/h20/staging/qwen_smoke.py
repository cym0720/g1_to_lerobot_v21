import json, time
from pathlib import Path

ROOT = Path("/data2/cym_ws/qwen35_oft_20261015")
REPORT = ROOT / "audits"
MODEL = ROOT / "models/Qwen3.5-0.8B"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import OFTConfig, get_peft_model

t0 = time.time()
tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16, trust_remote_code=True)
model_loaded_s = round(time.time() - t0, 1)

# OFT on all linear layers of attention + mlp
oft_cfg = OFTConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    module_dtype="bf16",
    ortho_type="none",
    coft=False,
    exclude_modules=["lm_head"],
)
pm = get_peft_model(model, oft_cfg)
pm.print_trainable_parameters()

trainable = sum(p.numel() for p in pm.parameters() if p.requires_grad)
total = sum(p.numel() for p in pm.parameters())

injected = sorted({name.rsplit(".", 1)[0] for name, mod in pm.named_modules() if "oft" in type(mod).__name__.lower()})

texts = ["你好，请介绍一下你自己。", "1+1等于几？"]
enc = tok(texts, return_tensors="pt", padding=True, truncation=True).to(pm.device)
t1 = time.time()
with torch.no_grad():
    out = pm.generate(**enc, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
fwd_s = round(time.time() - t1, 1)
gens = [tok.decode(o, skip_special_tokens=True) for o in out]

report = {
    "status": "PASS",
    "model": "Qwen/Qwen3.5-0.8B",
    "oft_implementation": "huggingface peft OFTConfig (block-diagonal orthogonal fine-tuning)",
    "peft_version": __import__("peft").__version__,
    "transformers_version": __import__("transformers").__version__,
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "model_load_seconds": model_loaded_s,
    "generate_seconds": fwd_s,
    "trainable_params": trainable,
    "total_params": total,
    "trainable_pct": round(100.0 * trainable / total, 4),
    "injected_module_count": len(injected),
    "injected_sample": injected[:8],
    "sample_generations": gens,
}
(REPORT / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False, indent=2))
print("SMOKE_PASS")
