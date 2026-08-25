# merge_lora.py의 --lora-full-model(백본 포함) 버전.
import copy
import shutil
from pathlib import Path

from peft import PeftModel

from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead

CHECKPOINT = Path("./cupstack-checkpoints-fulllora/checkpoint-30000").expanduser()
BASE_MODEL = "nvidia/GR00T-N1.5-3B"
OUT_DIR = Path("./cupstack-checkpoints-fulllora/checkpoint-30000-merged").expanduser()
ACTION_HORIZON = 10  # unchanged from the first run -- this is the H10 config that pi0 also hit 80%+ with

print(f"Loading base model {BASE_MODEL}")
model = GR00T_N1_5.from_pretrained(
    BASE_MODEL, tune_llm=False, tune_visual=False, tune_projector=True, tune_diffusion_model=True
)

new_cfg = copy.deepcopy(model.action_head.config)
new_cfg.action_horizon = ACTION_HORIZON
new_head = FlowmatchingActionHead(new_cfg)
new_head.load_state_dict(model.action_head.state_dict(), strict=False)
model.action_head = new_head
model.config.action_horizon = ACTION_HORIZON
model.action_horizon = ACTION_HORIZON
model.config.action_head_cfg["action_horizon"] = ACTION_HORIZON

model.compute_dtype = "bfloat16"
model.config.compute_dtype = "bfloat16"

print(f"Loading LoRA adapter from {CHECKPOINT}")
model = PeftModel.from_pretrained(model, str(CHECKPOINT))

print("Merging adapter into base weights")
model = model.merge_and_unload()

print(f"Saving merged model to {OUT_DIR}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT_DIR)

src_cfg = CHECKPOINT / "experiment_cfg"
if src_cfg.exists():
    shutil.copytree(src_cfg, OUT_DIR / "experiment_cfg", dirs_exist_ok=True)

print("Done.")
