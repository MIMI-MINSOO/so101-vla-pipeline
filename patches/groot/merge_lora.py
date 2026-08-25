# Gr00tPolicy/inference_service.py는 config.json이 있는 완전한 모델 디렉토리만 로드
# 가능 -- adapter-only 체크포인트(adapter_config.json+adapter_model.safetensors)는 그대로
# 못 읽음. base 모델 + adapter를 병합해서 독립 체크포인트로 export.
# (action head-only LoRA 버전. 백본 포함 버전은 merge_lora_fulllora.py)
import copy
import shutil
from pathlib import Path

from peft import PeftModel

from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead

CHECKPOINT = Path("./cupstack-checkpoints/checkpoint-30000").expanduser()
BASE_MODEL = "nvidia/GR00T-N1.5-3B"
OUT_DIR = Path("./cupstack-checkpoints/checkpoint-30000-merged").expanduser()
ACTION_HORIZON = 10

model = GR00T_N1_5.from_pretrained(
    BASE_MODEL, tune_llm=False, tune_visual=False, tune_projector=True, tune_diffusion_model=True
)

# action_horizon을 학습 때와 동일하게 재구성해야 LoRA shape이 맞음
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

model = PeftModel.from_pretrained(model, str(CHECKPOINT))
model = model.merge_and_unload()

OUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT_DIR)

src_cfg = CHECKPOINT / "experiment_cfg"
if src_cfg.exists():
    shutil.copytree(src_cfg, OUT_DIR / "experiment_cfg", dirs_exist_ok=True)
