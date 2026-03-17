"""Import RunPod endpoint modules so Flash can discover both workers from one place."""

from runpod_workers.lora_train.endpoint import lora_train
from runpod_workers.nova_generate.endpoint import nova_generate

__all__ = ["nova_generate", "lora_train"]

