import os
import json
import torch
import shutil
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def merge_dora_robust(base_model_dir, adapter_dir, output_dir):
    print("Loading Adapter tensors...")
    adapter_tensors = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        lora_config = json.load(f)
    
    scaling = lora_config["lora_alpha"] / lora_config["r"]
    
    # Map adapter keys to base keys
    # Adapter: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    # Base: model.language_model.layers.0.self_attn.q_proj.weight
    lora_map = {}
    for k, v in adapter_tensors.items():
        base_key = k.replace("base_model.model.", "").replace(".lora_A.", ".").replace(".lora_B.", ".").replace(".lora_magnitude_vector", "")
        # Adjust for Qwen3.5 naming discrepancy if any
        if not base_key.startswith("model.language_model") and "layers." in base_key:
            base_key = base_key.replace("model.", "model.language_model.")
        
        # Clean up double dots and weight suffix
        base_key = base_key.replace("..", ".").replace(".weight.weight", ".weight")
        if not base_key.endswith(".weight"): base_key += ".weight"

        lora_map.setdefault(base_key, {})
        if ".lora_A." in k: lora_map[base_key]["A"] = v.to(torch.float32)
        elif ".lora_B." in k: lora_map[base_key]["B"] = v.to(torch.float32)
        elif ".lora_magnitude_vector" in k: lora_map[base_key]["m"] = v.to(torch.float32)

    # Process base safetensors
    os.makedirs(output_dir, exist_ok=True)
    st_files = [f for f in os.listdir(base_model_dir) if f.endswith(".safetensors") and "index" not in f]
    
    for st_file in st_files:
        print(f"Processing {st_file}...")
        base_tensors = load_file(os.path.join(base_model_dir, st_file))
        new_tensors = {}
        merged_count = 0
        
        for k, w in tqdm(base_tensors.items()):
            if k in lora_map and "A" in lora_map[k] and "B" in lora_map[k]:
                A, B = lora_map[k]["A"], lora_map[k]["B"]
                m = lora_map[k].get("m")
                dtype = w.dtype
                w_fp32 = w.to(torch.float32)
                
                # W_new = W0 + (B @ A) * scale
                delta_w = (B @ A) * scaling
                merged_w = w_fp32 + delta_w
                
                # DoRA magnitude recovery
                if m is not None:
                    norm = torch.linalg.norm(merged_w, dim=1, keepdim=True)
                    merged_w = merged_w / (norm + 1e-6) * m.unsqueeze(-1)
                
                new_tensors[k] = merged_w.to(dtype)
                merged_count += 1
            else:
                new_tensors[k] = w
        
        save_path = os.path.join(output_dir, st_file)
        save_file(new_tensors, save_path)
        print(f"Saved {len(new_tensors)} tensors (Merged {merged_count}) to {st_file}")


    print("Copying configs from base model...")
    for f in os.listdir(base_model_dir):
        if not f.endswith(".safetensors"):
            shutil.copy(os.path.join(base_model_dir, f), os.path.join(output_dir, f))
    

if __name__ == "__main__":
    BASE = "/home/trongdz/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
    ADAPTER = "./cpt-checkpoints"
    OUT = "./merged-model-qwen3.5-finetuned"
    merge_dora_robust(BASE, ADAPTER, OUT)
