import torch
import comfy.utils
from safetensors import safe_open

def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def load_flux_lora(path):
    if path is not None:
        if '.safetensors' in path:
            checkpoint = load_safetensors(path)
        else:
            checkpoint = torch.load(path, map_location='cpu')
    else:
        checkpoint = None
        print("Invalid path")
    a1 = sorted(list(checkpoint[list(checkpoint.keys())[0]].shape))[0]
    a2 = sorted(list(checkpoint[list(checkpoint.keys())[1]].shape))[0]
    if a1==a2:
        return checkpoint, int(a1)
    return checkpoint, 16

def load_comfy_lora(path):
    lora = comfy.utils.load_torch_file(path, safe_load=True)
    
    def convert_to_flux_lora(lora_key):
        # flux_lora_key = lora_key.replace("lora_unet_", "diffusion_model.")
        flux_lora_key = lora_key.replace("lora_unet_", "")
        if "double_blocks" in lora_key:
            flux_lora_key = flux_lora_key.replace("double_blocks_", "double_blocks.")
            flux_lora_key = flux_lora_key.replace("_img_attn_", ".img_attn.")
            flux_lora_key = flux_lora_key.replace("_img_mlp_", ".img_mlp.")
            flux_lora_key = flux_lora_key.replace("_img_mod_", ".img_mod.")
            flux_lora_key = flux_lora_key.replace("_txt_attn_", ".txt_attn.")
            flux_lora_key = flux_lora_key.replace("_txt_mlp_", ".txt_mlp.")
            flux_lora_key = flux_lora_key.replace("_txt_mod_", ".txt_mod.")
        elif "single_blocks" in lora_key:
            flux_lora_key = flux_lora_key.replace("single_blocks_", "single_blocks.")
            flux_lora_key = flux_lora_key.replace("_linear1", ".linear1")
            flux_lora_key = flux_lora_key.replace("_linear2", ".linear2")
            flux_lora_key = flux_lora_key.replace("_modulation_", ".modulation.")
        return flux_lora_key    
    
    def merge_lora_keys(original_dict):
        merged_dict = {}
        
        for key in original_dict.keys():
            parts = key.split('.')
            if parts[-1] == 'weight' and parts[-2] in ['lora_down', 'lora_up']:
                base_key = '.'.join(parts[:-2])
                sub_key = parts[-2]
                
                if base_key not in merged_dict:
                    merged_dict[base_key] = {}
                merged_dict[base_key][sub_key] = original_dict[key]
            elif parts[-1] == 'alpha':
                base_key = '.'.join(parts[:-1])
                sub_key = parts[-1]
                if base_key not in merged_dict:
                    merged_dict[base_key] = {}
                merged_dict[base_key][sub_key] = original_dict[key]
            else:
                merged_dict[key] = original_dict[key]
        
        return merged_dict

    flux_lora = {}
    for key, value in lora.items():
        flux_lora_key = convert_to_flux_lora(key)
        flux_lora[flux_lora_key] = value
    flux_lora_merged = merge_lora_keys(flux_lora)

    return flux_lora_merged

def apply_lora(weight, lora_value, strength=1.0, intermediate_dtype=torch.float32):
    assert "lora_down" in lora_value and "lora_up" in lora_value, "lora_down and lora_up must be in lora_value"
    rank = lora_value["lora_down"].shape[0]
    mat_down = comfy.model_management.cast_to_device(lora_value["lora_down"], weight.device, intermediate_dtype)
    mat_up = comfy.model_management.cast_to_device(lora_value["lora_up"], weight.device, intermediate_dtype)
    if "alpha" in lora_value:
        alpha = lora_value["alpha"] / rank
    else:
        alpha = 1.0

    lora_diff = torch.mm(mat_up.flatten(start_dim=1), mat_down.flatten(start_dim=1)).reshape(weight.shape)
    weight += ((strength * alpha) * lora_diff).type(weight.dtype)
    return weight

def check_is_comfy_lora(sd):
    for k in sd:
        if "lora_down" in k or "lora_up" in k:
            return True
    return False

def comfy_to_xlabs_lora(sd):
    sd_out = {}
    for k in sd:
        if "diffusion_model" in k:
            new_k =  (k
                    .replace(".lora_down.weight", ".down.weight")
                    .replace(".lora_up.weight", ".up.weight")
                    .replace(".img_attn.proj.", ".processor.proj_lora1.")
                    .replace(".txt_attn.proj.", ".processor.proj_lora2.")
                    .replace(".img_attn.qkv.", ".processor.qkv_lora1.")
                    .replace(".txt_attn.qkv.", ".processor.qkv_lora2."))
            new_k = new_k[len("diffusion_model."):]
        else:
            new_k=k
        sd_out[new_k] = sd[k]
    return sd_out