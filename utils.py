import comfy
import logging
import torch
import copy

from safetensors import safe_open
from comfy import model_management
from comfy import model_detection
from layers import (DoubleStreamBlockLoraProcessor,
                    DoubleStreamMixerProcessor,
                    DoubleStreamBlockProcessor)
from layers import DoubleStreamBlock as DSBnew
from comfy.ldm.flux.layers import DoubleStreamBlock as DSBold

def load_diffusion_model_state_dict(sd, model_options={}): #load unet in diffusers or regular format
    dtype = model_options.get("dtype", None)

    #Allow loading unets from checkpoint files
    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd

    parameters = comfy.utils.calculate_parameters(sd)
    load_device = model_management.get_torch_device()
    model_config = model_detection.model_config_from_unet(sd, "")

    if model_config is not None:
        new_sd = sd
    else:
        new_sd = model_detection.convert_diffusers_mmdit(sd, "")
        if new_sd is not None: #diffusers mmdit
            model_config = model_detection.model_config_from_unet(new_sd, "")
            if model_config is None:
                return None
        else: #diffusers unet
            model_config = model_detection.model_config_from_diffusers_unet(sd)
            if model_config is None:
                return None

            diffusers_keys = comfy.utils.unet_to_diffusers(model_config.unet_config)

            new_sd = {}
            for k in diffusers_keys:
                if k in sd:
                    new_sd[diffusers_keys[k]] = sd.pop(k)
                else:
                    logging.warning("{} {}".format(diffusers_keys[k], k))

    offload_device = model_management.unet_offload_device()
    if dtype is None:
        unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=model_config.supported_inference_dtypes)
    else:
        unet_dtype = dtype

    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model_config.custom_operations = model_options.get("custom_operations", None)
    model_config.unet_config["disable_unet_model_creation"] = True
    model = model_config.get_model(new_sd, "")

    # Add Executor Logic
    unet_config = model_config.unet_config
    del unet_config["disable_unet_model_creation"]
    manual_cast_dtype = model_config.manual_cast_dtype
    if model_config.custom_operations is None:
        operations = comfy.ops.pick_operations(unet_config.get("dtype", None), manual_cast_dtype)
    else:
        operations = model_config.custom_operations
    
    from .executor import FluxExecutor
    model.diffusion_model = FluxExecutor(**unet_config, device=None, operations=operations)
    model.load_model_weights(new_sd, "")
    left_over = sd.keys()
    if len(left_over) > 0:
        logging.info("left over keys in unet: {}".format(left_over))
    print(f"Load_device: {load_device}, Offload_device: {offload_device}")
    print(model)
    return comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)


def load_diffusion_model(unet_path, model_options={}):
    sd = comfy.utils.load_torch_file(unet_path)
    model = load_diffusion_model_state_dict(sd, model_options=model_options)
    if model is None:
        logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(unet_path))
    return model


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

def merge_loras(lora1, lora2):
    new_block = DoubleStreamMixerProcessor()
    if isinstance(lora1, DoubleStreamMixerProcessor):
        new_block.set_loras(*lora1.get_loras())
        new_block.set_ip_adapters(lora1.get_ip_adapters())
    elif isinstance(lora1, DoubleStreamBlockLoraProcessor):
        new_block.add_lora(lora1)
    else:
        pass
    if isinstance(lora2, DoubleStreamMixerProcessor):
        new_block.set_loras(*lora2.get_loras())
        new_block.set_ip_adapters(lora2.get_ip_adapters())
    elif isinstance(lora2, DoubleStreamBlockLoraProcessor):
        new_block.add_lora(lora2)
    else:
        pass
    return new_block

def attn_processors(model_flux):
    # set recursively
    processors = {}

    def fn_recursive_add_processors(name: str, module: torch.nn.Module, procs):
        if hasattr(module, "set_processor"):
            procs[f"{name}.processor"] = module.processor
            print(f"{name} have set_processor")
        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, procs)

        return procs

    for name, module in model_flux.named_children():
        fn_recursive_add_processors(name, module, processors)
    return processors

def CopyDSB(oldDSB):

    if isinstance(oldDSB, DSBold):
        tyan = copy.copy(oldDSB)

        if hasattr(tyan.img_mlp[0], 'out_features'):
            mlp_hidden_dim = tyan.img_mlp[0].out_features
        else:
            mlp_hidden_dim = 12288

        mlp_ratio = mlp_hidden_dim / tyan.hidden_size
        bi = DSBnew(hidden_size=tyan.hidden_size, num_heads=tyan.num_heads, mlp_ratio=mlp_ratio)
        #better use __dict__ but I bit scared
        (
            bi.img_mod, bi.img_norm1, bi.img_attn, bi.img_norm2,
            bi.img_mlp, bi.txt_mod, bi.txt_norm1, bi.txt_attn, bi.txt_norm2, bi.txt_mlp
        ) = (
            tyan.img_mod, tyan.img_norm1, tyan.img_attn, tyan.img_norm2,
            tyan.img_mlp, tyan.txt_mod, tyan.txt_norm1, tyan.txt_attn, tyan.txt_norm2, tyan.txt_mlp
        )
        bi.set_processor(DoubleStreamBlockProcessor())

        return bi
    return oldDSB

def FluxUpdateModules(flux_model, pbar=None):
    save_list = {}
    count = len(flux_model.double_blocks)
    patches = {}

    for i in range(count):
        flux_model.double_blocks[i]=CopyDSB(flux_model.double_blocks[i])