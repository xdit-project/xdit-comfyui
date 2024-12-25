import comfy
import logging
import socket
import torch
import numpy as np

from comfy import model_management
from comfy import model_detection
from comfy.sd import CLIP, VAE
from comfy import clip_vision
from .executor.executor import FluxExecutor, UNetExecutor
from .model.model_patcher import CustomModelPatcher

# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

def load_diffusion_model_state_dict(sd, model_options={}, load_weights=False): #load unet in diffusers or regular format
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

    model.diffusion_model = FluxExecutor(**unet_config, device=None, operations=operations)
    if load_weights:
        model.load_model_weights(new_sd, "")
        left_over = sd.keys()
        if len(left_over) > 0:
            logging.info("left over keys in unet: {}".format(left_over))
    print(f"Load_device: {load_device}, Offload_device: {offload_device}")
    print(model)
    
    def sd_size(sd):
        module_mem = 0
        for k in sd:
            t = sd[k]
            module_mem += t.nelement() * t.element_size()
        return module_mem

    return CustomModelPatcher(model, load_device=load_device, offload_device=offload_device, size=sd_size(sd))

def load_diffusion_model(unet_path, model_options={}):
    sd = comfy.utils.load_torch_file(unet_path)
    
    model = load_diffusion_model_state_dict(sd, model_options=model_options, load_weights=False)
    model.model.diffusion_model.load_state_dict_from_file(unet_path)
    
    if model is None:
        logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(unet_path))
    return model


def load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True, model_options={}, te_model_options={}):
    sd = comfy.utils.load_torch_file(ckpt_path)
    out = load_state_dict_guess_config(sd, output_vae, output_clip, output_clipvision, embedding_directory, output_model, model_options, te_model_options=te_model_options)
    if out is None:
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(ckpt_path))
    return out

def load_state_dict_guess_config(sd, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True, model_options={}, te_model_options={}):
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None

    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    parameters = comfy.utils.calculate_parameters(sd, diffusion_model_prefix)
    weight_dtype = comfy.utils.weight_dtype(sd, diffusion_model_prefix)
    load_device = model_management.get_torch_device()

    model_config = model_detection.model_config_from_unet(sd, diffusion_model_prefix)
    if model_config is None:
        return None

    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if weight_dtype is not None and model_config.scaled_fp8 is None:
        unet_weight_dtype.append(weight_dtype)

    model_config.custom_operations = model_options.get("custom_operations", None)
    unet_dtype = model_options.get("dtype", model_options.get("weight_dtype", None))

    if unet_dtype is None:
        unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype)

    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if model_config.clip_vision_prefix is not None:
        if output_clipvision:
            clipvision = clip_vision.load_clipvision_from_sd(sd, model_config.clip_vision_prefix, True)

    if output_model:
        inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
        model_config.unet_config["disable_unet_model_creation"] = True
        model = model_config.get_model(sd, diffusion_model_prefix, device=inital_load_device)

        # Add Executor Logic
        unet_config = model_config.unet_config
        del unet_config["disable_unet_model_creation"]
        manual_cast_dtype = model_config.manual_cast_dtype
        if model_config.custom_operations is None:
            operations = comfy.ops.pick_operations(unet_config.get("dtype", None), manual_cast_dtype)
        else:
            operations = model_config.custom_operations

        model.diffusion_model = UNetExecutor(**unet_config, device=None, operations=operations)
        # model = model_config.get_model(sd, diffusion_model_prefix, device=inital_load_device)
        model.load_model_weights(sd, diffusion_model_prefix)

    if output_vae:
        vae_sd = comfy.utils.state_dict_prefix_replace(sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True)
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        vae = VAE(sd=vae_sd)

    if output_clip:
        clip_target = model_config.clip_target(state_dict=sd)
        if clip_target is not None:
            clip_sd = model_config.process_clip_state_dict(sd)
            if len(clip_sd) > 0:
                parameters = comfy.utils.calculate_parameters(clip_sd)
                clip = CLIP(clip_target, embedding_directory=embedding_directory, tokenizer_data=clip_sd, parameters=parameters, model_options=te_model_options)
                m, u = clip.load_sd(clip_sd, full_model=True)
                if len(m) > 0:
                    m_filter = list(filter(lambda a: ".logit_scale" not in a and ".transformer.text_projection.weight" not in a, m))
                    if len(m_filter) > 0:
                        logging.warning("clip missing: {}".format(m))
                    else:
                        logging.debug("clip missing: {}".format(m))

                if len(u) > 0:
                    logging.debug("clip unexpected {}:".format(u))
            else:
                logging.warning("no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.")

    left_over = sd.keys()
    if len(left_over) > 0:
        logging.debug("left over keys: {}".format(left_over))

    if output_model:
        model_patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=model_management.unet_offload_device())
        if inital_load_device != torch.device("cpu"):
            logging.info("loaded straight to GPU")
            model_management.load_models_gpu([model_patcher], force_full_load=True)

    return (model_patcher, clip, vae, clipvision)

def tensor_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        cpu_tensor = tensor.contiguous().cpu()
        return cpu_tensor.numpy()
    else:
        return tensor

def numpy_to_tensor(numpy_array, device=torch.device("cuda"), dtype=torch.float32):
    if isinstance(numpy_array, np.ndarray):
        return torch.from_numpy(numpy_array).to(device=device, dtype=dtype)
    else:
        return numpy_array

has_patched_for_torch_compile = False

def patch_for_torch_compile():
    global has_patched_for_torch_compile
    if has_patched_for_torch_compile:
        return

    import sys
    from torch._inductor.compile_worker.subproc_pool import SubprocPool

    old_subproc_pool_init = SubprocPool.__init__

    # https://github.com/comfyanonymous/ComfyUI/issues/4196
    def new_subproc_pool_init(self, *args, **kwargs):
        to_insert = None
        for i, p in enumerate(sys.path):
            if p.endswith("comfy"):
                to_insert = (i, p)
                break
        if to_insert is not None:
                sys.path.pop(to_insert[0])
        try:
            old_subproc_pool_init(self, *args, **kwargs)
        finally:
            if to_insert is not None:
                sys.path.insert(to_insert[0], to_insert[1])

    SubprocPool.__init__ = new_subproc_pool_init

    has_patched_for_torch_compile = True