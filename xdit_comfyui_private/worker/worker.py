import ray
import torch
import torch.distributed as dist
import os
import gc
import comfy

from comfy import model_detection
from xdit_comfyui_private.model.flux.flux import xFuserFlux
from xdit_comfyui_private.distributed.parallel_state import init_distributed_enviroment, init_model_parallel
from xdit_comfyui_private.modules.loras.utils import load_flux_lora, load_comfy_lora, apply_lora, check_is_comfy_lora, comfy_to_xlabs_lora
from xdit_comfyui_private.modules.loras.layers import DoubleStreamBlockLoraProcessor, DoubleStreamBlockLorasMixerProcessor

class FluxWorker:
    modules_to_convert = []

    def __init__(self, **kwargs):
        self.world_size = kwargs.pop('world_size', 1)
        self.ulysses_degree = kwargs.pop('ulysses_degree', 1)
        self.ring_degree = kwargs.pop('ring_degree', 1)
        rank = int(ray.get_gpu_ids()[0]) % self.world_size
        self.rank = self.local_rank = rank
        self.device = "cuda:0"
        init_distributed_enviroment(kwargs.pop('distributed_init_method', 'env://'), self.world_size, self.rank)
        init_model_parallel(self.ulysses_degree, self.ring_degree, self.rank, self.world_size)
        self.flux = xFuserFlux(**kwargs).to(self.device)
        self.lora_processors_dict = {}


    def forward(self, x, timestep, context, y, guidance, control=None, **kwargs):
        with torch.no_grad():
            x_worker = x.to(self.device)
            timestep_worker = timestep.to(self.device)
            context_worker = context.to(self.device)
            y_worker = y.to(self.device)
            guidance_worker = guidance.to(self.device)
            configs_worker = {'transformer_options': {'cond_or_uncond': [0], 'sigmas': torch.tensor([1.], device=self.device)}}
            output = self.flux.forward(x_worker, timestep_worker, context_worker, y_worker, guidance_worker, control, **configs_worker)
        
        return output

    def load_state_dict(self, sd, strict=False):
        m, u = self.flux.load_state_dict(sd, strict=strict)
        return m, u

    def load_state_dict_from_file(self, unet_path):
        sd = comfy.utils.load_torch_file(unet_path)
        diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
        temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
        if len(temp_sd) > 0:
            sd = temp_sd
        return self.load_state_dict(sd, strict=False)

    # call after all the weight are loaded(including LoRa)
    def parallelize_model(self):
        self._parallelize_flux_model()
        self.flux.to(self.device)

    def state_dict(self):
        return self.flux.state_dict()

    def execute_method(self, method, *args, **kwargs):
        return getattr(self, method)(*args, **kwargs)

    def _parallelize_flux_model(self):
        
        pass

    def load_lora(self, lora_path, strength_model):
        try:
            self.load_lora_dynamic(lora_path, strength_model)
        except Exception as e:
            self.load_lora_static(lora_path, strength_model)

    def load_lora_dynamic(self, lora_path, strength_model):
        checkpoint, lora_rank = load_flux_lora(lora_path)
        
        self.lora_processors_dict[lora_path] = []
        lora_attn_procs = {}
        if check_is_comfy_lora(checkpoint):
            checkpoint = comfy_to_xlabs_lora(checkpoint)

        for idx, double_block in enumerate(self.flux.double_blocks):
            name = f"double_blocks.{idx}.processor"
            lora_processor = DoubleStreamBlockLoraProcessor(
                dim=3072, rank=lora_rank, lora_weight=strength_model)
            lora_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    lora_state_dict[k[len(name) + 1:]] = checkpoint[k]
            lora_processor.load_state_dict(lora_state_dict)
            lora_processor.to(self.device)
            self.lora_processors_dict[lora_path].append(lora_processor)
            
            loras_processor = DoubleStreamBlockLorasMixerProcessor()
            for lora_processors in self.lora_processors_dict.values():
                loras_processor.add_lora(lora_processors[idx])
            double_block.set_lora_processor(loras_processor)

    def load_lora_static(self, lora_path, strength_model):
        checkpoint = load_comfy_lora(lora_path)
        for key, value in checkpoint.items():
            model_key = key + ".weight"
            if model_key in self.flux.state_dict():
                weight = self.flux.state_dict()[model_key]
                temp_weight = weight.to(torch.float32, copy=True)
                out_weight = apply_lora(temp_weight, value, strength=strength_model, intermediate_dtype=torch.float32)
                out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype, seed=0)
                comfy.utils.copy_to_param(self.flux, model_key, out_weight)

    def clean_cache(self):
        torch.cuda.empty_cache()
        gc.collect()

    def clean_lora(self):
        for double_block in self.flux.double_blocks:
            double_block.set_lora_processor(None)
        self.lora_processors_dict = {}
