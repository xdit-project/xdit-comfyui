import ray
import torch
import torch.distributed as dist
import os
import gc
import comfy
import time
import numpy as np
from comfy import model_detection
from xdit_comfyui_private.model.flux.flux import xFuserFlux
from xdit_comfyui_private.distributed.parallel_state import init_distributed_enviroment, init_model_parallel
from xdit_comfyui_private.modules.loras.utils import load_flux_lora, check_is_comfy_lora, comfy_to_xlabs_lora
from xdit_comfyui_private.modules.loras.layers import DoubleStreamBlockLoraProcessor, DoubleStreamBlockLorasMixerProcessor

from xdit_comfyui_private.model.sd.unet import xFuserUnet

class RayWorker:
    def __init__(self, **kwargs):
        self.world_size = kwargs.pop('world_size', 1)
        self.ulysses_degree = kwargs.pop('ulysses_degree', 1)
        self.ring_degree = kwargs.pop('ring_degree', 1)
        rank = int(ray.get_gpu_ids()[0]) % self.world_size
        self.rank = self.local_rank = rank
        self.device = "cuda:0"
        init_distributed_enviroment(kwargs.pop('distributed_init_method', 'env://'), self.world_size, self.rank)
        init_model_parallel(self.ulysses_degree, self.ring_degree, self.rank, self.world_size)
        self.flux_worker = None
        self.unet_worker = None

    def initialize_flux(self, *args, **kwargs):
        self.flux_worker = FluxWorker(**kwargs)

    def initialize_unet(self, *args, **kwargs):
        self.unet_worker = UNetWorker(**kwargs)

    def execute_method(self, method, *args, **kwargs):
        return getattr(self, method)(*args, **kwargs)

    def execute_flux_method(self, method, *args, **kwargs):
        return getattr(self.flux_worker, method)(*args, **kwargs)

    def execute_unet_method(self, method, *args, **kwargs):
        return getattr(self.unet_worker, method)(*args, **kwargs)

class UNetWorker:
    def __init__(self, **kwargs):
        self.world_size = 2
        self.rank = self.local_rank = int(ray.get_gpu_ids()[0]) % self.world_size
        self.device = "cuda:0"
        self.unet = xFuserUnet(**kwargs)
        self.unet = self.unet.to('cpu')
        self.is_compiled = False

    def execute_method(self, method, *args, **kwargs):
        return getattr(self, method)(*args, **kwargs)

    def forward(self, x, timestep, context, y, control=None, transformer_options={}, dtype=torch.float16, use_tensor_to_numpy=False, **kwargs):
        from xdit_comfyui_private.utils import tensor_to_numpy, numpy_to_tensor
        time_start = time.time()
        '''if not self.is_compiled:
            print("Compiling UNet")
            from xdit_comfyui_private.utils import patch_for_torch_compile
            patch_for_torch_compile()
            self.unet.to(dtype=dtype, memory_format=torch.channels_last)
            self.unet = torch.compile(self.unet, mode="max-autotune-no-cudagraphs")
            self.is_compiled = True
            torch.cuda.synchronize()'''
        
        if use_tensor_to_numpy:
            x = numpy_to_tensor(x, dtype=dtype)
            timestep = numpy_to_tensor(timestep, dtype=dtype)
            context = numpy_to_tensor(context, dtype=dtype)
            y = numpy_to_tensor(y, dtype=dtype)

        bs, c, h, w = x.shape

        with torch.no_grad():

            if bs % self.world_size == 0 and self.world_size > 1:
                start_idx = self.rank * (bs // self.world_size)
                end_idx = start_idx + (bs // self.world_size)
                x = x[start_idx:end_idx]
                timestep = timestep[start_idx:end_idx]
                y = y[start_idx:end_idx]
                context = context[start_idx:end_idx]
                output= self.unet(x, timestep, context, y, control, transformer_options, **kwargs).contiguous()
                out_list = [
                    torch.empty_like(output) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(out_list, output)
                output = torch.cat(out_list, dim=0)
            else:
                output = self.unet(x, timestep, context, y, control, transformer_options, **kwargs)
        
        if use_tensor_to_numpy:
            output = tensor_to_numpy(output)
        
        torch.cuda.synchronize()
        time_end = time.time()
        print(f"[RANK {self.rank}] UNet forward time(in worker): {time_end - time_start:.4f} seconds")
        return output
    
    def load_state_dict(self, sd, strict=False):
        m, u = self.unet.load_state_dict(sd, strict=strict)
        return m, u

    def to_gpu(self):
        self.unet = self.unet.to(torch.device('cuda:0'))
        
    def to_cpu(self):
        self.unet = self.unet.to(torch.device('cpu'))

class FluxWorker:
    def __init__(self, **kwargs):
        self.device = 'cuda:0'
        self.flux = xFuserFlux(**kwargs)
        self.flux = self.flux.to('cpu')
        self.lora_processors_dict = {}

    def forward_orig(self, img, img_ids, txt, txt_ids, timesteps, y, guidance, control=None, neg_mode=None, block_controlnet_hidden_states=None, block_controlnet_hidden_states_npy=None, **kwargs):
        with torch.no_grad():
            if block_controlnet_hidden_states is not None:
                pass
            elif block_controlnet_hidden_states_npy is not None:
                block_controlnet_hidden_states = []
                for npy in block_controlnet_hidden_states_npy:
                    state = torch.from_numpy(npy).to(dtype=img.dtype).to(self.device)
                    block_controlnet_hidden_states.append(state)
            out = self.flux.forward_orig(img, 
                                            img_ids, 
                                            txt, 
                                            txt_ids, 
                                            timesteps, 
                                            y, 
                                            guidance, 
                                            control, 
                                            neg_mode, 
                                            block_controlnet_hidden_states)
            
            if dist.get_world_size() > 1:
                out_list = [
                    torch.empty_like(out) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(out_list, out)
                out = torch.cat(out_list, dim=1)

        return out

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

    def clean_cache(self):
        torch.cuda.empty_cache()
        gc.collect()

    def clean_lora(self):
        for double_block in self.flux.double_blocks:
            double_block.set_lora_processor(None)
        self.lora_processors_dict = {}

    def to_gpu(self):
        self.flux = self.flux.to(torch.device('cuda:0'))
        
    def to_cpu(self):
        self.flux = self.flux.to(torch.device('cpu'))
