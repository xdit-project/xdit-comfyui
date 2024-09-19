import ray
import torch
import sys
import os
from comfy.ldm.flux.model import Flux, FluxParams


@ray.remote(num_gpus=1)
class FluxWorker:
    def __init__(self, **kwargs):
        self.device = "cuda:0"
        self.flux = Flux(**kwargs).to(self.device)

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

    def state_dict(self):
        return self.flux.state_dict()

    def load_lora(self, lora_path, strength_model):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        from utils import (load_flux_lora, check_is_comfy_lora, comfy_to_xlabs_lora, 
                            attn_processors, merge_loras, FluxUpdateModules)
        from layers import (DoubleStreamBlockLoraProcessor,
                            DoubleStreamMixerProcessor)
                            
        checkpoint, lora_rank = load_flux_lora(lora_path)

        FluxUpdateModules(self.flux)
        lora_attn_procs = {}
        if checkpoint is not None:
            if check_is_comfy_lora(checkpoint):
                checkpoint = comfy_to_xlabs_lora(checkpoint)
            for name, _ in attn_processors(self.flux).items():
                lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(
                    dim=3072, rank=lora_rank, lora_weight=strength_model)
                lora_state_dict = {}
                for k in checkpoint.keys():
                    if name in k:
                        lora_state_dict[k[len(name) + 1:]] = checkpoint[k]
                lora_attn_procs[name].load_state_dict(lora_state_dict)
                lora_attn_procs[name].to(self.device)
                tmp=DoubleStreamMixerProcessor()
                tmp.add_lora(lora_attn_procs[name])
                lora_attn_procs[name]=tmp

        for name, _ in attn_processors(self.flux).items():
            attribute = f"{name}"
            # if attribute in model.object_patches.keys():
            #     old = copy.copy((model.object_patches[attribute]))
            # else:
            #     old = None
            old = None
            lora = merge_loras(old, lora_attn_procs[name])


            attrs = attribute.split(".")
            obj = self.flux
            for name in attrs[:-1]:
                obj = getattr(obj, name)
            setattr(obj, attrs[-1], lora)

        print("Lora Loading Success!")
        print(self.flux)