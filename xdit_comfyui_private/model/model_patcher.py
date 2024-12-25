import copy
import torch
from comfy.model_patcher import ModelPatcher

class CustomModelPatcher(ModelPatcher):
    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        super().__init__(model, load_device, offload_device, size, weight_inplace_update)
        self.lora_cache = {}

    def clone(self):
        n = CustomModelPatcher(self.model, self.load_device, self.offload_device, self.size, weight_inplace_update=self.weight_inplace_update)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        n.lora_cache = copy.copy(self.lora_cache)
        return n

    def partially_unload(self, device_to, memory_to_free=0):
        with self.use_ejected():
            #unload
            self.model.diffusion_model.to_cpu()
            
            self.model.device = torch.device('cpu')
            memory_freed = self.model.model_loaded_weight_memory
            self.model.model_loaded_weight_memory = 0
            return memory_freed

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        with self.use_ejected():
            self.unpatch_hooks()
            #load
            self.model.diffusion_model.to_gpu()

            self.model.device = torch.device('cuda:0')
            self.model.model_loaded_weight_memory = self.size
            self.apply_hooks(self.forced_hooks, force_apply=True)