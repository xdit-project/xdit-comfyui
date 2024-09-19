import time
import os
import torch
import folder_paths

from .utils import load_diffusion_model

dir_xlabs = os.path.join(folder_paths.models_dir, "xlabs")
os.makedirs(dir_xlabs, exist_ok=True)
dir_xlabs_loras = os.path.join(dir_xlabs, "loras")
os.makedirs(dir_xlabs_loras, exist_ok=True)

folder_paths.folder_names_and_paths["xlabs"] = ([dir_xlabs], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["xlabs_loras"] = ([dir_xlabs_loras], folder_paths.supported_pt_extensions)


def cleanprint(a):
    print(a)
    return a

class XfuserUNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                              "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "XDiT"

    def load_unet(self, unet_name, weight_dtype):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        model = load_diffusion_model(unet_path, model_options=model_options)
        return (model,)

def print_if_not_empty(a):
    b = list(a.items())
    if len(b)<1:
        return "{}"
    return b[0]
class XfuserFluxLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "lora_name": (cleanprint(folder_paths.get_filename_list("xlabs_loras")), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                              }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "loadmodel"
    CATEGORY = "XDiT"

    def loadmodel(self, model, lora_name, strength_model):
        model.model.diffusion_model.load_lora(os.path.join(dir_xlabs_loras, lora_name), strength_model)
        return (model,)

NODE_CLASS_MAPPINGS = {
    "XfuserUNETLoader": XfuserUNETLoader,
    "XfuserFluxLoraLoader": XfuserFluxLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XfuserUNETLoader": "Load Diffusion Model(Xfuser)",
    "XfuserFluxLoraLoader": "Load Flux Lora(Xfuser)",
}
