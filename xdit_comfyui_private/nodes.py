import time
import os
import torch
import folder_paths
import comfy
import latent_preview

from .utils import load_diffusion_model

dir_xdit = os.path.join(folder_paths.models_dir, "xdit")
os.makedirs(dir_xdit, exist_ok=True)
dir_xdit_loras = os.path.join(dir_xdit, "loras")
os.makedirs(dir_xdit_loras, exist_ok=True)

folder_paths.folder_names_and_paths["xdit"] = ([dir_xdit], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["xdit_loras"] = ([dir_xdit_loras], folder_paths.supported_pt_extensions)


def cleanprint(a):
    # print(a)
    return a

class XDiTUNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                              "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "XDiTNodes"

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

class XDiTFluxLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "lora_name": (cleanprint(folder_paths.get_filename_list("xdit_loras")), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                              }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "loadmodel"
    CATEGORY = "XDiTNodes"

    def loadmodel(self, model, lora_name, strength_model):
        bi = model.clone()
        lora_path = os.path.join(dir_xdit_loras, lora_name)
        bi.lora_cache[lora_path] = strength_model
        print(f"Applying Lora: {lora_path} with strength {strength_model}")
        bi.model.diffusion_model.load_lora(lora_path, strength_model)
        return (bi,)

class XDiTSamplerCustomAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"noise": ("NOISE", ),
                    "guider": ("GUIDER", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),
                     }
                }

    RETURN_TYPES = ("LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output")

    FUNCTION = "sample"

    CATEGORY = "XDiTNodes"

    def sample(self, noise, guider, sampler, sigmas, latent_image):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        latent["samples"] = latent_image

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # if hasattr(guider.model_patcher.model.diffusion_model, 'clean_lora'):
        #     guider.model_patcher.model.diffusion_model.clean_lora()
        #     for lora_path, strength_model in guider.model_patcher.lora_cache.items():
        #         print(f"Applying Lora: {lora_path} with strength {strength_model}")
        #         guider.model_patcher.model.diffusion_model.load_lora(lora_path, strength_model)

        samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        
        return (out, out_denoised)

NODE_CLASS_MAPPINGS = {
    "XDiTUNETLoader": XDiTUNETLoader,
    "XDiTFluxLoraLoader": XDiTFluxLoraLoader,
    "XDiTSamplerCustomAdvanced": XDiTSamplerCustomAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XDiTUNETLoader": "XDiTUNETLoader",
    "XDiTFluxLoraLoader": "XDiTFluxLoraLoader",
    "XDiTSamplerCustomAdvanced": "XDiTSamplerCustomAdvanced",
}