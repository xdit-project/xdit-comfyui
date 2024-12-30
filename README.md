# xdit-comfyui-private
## Environment
### Install ComfyUI
Follow the [ComfyUI](https://github.com/comfyanonymous/ComfyUI) repo to install the dependencies.

You can simply install the dependencies by running the following command:
```bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
pip install -r requirements.txt
```

### Install [Ray](https://docs.ray.io/en/latest/ray-overview/installation.html)
```bash
pip install -U "ray[data,train,tune,serve]"
```

### Install [yunchang](https://github.com/feifeibear/long-context-attention)
```bash
git clone git@github.com:feifeibear/long-context-attention.git
cd long-context-attention
pip install ninja
pip install .
```

### Put the xdit-comfyui-private folder into the ComfyUI/custom_nodes folder
You can put the xdit-comfyui-private folder into the ComfyUI/custom_nodes folder by running the following command:
```bash
cd ${ComfyUI}/custom_nodes
git clone git@github.com:xdit-project/xdit-comfyui-private.git
cd xdit-comfyui-private
pip install -e .
```

## Prepare models checkpoint
Please follow the [Flux Examples](https://comfyanonymous.github.io/ComfyUI_examples/flux/) to prepare the corresponding checkpoints:
1. Put in the `${ComfyUI}/models/clip` folder: https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main
2. Put in the `${ComfyUI}/models/vae` folder: https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/ae.safetensors
3. Put in the `${ComfyUI}/models/unet` folder: https://huggingface.co/black-forest-labs/FLUX.1-dev

## Run the demo
You can run the demo by running the following command:
```bash
cd ${ComfyUI}
python main.py --disable-cuda-malloc
```

The option `--disable-cuda-malloc` is essential to ensure that ComfyUI can accurately manage the loading and offloading processes of our parallelized models.

**You can load the default workflow in the xdit-comfyui-private/workflows folder: `xdit-flux1-dev.json`**

## Loras Supports
We also provide some loras for you to try. You can put the loras into the `${ComfyUI}/models/xdit/loras` folder(the folder will be created automatically if not exists). Currently, we support the following loras:
1. [flux-RealismLora](https://huggingface.co/XLabs-AI/flux-RealismLora)
2. [flux-lora-collection](https://huggingface.co/XLabs-AI/flux-lora-collection)
3. [flux-furry-lora](https://huggingface.co/XLabs-AI/flux-furry-lora)

**You can load the example loras workflow in the xdit-comfyui-private/workflows folder: `xdit-flux1-dev-loras.json`**

## FP8 Supports
We also provide some FP8 models for you to try if you don't have enough VRAM. You can put the FP8 models into the `${ComfyUI}/models/unet` folder. Download the FP8 models from [here](https://huggingface.co/Comfy-Org/flux1-dev/blob/main/flux1-dev-fp8.safetensors).

Use the `xdit-flux1-dev-fp8.json` workflow in the ComfyUI/workflows folder to try the FP8 models.

## ControlNet Supports
We also provide some ControlNet for you to try. Before you use this function, you need to install the `x-flux-comfyui` node in the ComfyUI from [here](https://github.com/XLabs-AI/x-flux-comfyui/tree/main). Please follow the [instruction](https://github.com/XLabs-AI/x-flux-comfyui/tree/main?tab=readme-ov-file#installation) in the `x-flux-comfyui` repo to install the node.

Currently, we support the following ControlNet:
1. [flux-controlnet-canny] (https://huggingface.co/XLabs-AI/flux-controlnet-canny)

Use the `xdit-flux1-dev-controlnet-canny.json` workflow in the ComfyUI/workflows folder to try the ControlNet.