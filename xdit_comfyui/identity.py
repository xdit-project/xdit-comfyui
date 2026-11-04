"""Canonical product identity shared across ComfyUI and package surfaces."""

DISPLAY_NAME = "xDiT"
COMFY_CATEGORY = DISPLAY_NAME
DISTRIBUTION_NAME = "xdit-comfyui"
TEMPLATE_MODULE = "xdit_comfyui"
USER_AGENT = f"{DISTRIBUTION_NAME}/1.0"

PRESET_NODE_ID = "xDiT.Preset"
MODEL_NODE_ID = "xDiT.Model"
SAMPLE_NODE_ID = "xDiT.Sample"
SAVE_IMAGE_NODE_ID = "xDiT.SaveImage"
SAVE_VIDEO_NODE_ID = "xDiT.SaveVideo"

NODE_DISPLAY_NAMES = {
    PRESET_NODE_ID: "xDiT Preset",
    MODEL_NODE_ID: "xDiT Model",
    SAMPLE_NODE_ID: "xDiT Sample",
    SAVE_IMAGE_NODE_ID: "Save xDiT Image",
    SAVE_VIDEO_NODE_ID: "Save xDiT Video",
}

# Output order is part of ComfyUI's persisted graph contract.
NODE_OUTPUTS = {
    PRESET_NODE_ID: (("model", "*"), ("images", "IMAGE"), ("sample", "*")),
    MODEL_NODE_ID: (("model", "*"),),
    SAMPLE_NODE_ID: (("images", "IMAGE"), ("video", "VIDEO"), ("result", "*")),
    SAVE_IMAGE_NODE_ID: (),
    SAVE_VIDEO_NODE_ID: (),
}

SAMPLE_IMAGE_OUTPUT = 0
SAMPLE_VIDEO_OUTPUT = 1
SAMPLE_RESULT_OUTPUT = 2
