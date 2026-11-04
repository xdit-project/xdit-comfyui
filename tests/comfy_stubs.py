"""Small ComfyUI module stubs shared by unit and integration tests."""

import types


def comfy_video_modules():
    class VideoComponents:
        def __init__(self, images, frame_rate):
            self.images = images
            self.frame_rate = frame_rate

    class VideoFromComponents:
        def __init__(self, components):
            self.components = components

    names = (
        "comfy_api",
        "comfy_api.latest",
        "comfy_api.latest._input_impl",
        "comfy_api.latest._input_impl.video_types",
        "comfy_api.latest._util",
        "comfy_api.latest._util.video_types",
    )
    modules = {name: types.ModuleType(name) for name in names}
    modules["comfy_api.latest._input_impl.video_types"].VideoFromComponents = VideoFromComponents
    modules["comfy_api.latest._util.video_types"].VideoComponents = VideoComponents
    return modules, VideoFromComponents
