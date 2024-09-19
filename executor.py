import ray
from .worker import FluxWorker

class FluxExecutor:
    def __init__(self, **kwargs):
        ray.init(ignore_reinit_error=True)
        self.worker = FluxWorker.remote(**kwargs)
        
        self.dtype = kwargs['dtype']

    def __call__(self, x, timestep, context, y, guidance, control=None, **kwargs):
        result_ref = self.worker.forward.remote(x, timestep, context, y, guidance, **kwargs)
        result = ray.get(result_ref).to("cuda:0")
        del result_ref
        
        return result
        
    def load_state_dict(self, sd, strict=False):
        return ray.get(self.worker.load_state_dict.remote(sd, strict=strict))

    def state_dict(self):
        return ray.get(self.worker.state_dict.remote())

    def load_lora(self, lora_path, strength_model):
        return ray.get(self.worker.load_lora.remote(lora_path, strength_model))