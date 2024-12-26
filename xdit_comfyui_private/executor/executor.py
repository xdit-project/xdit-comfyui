import time
from typing import Any, Dict, List, Optional, Tuple
from itertools import repeat
from logging import getLogger

import torch
import ray
import copy
import numpy as np
from ray.util.placement_group import PlacementGroup
from ray.util.placement_group import PlacementGroupSchedulingStrategy

from xdit_comfyui_private.worker.worker import RayWorker
from .utils import get_open_port, get_distributed_init_method

logger = getLogger(__name__)

PG_WAIT_TIMEOUT = 1800

_RAY_EXECUTOR = None

def _wait_until_pg_ready(current_placement_group: "PlacementGroup"):
    """Wait until a placement group is ready.

    It prints the informative log messages if the placement group is
    not created within time.

    """
    # Wait until PG is ready - this will block until all
    # requested resources are available, and will timeout
    # if they cannot be provisioned.
    placement_group_specs = current_placement_group.bundle_specs

    s = time.time()
    pg_ready_ref = current_placement_group.ready()
    wait_interval = 10
    while time.time() - s < PG_WAIT_TIMEOUT:
        ready, _ = ray.wait([pg_ready_ref], timeout=wait_interval)
        if len(ready) > 0:
            break

        # Exponential backoff for warning print.
        wait_interval *= 2
        logger.info(
            "Waiting for creating a placement group of specs for "
            "%d seconds. specs=%s. Check "
            "`ray status` to see if you have enough resources.",
            int(time.time() - s), placement_group_specs)

    try:
        ray.get(pg_ready_ref, timeout=0)
    except ray.exceptions.GetTimeoutError:
        raise ValueError(
            "Cannot provide a placement group of "
            f"{placement_group_specs=} within {PG_WAIT_TIMEOUT} seconds. See "
            "`ray status` to make sure the cluster has enough resources."
        ) from None

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        else:
            instances[cls].clean()
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class RayExecutor:
    def __init__(self, max_devices_use):
        self.max_devices_use = max_devices_use
        self._init_parallel_degree()
        self._init_ray_workers()

    def _init_parallel_degree(self):
        if self.max_devices_use == 1:
            self.ulysses_degree = 1
            self.ring_degree = 1
        elif self.max_devices_use == 2:
            self.ulysses_degree = 2
            self.ring_degree = 1
        elif self.max_devices_use == 4:
            self.ulysses_degree = 2
            self.ring_degree = 2
        elif self.max_devices_use == 8:
            self.ulysses_degree = 4
            self.ring_degree = 2
        elif self.max_devices_use == 16:
            self.ulysses_degree = 4
            self.ring_degree = 4
        else:
            raise ValueError(f"Invalid number of devices: {self.max_devices_use}")

    def _init_ray_workers(self):
        self._initialize_ray_cluster()
        self.workers = []

        distributed_init_method = get_distributed_init_method(
            "127.0.0.1",
            get_open_port(),
        )
        self.world_size = min(len(self.placement_group.bundle_specs), self.max_devices_use)
        print(f"{self.world_size=}")
        for bundle_id, bundle in enumerate(self.placement_group.bundle_specs):
            if bundle_id >= self.world_size:
                break
            if not bundle.get("GPU", 0):
                continue
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=self.placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            worker = ray.remote(
                num_cpus=0,
                num_gpus=1,
                scheduling_strategy=scheduling_strategy,
            )(RayWorker).remote(
                world_size=self.world_size, 
                ulysses_degree=self.ulysses_degree,
                ring_degree=self.ring_degree,
                distributed_init_method=distributed_init_method
            )

            self.workers.append(worker) 

    def _initialize_ray_cluster(self,):
        ray.init(ignore_reinit_error=True)

        device_str = "GPU"
        num_devices_in_cluster = ray.cluster_resources().get(device_str, 0)
        print(f"{num_devices_in_cluster=}")
        # Create a new placement group
        placement_group_specs: List[Dict[str, float]] = ([{
            device_str: 1.0
        } for _ in range(int(num_devices_in_cluster))])

        # By default, Ray packs resources as much as possible.
        current_placement_group = ray.util.placement_group(
            placement_group_specs, strategy="PACK")
        _wait_until_pg_ready(current_placement_group)

        assert current_placement_group is not None
        # _verify_bundles(current_placement_group, parallel_config, device_str)
        # Set the placement group in the parallel config
        self.placement_group = current_placement_group

    def _run_workers(
        self,
        method: str,
        *args,
        target: str = 'none',
        all_args: Optional[List[Tuple[Any, ...]]] = None,
        all_kwargs: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers. Can be used in the following
        ways:

        Args:
        - args/kwargs: All workers share the same args/kwargs
        - all_args/all_kwargs: args/kwargs for each worker are specified
          individually
        """
        num_workers = len(self.workers)
        all_worker_args = repeat(args, num_workers) if all_args is None \
            else all_args 
        all_worker_kwargs = repeat(kwargs, num_workers) if all_kwargs is None \
            else all_kwargs

        if target == 'none':
            ray_worker_outputs = [
                worker.execute_method.remote(method, *worker_args, **worker_kwargs)
                for (worker, worker_args, worker_kwargs
                    ) in zip(self.workers, all_worker_args, all_worker_kwargs)
            ]
        elif target == 'flux':
            ray_worker_outputs = [
                worker.execute_flux_method.remote(method, *worker_args, **worker_kwargs)
                for (worker, worker_args, worker_kwargs
                    ) in zip(self.workers, all_worker_args, all_worker_kwargs)
            ]
        elif target == 'unet':
            ray_worker_outputs = [
                worker.execute_unet_method.remote(method, *worker_args, **worker_kwargs)
                for (worker, worker_args, worker_kwargs
                    ) in zip(self.workers, all_worker_args, all_worker_kwargs)
            ]

        # Get the results of the ray workers.
        if self.workers:
            ray_worker_outputs = ray.get(ray_worker_outputs)

        spmd_worker_output = ray_worker_outputs[0]
        for worker_output in ray_worker_outputs[1:]:
            if isinstance(spmd_worker_output, torch.Tensor):
                assert torch.allclose(spmd_worker_output, worker_output), "Outputs do not match"
            elif isinstance(spmd_worker_output, np.ndarray):
                assert spmd_worker_output.all() == worker_output.all(), "Outputs do not match"
            else:
                assert spmd_worker_output == worker_output, "Outputs do not match"

        return spmd_worker_output

    def clean(self):
        for worker in self.workers:
            ray.kill(worker)
        ray.util.remove_placement_group(self.placement_group)
        self.workers = []


class UNetExecutor:
    def __init__(self, max_devices_use=2, fix_on_gpu=False, **kwargs):
        global _RAY_EXECUTOR
        self.dtype = kwargs.get('dtype', None)
        if _RAY_EXECUTOR == None:
            _RAY_EXECUTOR = RayExecutor(max_devices_use)
            self.ray_executor = _RAY_EXECUTOR
        else:
            assert max_devices_use == _RAY_EXECUTOR.max_devices_use, \
                f"max_devices_use ({max_devices_use}) in UNetExecutor " + \
                f"is not identical to the previous value ({_RAY_EXECUTOR.max_devices_use})."
            self.ray_executor = _RAY_EXECUTOR
        # initialize UNetWorker
        self.ray_executor._run_workers('initialize_unet', fix_on_gpu, **kwargs)
        self.use_tensor_to_numpy = True

    def _run_workers(
        self,
        method: str,
        *args,
        target: str = 'unet',
        **kwargs,
    ):
        return self.ray_executor._run_workers(method, *args, target=target, **kwargs)

    def __call__(self, x, timestep=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        from xdit_comfyui_private.utils import tensor_to_numpy, numpy_to_tensor
        time_start = time.time()
        dtype = x.dtype
        if self.use_tensor_to_numpy:
            print("Using tensor_to_numpy")
            x = tensor_to_numpy(x)
            timestep = tensor_to_numpy(timestep)
            context = tensor_to_numpy(context)
            y = tensor_to_numpy(y)

        result= self._run_workers("forward", x, timestep, context, y, control, transformer_options, dtype=dtype, use_tensor_to_numpy=self.use_tensor_to_numpy, **kwargs)
        if isinstance(result, np.ndarray):
            result = numpy_to_tensor(result, dtype=dtype)
        time_end = time.time()
        print(f"UNet forward time(in executor): {time_end - time_start:.4f} seconds")
        return result

    def load_state_dict(self, sd, strict=False):
        return self._run_workers("load_state_dict", sd, strict=strict)

    def load_state_dict_from_file(self, unet_path):
        return self._run_workers("load_state_dict_from_file", unet_path)

    def to_gpu(self):
        return self._run_workers("to_gpu")

    def to_cpu(self):
        return self._run_workers("to_cpu")
    

class FluxExecutor:
    def __init__(self, max_devices_use=2, fix_on_gpu=False, **kwargs):
        global _RAY_EXECUTOR
        self.dtype = kwargs.get('dtype', None)
        if _RAY_EXECUTOR == None:
            _RAY_EXECUTOR = RayExecutor(max_devices_use)
            self.ray_executor = _RAY_EXECUTOR
        else:
            assert max_devices_use == _RAY_EXECUTOR.max_devices_use, \
                f"max_devices_use ({max_devices_use}) in FluxExecutor " + \
                f"is not identical to the previous value ({_RAY_EXECUTOR.max_devices_use})."
            self.ray_executor = _RAY_EXECUTOR
        # initialize FluxWorker
        self.ray_executor._run_workers('initialize_flux', fix_on_gpu, **kwargs)
        
    def _run_workers(
        self,
        method: str,
        *args,
        target: str = 'flux',
        **kwargs,
    ):
        return self.ray_executor._run_workers(method, *args, target=target, **kwargs)

    def __call__(self, x, timestep, context, y, guidance, control=None, block_controlnet_hidden_states=None, **kwargs):
        return self._run_workers("forward", x, timestep, context, y, guidance, control, **kwargs)

    def forward_orig(self, img, img_ids, txt, txt_ids, timesteps, y, guidance, control=None, neg_mode=None, block_controlnet_hidden_states=None, block_controlnet_hidden_states_npy=None, **kwargs):
        return self._run_workers("forward_orig", 
                                 img, 
                                 img_ids, 
                                 txt, 
                                 txt_ids, 
                                 timesteps, 
                                 y, 
                                 guidance, 
                                 control, 
                                 neg_mode, 
                                 block_controlnet_hidden_states, 
                                 block_controlnet_hidden_states_npy,
                                 **kwargs)

    def load_state_dict(self, sd, strict=False):
        return self._run_workers("load_state_dict", sd, strict=strict)

    def load_state_dict_from_file(self, unet_path):
        return self._run_workers("load_state_dict_from_file", unet_path)

    def state_dict(self):
        return self._run_workers("state_dict")

    def load_lora(self, lora_path, strength_model):
        return self._run_workers("load_lora", lora_path, strength_model)

    def clean_cache(self):
        return self._run_workers("clean_cache")

    def clean_lora(self):
        return self._run_workers("clean_lora")
    
    def to_gpu(self):
        return self._run_workers("to_gpu")

    def to_cpu(self):
        return self._run_workers("to_cpu")
    