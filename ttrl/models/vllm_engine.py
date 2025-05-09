import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import vllm
from ttrl.helper.logging_utils import init_logger
from ttrl.models.vllm_wrapper import ray_noset_visible_devices

logger = init_logger(__name__)


@ray.remote
def get_all_env_variables():
    import os

    return os.environ


@ray.remote
class LLMRayActor:
    def __init__(self, *args, **kwargs):
        self.__version__ = vllm.__version__
        # Assuming TTRL was primarily developed for vLLM around 0.4.x - 0.6.x
        # We keep the 0.4.2 assert but acknowledge later versions might also work with this simplified code
        assert self.__version__ >= "0.4.2", "TTRL might require vLLM >= 0.4.2"

        noset_visible_devices = kwargs.pop("noset_visible_devices", False)
        self.use_gpu_executor = kwargs["tensor_parallel_size"] == 1 and not noset_visible_devices

        if self.use_gpu_executor:
            from ttrl.models.vllm_wrapper import WorkerWrap
            vllm.worker.worker.Worker = WorkerWrap
        else:
            kwargs["worker_use_ray"] = True
            if vllm.__version__ >= "0.4.3":
                kwargs["distributed_executor_backend"] = "ray"
            # For vLLM versions <= 0.6.4.post1 that used RayWorkerWrapper patching
            if "worker_cls" not in kwargs and vllm.__version__ <= "0.6.4.post1": # Check if worker_cls is already set by newer logic
                RayWorkerWrapperPath = vllm.executor.ray_utils
                class RayWorkerWrapper(RayWorkerWrapperPath.RayWorkerWrapper):
                    def __init__(self, *args, **kwargs) -> None:
                        kwargs["worker_module_name"] = "ttrl.models.vllm_wrapper"
                        kwargs["worker_class_name"] = "WorkerWrap"
                        super().__init__(*args, **kwargs)
                RayWorkerWrapperPath.RayWorkerWrapper = RayWorkerWrapper
            elif vllm.__version__ > "0.6.4.post1": # For versions explicitly setting worker_cls
                 kwargs["worker_cls"] = "ttrl.models.vllm_wrapper.WorkerWrap"


        self.llm = vllm.LLM(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        # Simplified version assuming vLLM 0.6.x or compatible API
        # This path might have been the original one before complex version handling was added.
        if hasattr(self.llm.llm_engine, 'model_executor'):
            if self.use_gpu_executor:
                logger.info(f"Using GPU executor path for init_process_group (vLLM {self.__version__})")
                return self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                    master_address, master_port, rank_offset, world_size, group_name, backend
                )
            else:
                logger.info(f"Using distributed executor path for init_process_group (vLLM {self.__version__})")
                return self.llm.llm_engine.model_executor._run_workers(
                    "init_process_group", master_address, master_port, rank_offset, world_size, group_name, backend
                )
        elif hasattr(self.llm.llm_engine, 'worker'): # Fallback for newer vLLM if model_executor is missing
            logger.info(f"Falling back to .worker for init_process_group (vLLM {self.__version__})")
            return self.llm.llm_engine.worker.init_process_group(
                master_address, master_port, rank_offset, world_size, group_name, backend
            )
        else:
            logger.error(f"Cannot initialize process group: LLMEngine has no 'model_executor' or 'worker' attribute (vLLM {self.__version__}).")
            raise AttributeError(f"LLMEngine has no 'model_executor' or 'worker' attribute needed for init_process_group.")

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self.stop_remote_worker_execution_loop() # Call this first
        if hasattr(self.llm.llm_engine, 'model_executor'):
            if self.use_gpu_executor:
                return self.llm.llm_engine.model_executor.driver_worker.update_weight(name, dtype, shape, empty_cache)
            else:
                return self.llm.llm_engine.model_executor._run_workers("update_weight", name, dtype, shape, empty_cache)
        elif hasattr(self.llm.llm_engine, 'worker'):
             return self.llm.llm_engine.worker.update_weight(name, dtype, shape, empty_cache)
        else:
            logger.error(f"Cannot update weight: LLMEngine has no 'model_executor' or 'worker' attribute (vLLM {self.__version__}).")
            raise AttributeError(f"LLMEngine has no 'model_executor' or 'worker' attribute needed for update_weight.")

    def reset_prefix_cache(self):
        if vllm.__version__ < "0.7.0":
            logger.warning("Reset prefix cache API is available only from vLLM 0.7.0!")
            return
        if hasattr(self.llm.llm_engine, 'reset_prefix_cache'):
             self.llm.llm_engine.reset_prefix_cache()
        else:
            logger.warning(f"LLMEngine has no 'reset_prefix_cache' method (vLLM {self.__version__}).")

    def stop_remote_worker_execution_loop(self):
        if self.__version__ > "0.4.2": # This check seems generally applicable
            if hasattr(self.llm.llm_engine, 'model_executor') and hasattr(self.llm.llm_engine.model_executor, 'stop_remote_worker_execution_loop'):
                self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()
            # No clear equivalent for newer .worker path, might not be needed or handled differently.
            # else: 
            #    logger.info("stop_remote_worker_execution_loop: Not applicable for current vLLM structure or version.")

def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
):
    vllm_engines = []
    noset_visible_devices = ray_noset_visible_devices(ray.get(get_all_env_variables.remote()))
    for i in range(num_engines):
        num_gpus = int(tensor_parallel_size == 1 and not noset_visible_devices)
        scheduling_strategy = None

        if tensor_parallel_size > 1 or noset_visible_devices:
            bundles = [{"GPU": 1, "CPU": 1}] * tensor_parallel_size
            pg = placement_group(bundles)
            ray.get(pg.ready())
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
            )

        vllm_engines.append(
            LLMRayActor.options(
                num_cpus=1,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                pretrain,
                noset_visible_devices=noset_visible_devices,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size,
                dtype="bfloat16",
                seed=seed + i,
                enable_prefix_caching=enable_prefix_caching,
                enforce_eager=enforce_eager,
                max_model_len=max_model_len,
            )
        )
    return vllm_engines

if __name__ == "__main__":
    # This is an example, ensure your Ray cluster is initialized before running
    # ray.init() or ray.init(address='auto')
    # llm = LLMRayActor.remote("meta-llama/Llama-2-7b-chat-hf", tensor_parallel_size=4) # Example with TP
    # output = ray.get(llm.generate.remote("San Franciso is a"))
    # print(f"output: {output}")
    pass
