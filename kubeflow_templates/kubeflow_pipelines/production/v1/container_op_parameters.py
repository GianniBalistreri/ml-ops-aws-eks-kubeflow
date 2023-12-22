"""

Parameters for container operation

"""

from kfp import dsl


def add_container_op_parameters(container_op: dsl.component,
                                n_cpu_request: str = None,
                                n_cpu_limit: str = None,
                                n_gpu: str = None,
                                gpu_vendor: str = 'nvidia',
                                memory_request: str = '1G',
                                memory_limit: str = None,
                                ephemeral_storage_request: str = '5G',
                                ephemeral_storage_limit: str = None,
                                instance_name: str = None,
                                max_cache_staleness: str = 'P0D'
                                ):
    """
    Add parameter to container operation

    :param n_cpu_request: str
        Number of requested CPU's

    :param n_cpu_limit: str
        Maximum number of requested CPU's

    :param n_gpu: str
        Maximum number of requested GPU's

    :param gpu_vendor: str
        Name of the GPU vendor
            -> amd: AMD
            -> nvidia: NVIDIA

    :param container_op: dsl.ContainerOp
        DSL container class

    :param memory_request: str
        Memory request

    :param memory_limit: str
        Limit of the requested memory

    :param ephemeral_storage_request: str
        Ephemeral storage request (cloud based additional memory storage)

    :param ephemeral_storage_limit: str
        Limit of the requested ephemeral storage (cloud based additional memory storage)

    :param instance_name: str
        Name of the used AWS instance (value)

    :param max_cache_staleness: str
        Maximum of staleness days of the component cache
    """
    if instance_name is not None:
        container_op.add_node_selector_constraint(label_name="beta.kubernetes.io/instance-type",
                                                  value=instance_name
                                                  )
    if n_cpu_request is not None:
        container_op.container.set_cpu_request(cpu=n_cpu_request)
    if n_cpu_limit is not None:
        container_op.container.set_cpu_limit(cpu=n_cpu_limit)
    if n_gpu is not None:
        container_op.container.set_gpu_limit(gpu=n_gpu, vendor=gpu_vendor)
    container_op.container.set_memory_request(memory=memory_request)
    if memory_limit is not None:
        container_op.container.set_memory_limit(memory=memory_limit)
    container_op.container.set_ephemeral_storage_request(size=ephemeral_storage_request)
    if ephemeral_storage_limit is not None:
        container_op.container.set_ephemeral_storage_limit(size=ephemeral_storage_limit)
    container_op.container.execution_options.caching_strategy.max_cache_staleness = max_cache_staleness
