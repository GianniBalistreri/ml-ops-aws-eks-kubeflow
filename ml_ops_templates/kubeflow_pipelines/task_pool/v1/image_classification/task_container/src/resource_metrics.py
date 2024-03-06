"""

Collect technical resources metrics on component and operation level

"""

import psutil

from custom_logger import Log
from typing import List


def get_available_cpu(logging: bool = True) -> int:
    """
    Get available number of CPU's

    :param logging: bool
        Whether to log results or not

    :return: int
        Number of CPU's
    """
    _n_cpu: int = psutil.cpu_count(logical=False)
    if logging:
        Log().log(msg=f'Number of available CPUs: {_n_cpu}')
    return _n_cpu


def get_cpu_utilization(interval: int = 1, logging: bool = True) -> float:
    """
    Get CPU utilization metric

    :param interval: int
        CPU utilization interval in seconds

    :param logging: bool
        Whether to log results or not

    :return: float
        CPU utilization per second in percent
    """
    _cpu_utilization: float = psutil.cpu_percent(percpu=False, interval=interval)
    if logging:
        Log().log(msg=f'CPU utilization: {_cpu_utilization} % per second')
    return _cpu_utilization


def get_cpu_utilization_per_core(interval: int = 1, logging: bool = True) -> List[float]:
    """
    Get CPU utilization metric per core

    :param interval: int
        CPU utilization interval in seconds

    :param logging: bool
        Whether to log results or not

    :return: List[float]
        CPU utilization per core per second in percent
    """
    _cpu_utilization_per_core: List[float] = psutil.cpu_percent(percpu=True, interval=interval)
    if logging:
        Log().log(msg=f'CPU utilization per core: {_cpu_utilization_per_core} % per second')
    return _cpu_utilization_per_core


def get_memory(total: bool = True, logging: bool = True) -> float:
    """
    Get memory metric

    :param total: bool
        Whether to get total or available memory

    :param logging: bool
        Whether to log results or not

    :return: float
        Memory metric
    """
    _memory = psutil.virtual_memory()
    if total:
        _memory_total: float = round(_memory.total / (1024 ** 3), 2)
        if logging:
            Log().log(msg=f'Memory total: {_memory_total} GB')
        return _memory_total
    else:
        _memory_available: float = round(_memory.available / (1024 ** 3), 2)
        if logging:
            Log().log(msg=f'Memory available: {_memory_available} GB')
        return _memory_available


def get_memory_utilization(logging: bool = True) -> float:
    """
    Get memory utilization metric

    :param logging: bool
        Whether to log results or not

    :return: float
        Memory utilization in percent
    """
    _memory = psutil.virtual_memory()
    if logging:
        Log().log(msg=f'Memory utilization: {_memory.percent} %')
    return _memory.percent
