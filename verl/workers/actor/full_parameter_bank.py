"""Independent full-model states in RAM; never adapters or disk checkpoints."""

import copy

import torch


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


class TensorBank:
    """Copy local parameter shards in place, preserving FSDP/CUDA graph handles."""

    def __init__(self, module):
        self.module = module
        self.states = {}
        self.active = None

    def tensors(self):
        values = dict(self.module.named_parameters())
        values.update(dict(self.module.named_buffers()))
        return {k: v.to_local() if hasattr(v, 'to_local') else v for k, v in values.items()}

    def capture(self, route):
        self.states[route] = cpu_copy(self.tensors())
        self.active = route

    @torch.no_grad()
    def activate(self, route):
        if route not in self.states:
            raise ValueError(f'Unknown full-parameter route: {route}')
        current = self.tensors()
        saved = self.states[route]
        if current.keys() != saved.keys():
            raise RuntimeError('Full-parameter tensor topology changed')
        for name, parameter in current.items():
            if parameter.shape != saved[name].shape or parameter.dtype != saved[name].dtype:
                raise RuntimeError(f'Full-parameter local shard changed: {name}')
            parameter.copy_(saved[name])
        self.active = route


class FullParameterBank:
    """One independent weight/Adam/scheduler state per agent, one execution slot."""

    def __init__(self, module, optimizer, scheduler, routes):
        if not routes or len(set(routes)) != len(routes):
            raise ValueError('Unique explicit full-parameter routes required')
        self.weights = TensorBank(module)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.routes = tuple(routes)
        self.states = {}
        for route in self.routes:
            self.weights.capture(route)
            self.states[route] = (cpu_copy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict()))
        self.active = self.routes[0]

    def save_active(self):
        self.weights.capture(self.active)
        self.states[self.active] = (cpu_copy(self.optimizer.state_dict()), copy.deepcopy(self.scheduler.state_dict()))

    def activate(self, route):
        if route not in self.states:
            raise ValueError(f'Unknown full-parameter agent: {route}')
        if route == self.active:
            return
        self.save_active()
        self.weights.activate(route)
        optimizer, scheduler = self.states[route]
        # load_state_dict may retain CPU tensor references; keep bank snapshots immutable.
        self.optimizer.load_state_dict(cpu_copy(optimizer))
        self.scheduler.load_state_dict(copy.deepcopy(scheduler))
        self.optimizer.zero_grad(set_to_none=True)
        self.active = route
