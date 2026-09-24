"""Host-local port leases for independent native vLLM DP servers."""

from __future__ import annotations

import fcntl
from pathlib import Path
import socket

WIDTH = 32


class PortLease:
    """Keep an advisory block lock until the evaluator and its children exit."""

    def __init__(self, directory='/tmp/um_vllm_dp_ports', candidates=None, exclude=()):
        self.directory = Path(directory)
        if candidates is None:
            low, high = 32768, 60999
            ephemeral = Path('/proc/sys/net/ipv4/ip_local_port_range')
            if ephemeral.exists():
                low, high = map(int, ephemeral.read_text().split())
            candidates = [p for p in range(22000, 32000-WIDTH, WIDTH)
                          if p+WIDTH-1 < low or p > high]
        self.candidates = [p for p in candidates if p not in exclude]
        self.handle = None
        self.base = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        for base in self.candidates:
            handle = (self.directory / f'{base}.lock').open('a+')
            sockets = []
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for port in range(base, base + WIDTH):
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sockets.append(sock)
                    sock.bind(('0.0.0.0', port))
                self.handle, self.base = handle, base
                return self
            except OSError:
                handle.close()
            finally:
                for sock in sockets:
                    sock.close()
        raise RuntimeError('No free non-ephemeral vLLM DP port block')

    def __exit__(self, *_):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def install_port_contract(base, *, config_class=None):
    """Override the native DP allocator, which otherwise ignores the env port."""
    base = int(base)
    if not 1024 <= base <= 65535-WIDTH:
        raise ValueError(f'Invalid DP port block: {base}')
    if config_class is None:
        from vllm.config import ParallelConfig
        config_class = ParallelConfig
    original_init = config_class.__post_init__
    original_next = config_class.get_next_dp_init_port

    def initialize(config):
        original_init(config)
        if config.data_parallel_size > 1:
            config.data_parallel_master_port = base
            config.data_parallel_rpc_port = base + WIDTH - 1

    def next_port(config):
        if config.data_parallel_size > 1:
            if not base <= config.data_parallel_master_port < base+WIDTH-1:
                raise RuntimeError('Leased vLLM DP port block exhausted')
        return original_next(config)

    config_class.__post_init__ = initialize
    config_class.get_next_dp_init_port = next_port
