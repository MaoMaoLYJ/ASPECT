import pytest

from dashboard import tp1_replica_eval as replicas


@pytest.mark.parametrize('count', [1, 2, 4])
def test_replicas_are_separate_internal_dp1_servers(monkeypatch, count):
    monkeypatch.setattr(replicas, 'REPLICA_COUNT', count)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', ','.join(str(i + 2) for i in range(count)))
    events = []

    class Server:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            events.append(('start', self.kwargs['port']))

        def load_lora(self, name, path):
            events.append(('load', self.kwargs['port'], name, path))

        def unload_all_loras(self):
            events.append(('unload', self.kwargs['port']))

        def stop(self):
            events.append(('stop', self.kwargs['port']))

    monkeypatch.setattr(replicas, 'SingleServer', Server)
    manager = replicas.ReplicaServers(model='/model', tensor_parallel_size=1,
                                     data_parallel_size=count, port=8250)
    assert len(manager.servers) == count
    for index, server in enumerate(manager.servers):
        assert server.kwargs['data_parallel_size'] == 1
        assert server.kwargs['tensor_parallel_size'] == 1
        assert server.kwargs['port'] == 8250 + index
        assert server.kwargs['process_env']['CUDA_VISIBLE_DEVICES'] == str(index + 2)
    manager.start()
    manager.load_lora('worker', '/adapter')
    manager.unload_all_loras()
    manager.stop()
    for kind in ('start', 'load', 'unload', 'stop'):
        assert len([event for event in events if event[0] == kind]) == count


@pytest.mark.parametrize('devices', ['2,2', '2', ''])
def test_replica_gpu_contract_rejects_duplicates_and_missing_devices(monkeypatch, devices):
    monkeypatch.setattr(replicas, 'REPLICA_COUNT', 2)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', devices)
    with pytest.raises(ValueError, match='distinct'):
        replicas.ReplicaServers(model='/model', tensor_parallel_size=1,
                                data_parallel_size=2, port=8250)
