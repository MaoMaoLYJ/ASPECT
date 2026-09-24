"""Register the trainer's resolver before Hydra's config-only path executes."""
import runpy
import sys


def register_resolvers():
    from omegaconf import OmegaConf
    if not OmegaConf.has_resolver('mul'):
        OmegaConf.register_new_resolver('mul', lambda x, y: int(x) * int(y))


def main():
    module = sys.argv.pop(1)
    register_resolvers()
    runpy.run_module(module, run_name='__main__')


if __name__ == '__main__':
    main()
