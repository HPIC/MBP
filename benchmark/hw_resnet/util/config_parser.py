from typing import Dict

import toml


class DotDict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val


class ConfigParser:
    def __init__(self, config: Dict) -> None:
        self._config = DotDict(config)

    @classmethod
    def parse(cls, config_path: str) -> Dict:
        config = toml.load(config_path)
        return cls(config=config)

    # setting read-only attributes
    @property
    def data(self):
        return self._config
