# -*- coding: utf-8 -*-

class Registry:
    """
    Registry for managing Planner implementations
    """
    def __init__(self, name):
        self.name = name
        self._module_dict = {}

    def register(self, module_name=None):
        def _register(cls):
            name = module_name or cls.__name__
            if name in self._module_dict:
                raise KeyError(f"{name} is already registered in {self.name}")
            self._module_dict[name] = cls
            return cls
        return _register

    def get(self, name):
        if name not in self._module_dict:
            raise KeyError(f"{name} is not registered in {self.name}")
        return self._module_dict[name]

PLANNER_REGISTRY = Registry("Planner")
