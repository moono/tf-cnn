import importlib


def get_proper_fn(module_name, function_name):
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name)
    return fn
