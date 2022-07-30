
from .thermal_loader import thermalLoader


def get_loader(name):
    """
    Parameters:
        -   name: the name of the parmeter in the YAML config file
    Returns:
        -   The loader class
    """
    return {
        "thermal_loader": thermalLoader,
    }[name]