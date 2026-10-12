import os

from torchparity.backends import Backend


class Crashing(Backend):
    """Kills its process on cases for the sign op."""

    name = "crashing"

    def compile(self, module, inputs):
        if "sign" in str(module.fn.__code__.co_names):
            os.abort()
        return lambda args: module(*args)
