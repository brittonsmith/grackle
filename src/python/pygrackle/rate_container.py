from functools import partial

import pygrackle.grackle_wrapper as grackle_wrapper

class RateContainer:
    def __init__(self, chemistry_data):
        self.chemistry_data = chemistry_data

    def __getattribute__(self, key):
        fname = f"get_{key}_rate"
        rfunc = getattr(grackle_wrapper, fname, None)
        if rfunc is None:
            return super().__getattribute__(key)
        return partial(rfunc, self)
