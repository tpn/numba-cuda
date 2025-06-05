from functools import cached_property
from io import StringIO

#from numba.cuda import CUsource

# Steal some ideas from Michal Yang's example rewrite on the 4776 issue
# (https://github.com/NVIDIA/cccl/issues/4776#issuecomment-2899508543)
# regarding keyed StringIO, shim source, etc.

class _KeyedStringIO(StringIO):
    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, *kwarg)
        self._keys = set()

    def write_with_key(self, key: str, value: str):
        if key in self._keys:
            return
        self._keys.add(key)
        self.write(value)

    def reset(self):
        self._keys.clear()
        self.seek(0)

class BasePrimitive:

    @property
    def temp_storage_bytes(self):
        return self.specialization.temp_storage_bytes

    @property
    def temp_storage_alignment(self):
        return self.specialization.temp_storage_alignment

    @cached_property
    def temp_files(self):
        from ._common import make_binary_tempfile
        return [
            make_binary_tempfile(ltoir, ".ltoir")
            for ltoir in self.specialization.get_lto_ir()
        ]

    @cached_property
    def invocable(self):
        from ._types import Invocable
        return Invocable()
