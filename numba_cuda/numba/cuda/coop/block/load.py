# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import IntEnum
from functools import cached_property

import numba
from numba import cuda
from numba.core.typing import signature
from numba.core import types

from .._common import (
    make_binary_tempfile,
    normalize_dim_param,
    normalize_dtype_param,
)

from .._types import (
    Algorithm,
    Dependency,
    DependentArray,
    DependentPointer,
    Invocable,
    Pointer,
    TemplateParameter,
)

class BlockLoadAlgorithm(IntEnum):
    DIRECT = 0
    STRIPED = 1
    VECTORIZE = 2
    TRANSPOSE = 3
    WARP_TRANSPOSE = 4
    WARP_TRANSPOSE_TIMESLICED = 5


CUB_BLOCK_LOAD_ALGOS = {
    "direct": "::cub::BLOCK_LOAD_DIRECT",
    "striped": "::cub::BLOCK_LOAD_STRIPED",
    "vectorize": "::cub::BLOCK_LOAD_VECTORIZE",
    "transpose": "::cub::BLOCK_LOAD_TRANSPOSE",
    "warp_transpose": "::cub::BLOCK_LOAD_WARP_TRANSPOSE",
    "warp_transpose_timesliced": "::cub::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED",
    int(BlockLoadAlgorithm.DIRECT): "::cub::BLOCK_LOAD_DIRECT",
    int(BlockLoadAlgorithm.STRIPED): "::cub::BLOCK_LOAD_STRIPED",
    int(BlockLoadAlgorithm.VECTORIZE): "::cub::BLOCK_LOAD_VECTORIZE",
    int(BlockLoadAlgorithm.TRANSPOSE): "::cub::BLOCK_LOAD_TRANSPOSE",
    #cuda.BlockLoadAlgorithm.WARP_TRANSPOSE: "::cub::BLOCK_LOAD_WARP_TRANSPOSE",
    #cuda.BlockLoadAlgorithm.WARP_TRANSPOSE_TIMESLICED: "::cub::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED",
}

class load:
    default_algorithm = BlockLoadAlgorithm.DIRECT
    struct_name = "BlockLoad"
    method_name = "Load"
    c_name = "block_load"
    includes = ["cub/block/block_load.cuh"]

    @staticmethod
    def _typer_implicit_temp_storage(src, dst):
        return signature(
            types.none,
            args=(src, dst),
            recvr=None,
            pysig=None,
        )

    @staticmethod
    def _typer_explicit_temp_storage(temp_storage, src, dst):
        return signature(
            types.none,
            args=(src, dst, temp_storage),
            recvr=None,
            pysig=None,
        )

    def __init__(self, dtype, dim, items_per_thread, algorithm=None):
        self.dtype = normalize_dtype_param(dtype)
        self.dim = normalize_dim_param(dim)
        self.items_per_thread = items_per_thread
        if algorithm is None:
            algorithm = self.default_algorithm
        self.algorithm_enum = algorithm

        self.template_parameters = [
            TemplateParameter("T"),
            TemplateParameter("BLOCK_DIM_X"),
            TemplateParameter("ITEMS_PER_THREAD"),
            TemplateParameter("ALGORITHM"),
            TemplateParameter("BLOCK_DIM_Y"),
            TemplateParameter("BLOCK_DIM_Z"),
        ]

        self.parameters = [
            [
                Pointer(numba.uint8),
                DependentPointer(Dependency("T")),
                DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
            ]
        ]

        self.algorithm = Algorithm(
            self.struct_name,
            self.method_name,
            self.c_name,
            self.includes,
            self.template_parameters,
            self.parameters,
        )
        self.specialization = self.algorithm.specialize(
            {
                "T": self.dtype,
                "BLOCK_DIM_X": self.dim[0],
                "ITEMS_PER_THREAD": items_per_thread,
                "ALGORITHM": CUB_BLOCK_LOAD_ALGOS[self.algorithm_enum],
                "BLOCK_DIM_Y": self.dim[1],
                "BLOCK_DIM_Z": self.dim[2],
            }
        )

        #self.temp_storage_bytes = self.specialization.temp_storage_bytes
        #self.temp_storage_alignment = self.specialization.temp_storage_alignment

    @property
    def temp_storage_bytes(self):
        return self.specialization.temp_storage_bytes

    @property
    def temp_storage_alignment(self):
        return self.specialization.temp_storage_alignment

    @cached_property
    def temp_files(self):
        return [
            make_binary_tempfile(ltoir, ".ltoir")
            for ltoir in self.specialization.get_lto_ir()
        ]

    @cached_property
    def invocable(self):
        return Invocable()
