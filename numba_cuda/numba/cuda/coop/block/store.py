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

class BlockStoreAlgorithm(IntEnum):
    DIRECT = 0
    STRIPED = 1
    VECTORIZE = 2
    TRANSPOSE = 3
    WARP_TRANSPOSE = 4
    WARP_TRANSPOSE_TIMESLICED = 5


CUB_BLOCK_STORE_ALGOS = {
    "direct": "::cub::BLOCK_STORE_DIRECT",
    "striped": "::cub::BLOCK_STORE_STRIPED",
    "vectorize": "::cub::BLOCK_STORE_VECTORIZE",
    "transpose": "::cub::BLOCK_STORE_TRANSPOSE",
    "warp_transpose": "::cub::BLOCK_STORE_WARP_TRANSPOSE",
    "warp_transpose_timesliced": "::cub::BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED",
    int(BlockStoreAlgorithm.DIRECT): "::cub::BLOCK_STORE_DIRECT",
    int(BlockStoreAlgorithm.STRIPED): "::cub::BLOCK_STORE_STRIPED",
    int(BlockStoreAlgorithm.VECTORIZE): "::cub::BLOCK_STORE_VECTORIZE",
    int(BlockStoreAlgorithm.TRANSPOSE): "::cub::BLOCK_STORE_TRANSPOSE",
    int(BlockStoreAlgorithm.WARP_TRANSPOSE): "::cub::BLOCK_STORE_WARP_TRANSPOSE",
    int(BlockStoreAlgorithm.WARP_TRANSPOSE_TIMESLICED): "::cub::BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED",
}

class store:
    default_algorithm = BlockStoreAlgorithm.DIRECT
    struct_name = "BlockStore"
    method_name = "Store"
    c_name = "block_store"
    includes = ["cub/block/block_store.cuh"]

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


    def __init__(self, dtype, dim, items_per_thread, algorithm):
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
                "ALGORITHM": CUB_BLOCK_STORE_ALGOS[self.algorithm_enum],
                "BLOCK_DIM_Y": self.dim[1],
                "BLOCK_DIM_Z": self.dim[2],
            }
        )

        #self.temp_storage_bytes = self.specialization.temp_storage_bytes
        #self.temp_storage_alignment = self.specialization.temp_storage_alignment

        return

        self.temp_files = [
            make_binary_tempfile(ltoir, ".ltoir")
            for ltoir in self.specialization.get_lto_ir()
        ]

        self.invocable = Invocable(
            temp_files=self.temp_files,
            temp_storage_bytes=self.temp_storage_bytes,
            temp_storage_alignment=self.temp_storage_alignment,
            algorithm=self.specialization,
        )

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
