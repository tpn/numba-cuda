import numpy as np
from numba import cuda
from numba.cuda.testing import CUDATestCase
import unittest


class TestBlockLoadStore(CUDATestCase):
    def test_block_load_store_positional1(self):
        threads_per_block = 32
        items_per_thread = 4
        dtype = np.int32

        @cuda.jit
        def kernel(d_in, d_out):
            thread_data = cuda.local.array(items_per_thread, dtype=dtype)
            cuda.block.load(
                d_in,
                thread_data,
                threads_per_block,
                items_per_thread,
            )
            cuda.block.store(
                d_out,
                thread_data,
                threads_per_block,
                items_per_thread,
            )

        num_items = threads_per_block * items_per_thread
        a = np.arange(num_items, dtype=np.int32)
        b = np.zeros_like(a)

        k = kernel[1, threads_per_block]
        print(k)
        # k(a, b)

        # self.assertTrue(np.array_equal(a, b))

    def test_block_load_store_positional2(self):
        threads_per_block = 32
        items_per_thread = 4
        dtype = np.int32

        @cuda.jit
        def kernel(d_in, d_out):
            thread_data = cuda.local.array(items_per_thread, dtype=dtype)
            cuda.block.load(
                d_in,
                thread_data,
                threads_per_block,
                items_per_thread,
                cuda.BlockLoadAlgorithm.STRIPED,
            )
            cuda.block.store(
                d_out,
                thread_data,
                threads_per_block,
                items_per_thread,
                cuda.BlockStoreAlgorithm.STRIPED,
            )

        num_items = threads_per_block * items_per_thread
        a = np.arange(num_items, dtype=np.int32)
        b = np.zeros_like(a)

        k = kernel[1, threads_per_block]
        print(k)
        # k(a, b)

        # self.assertTrue(np.array_equal(a, b))

    def test_block_load_store_positional3(self):
        threads_per_block = 32
        items_per_thread = 4
        dtype = np.int32

        @cuda.jit
        def kernel(d_in, d_out):
            thread_data = cuda.local.array(items_per_thread, dtype=dtype)
            cuda.block.load(
                d_in,
                thread_data,
                threads_per_block,
                items_per_thread,
                algorithm=cuda.BlockLoadAlgorithm.WARP_TRANSPOSE,
            )
            cuda.block.store(
                d_out,
                thread_data,
                threads_per_block,
                items_per_thread,
                algorithm=cuda.BlockStoreAlgorithm.WARP_TRANSPOSE,
            )

        num_items = threads_per_block * items_per_thread
        a = np.arange(num_items, dtype=np.int32)
        b = np.zeros_like(a)

        k = kernel[1, threads_per_block]
        print(k)
        # k(a, b)

        # self.assertTrue(np.array_equal(a, b))

    def test_block_load_store_literals(self):
        threads_per_block = 32
        items_per_thread = 4
        dtype = np.int32

        @cuda.jit
        def kernel(d_in, d_out):
            thread_data = cuda.local.array(items_per_thread, dtype=dtype)
            cuda.block.load(
                d_in,
                thread_data,
                32,
                4,
            )
            cuda.block.store(
                d_out,
                thread_data,
                32,
                4,
            )

        num_items = threads_per_block * items_per_thread
        a = np.arange(num_items, dtype=np.int32)
        b = np.zeros_like(a)

        k = kernel[1, threads_per_block]
        print(k)
        # k(a, b)
        # self.assertTrue(np.array_equal(a, b))

    def test_block_load_store_kwds(self):
        threads_per_block = 32
        items_per_thread = 4
        num_items = threads_per_block * items_per_thread
        dtype = np.int32

        @cuda.jit
        def kernel1(d_in, d_out):
            thread_data = cuda.local.array(items_per_thread, dtype=dtype)
            cuda.block.load(
                src=d_in,
                dst=thread_data,
                threads_per_block=threads_per_block,
                items_per_thread=items_per_thread,
                algorithm=cuda.BlockLoadAlgorithm.WARP_TRANSPOSE,
            )
            cuda.block.store(
                dst=d_out,
                src=thread_data,
                threads_per_block=threads_per_block,
                items_per_thread=items_per_thread,
                algorithm=cuda.BlockStoreAlgorithm.WARP_TRANSPOSE,
            )

        a = np.arange(num_items, dtype=np.int32)
        b = np.zeros_like(a)
        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        k1 = kernel1[1, threads_per_block]
        print(k1)
        # k1(d_a, d_b)
        # h_a = d_a.copy_to_host()
        # h_b = d_b.copy_to_host()
        # self.assertTrue(np.array_equal(h_a, h_b))

        # --

        @cuda.jit
        def kernel2(d_in, d_out):
            thread_data = cuda.local.array(items_per_thread, dtype=dtype)
            cuda.block.load(
                d_in,
                thread_data,
                threads_per_block=threads_per_block,
                items_per_thread=items_per_thread,
                algorithm=cuda.BlockLoadAlgorithm.WARP_TRANSPOSE,
            )
            cuda.block.store(
                d_out,
                thread_data,
                threads_per_block=threads_per_block,
                items_per_thread=items_per_thread,
                algorithm=cuda.BlockStoreAlgorithm.WARP_TRANSPOSE,
            )

        a = np.arange(num_items, dtype=np.int32)
        b = np.zeros_like(a)
        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        k2 = kernel2[1, threads_per_block]
        print(k2)
        # k2(d_a, d_b)
        # h_a = d_a.copy_to_host()
        # h_b = d_b.copy_to_host()
        # self.assertTrue(np.array_equal(h_a, h_b))


class TestInvariants(CUDATestCase):
    def test_block_load_store_literals_required_for_tpb_ipt(self):
        threads_per_block = random.randint(1, 1024)
        items_per_thread = random.randint(1, 32)
        dtype = np.int32

        def kernel(d_in, d_out):
            thread_data = cuda.local.array(items_per_thread, dtype=dtype)
            cuda.block.load(
                d_in,
                thread_data,
                threads_per_block,
                items_per_thread,
                cuda.BlockLoadAlgorithm.STRIPED,
            )
            cuda.block.store(
                d_out,
                thread_data,
                threads_per_block,
                items_per_thread,
                cuda.BlockStoreAlgorithm.STRIPED,
            )

        num_items = threads_per_block * items_per_thread
        a = np.arange(num_items, dtype=np.int32)
        b = np.zeros_like(a)

        k = kernel[1, threads_per_block]
        print(k)
        # k(a, b)

        # self.assertTrue(np.array_equal(a, b))
        # Check that the invariants hold for block.load and block.store
        self.assertTrue(
            hasattr(cuda.block, "load"),
            "block.load should be defined",
        )
        self.assertTrue(
            hasattr(cuda.block, "store"),
            "block.store should be defined",
        )
        self.assertTrue(
            hasattr(cuda.block.load, "temp_storage_bytes"),
            "block.load should have temp_storage_bytes",
        )
        self.assertTrue(
            hasattr(cuda.block.store, "temp_storage_bytes"),
            "block.store should have temp_storage_bytes",
        )


if __name__ == "__main__":
    unittest.main()
