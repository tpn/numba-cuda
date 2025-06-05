import numpy as np
from numba import cuda

dtype = np.int32
threads_per_block = 32
items_per_thread = 4
num_items = threads_per_block * items_per_thread
a = np.arange(num_items, dtype=np.int32)
b = np.zeros_like(a)
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)


def kernel1(d_in, d_out):
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


def kernel2(d_in, d_out):
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


def kernel3(d_in, d_out):
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


def kernel3_2(d_in, d_out):
    thread_data = cuda.local.array(items_per_thread, dtype=dtype)
    cuda.block.load(
        d_in,
        thread_data,
        threads_per_block,
        items_per_thread,
        cuda.BlockStoreAlgorithm.WARP_TRANSPOSE,
    )
    cuda.block.store(
        d_out,
        thread_data,
        threads_per_block,
        items_per_thread,
        cuda.BlockStoreAlgorithm.WARP_TRANSPOSE,
    )


def kernel3_3(d_in, d_out):
    thread_data = cuda.local.array(items_per_thread, dtype=dtype)
    cuda.block.load(
        d_in,
        thread_data,
        threads_per_block,
        items_per_thread,
        cuda.BlockLoadAlgorithm.WARP_TRANSPOSE,
    )
    cuda.block.store(
        d_out,
        thread_data,
        threads_per_block,
        items_per_thread,
        cuda.BlockLoadAlgorithm.WARP_TRANSPOSE,
    )


def kernel4(d_in, d_out):
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


def kernel5(d_in, d_out):
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


def main():
    import sys
    import inspect

    if len(sys.argv) > 1:
        num = int(sys.argv[1])
    else:
        num = 1

    print(f"Attempting to run kernel {num}")
    kernel_name = f"kernel{num}"
    kernel = globals()[kernel_name]

    # Print the source code of the kernel.
    source_code = inspect.getsource(kernel)
    print(source_code)

    jk = cuda.jit(kernel)
    k = jk[1, threads_per_block]
    try:
        k(d_a, d_b)
        cuda.synchronize()
    except Exception as e:
        print(f"Error: {e}")
        print(f"Source code: {source_code}")
        raise e
    else:
        print(f"Kernel {num} executed")

    # Copy d_a and d_b back to h_a and h_b and compare against each other.
    import numpy as np
    h_a = d_a.copy_to_host()
    h_b = d_b.copy_to_host()
    print(f'a: {a}')
    print(f'b: {b}')
    print(f"h_a: {h_a}")
    print(f"h_b: {h_b}")
    np.testing.assert_array_equal(
        h_a, h_b, err_msg=f"Kernel {num} failed: h_a != h_b"
    )


if __name__ == "__main__":
    main()
