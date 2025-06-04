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

import cuda.cooperative.experimental as cudax

block_load = cudax.block.load(
    dtype,
    threads_per_block,
    items_per_thread,
)

block_store = cudax.block.store(
    dtype,
    threads_per_block,
    items_per_thread,
)

jit_files = block_load.files + block_store.files

def kernel1(d_in, d_out):
    thread_data = cuda.local.array(items_per_thread, dtype=dtype)
    block_load(d_in, thread_data)
    block_store(d_out, thread_data)


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

    jk = cuda.jit(kernel, link=jit_files)
    k = jk[1, threads_per_block]
    try:
        k(d_a, d_b)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Source code: {source_code}")
        raise e
    else:
        print(f"Kernel {num} executed")


if __name__ == "__main__":
    main()
