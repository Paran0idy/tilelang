# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import tilelang
import tilelang.language as T

def prefix_sum_inclusive_vectorize(M, N, blk_m, threads):
    dtype = "float"

    @T.prim_func
    def main(A: T.Buffer((M, N), dtype), B: T.Buffer((M, N), dtype)):
        with T.Kernel(M, threads=threads) as bx:
            A_shared = T.alloc_shared((N), dtype)
            tid = T.get_thread_binding()

            T.copy(A[bx * blk_m:(bx + 1) * blk_m, :], A_shared)            
            
            steps = T.alloc_var("int32")
            steps = T.log2(T.Cast("float32", N)).astype("int32")

            # Up-sweep phase
            for i in T.serial(steps):
                offset = 1 << i
                stride = threads * offset * 2
                K = (N + stride) // stride
                for k in T.serial(0, K):
                    idx = (tid + 1 + k * threads) * offset * 2 - 1
                    if idx < N:
                        A_shared[idx] += A_shared[idx - offset]

            # Down-sweep phase
            for i in T.serial(steps - 1):
                offset = (N // 2) // (1 << (i + 1))
                stride = threads * offset * 2
                K = (N + stride - 1) // stride
                for k in T.serial(0, K):
                    idx = (tid + 1 + k * threads) * offset * 2 - 1
                    if idx + offset < N:
                        A_shared[idx + offset] += A_shared[idx]

            T.copy(A_shared, B[bx * blk_m:(bx + 1) * blk_m, :])

    return main

def ref_program(x):
    return torch.cumsum(x, dim=1)


if __name__ == "__main__":
    
    M, N, blk_m = 16, 8192, 1
    program = prefix_sum_inclusive_vectorize(M, N, blk_m, 128)
    kernel = tilelang.compile(program, out_idx=-1, target="cuda", execution_backend="cython")
    print(kernel.get_kernel_source())
    
    profiler = kernel.get_profiler()
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("All checks pass.")

    latency = profiler.do_bench(ref_program, warmup=500)
    print("Ref: {:.6f} ms".format(latency))
    latency = profiler.do_bench(profiler.func, warmup=500)
    print("Tile-lang: {:.6f} ms".format(latency))
