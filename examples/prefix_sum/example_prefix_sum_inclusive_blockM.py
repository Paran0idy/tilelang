# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import tilelang
import tilelang.language as T
from tilelang.engine.callback import register_cuda_postproc_callback

def prefix_sum_inclusive(M, N, blk_m):
    dtype = "float"

    @T.prim_func
    def main(A: T.Buffer((M, N), dtype), B: T.Buffer((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            A_shared = T.alloc_shared((blk_m, N), dtype)
            A_local = T.alloc_local((blk_m, N), dtype)
            tid = T.get_thread_binding()

            T.copy(A[bx * blk_m:(bx + 1) * blk_m, :], A_shared)
            
            steps = T.alloc_var("int32")
            steps = T.log2(T.Cast("float32", N)).astype("int32")
            
            for row in T.serial(blk_m):
                # Up-sweep phase
                for i in T.serial(steps):
                    offset = 1 << i
                    idx = (tid + 1) * offset * 2 - 1
                    if idx < N:
                        A_shared[row, idx] += A_shared[row, idx - offset]
            # Down-sweep phase
                for i in T.serial(steps - 1):
                    offset = (N // 2) // (1 << (i + 1))
                    idx = (tid + 1) * offset * 2 - 1
                    if idx + offset < N:
                        A_shared[row, idx + offset] += A_shared[row, idx]

            T.copy(A_shared, B[bx * blk_m:(bx + 1) * blk_m, :])

    return main




def ref_program(x):
    return torch.cumsum(x, dim=1)


if __name__ == "__main__":
    @register_cuda_postproc_callback
    def tilelang_callback_cuda_postproc(code, _):
        print(code) # print the final CUDA code
        code = "// modified by tilelang_callback_cuda_postproc\n" + code
        return code
    
    M, N, blk_m = 16, 512, 8
    program = prefix_sum_inclusive(M, N, blk_m)
    kernel = tilelang.compile(program, out_idx=-1, target="cuda", execution_backend="cython")
    print(kernel.get_kernel_source())
    
    # profiler = kernel.get_profiler()
    # profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    # print("All checks pass.")

    # latency = profiler.do_bench(ref_program, warmup=500)
    # print("Ref: {:.6f} ms".format(latency))
    # latency = profiler.do_bench(profiler.func, warmup=500)
    # print("Tile-lang: {:.6f} ms".format(latency))
