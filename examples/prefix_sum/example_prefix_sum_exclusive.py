# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import tilelang
import tilelang.language as T
from tilelang.engine.callback import register_cuda_postproc_callback

tilelang.disable_cache()
def prefix_sum_exclusive(M, N, blk_m):
    dtype = "float"

    @T.prim_func
    def main(A: T.Buffer((M, N), dtype), B: T.Buffer((M, N), dtype)):
        with T.Kernel(M, threads=128) as bx:
            A_shared = T.alloc_shared((N), dtype)
            tid = T.get_thread_binding()

            T.copy(A[bx * blk_m:(bx + 1) * blk_m, :], A_shared)
            
            steps = T.alloc_var("int32")
            steps = T.log2(T.Cast("float32", N)).astype("int32")
            
            # Up-sweep phase
            for i in T.serial(steps):
                offset = 1 << i
                idx = tid * offset * 2 + offset - 1
                if tid < N // (2 * offset) and idx < N:
                    A_shared[idx + offset] += A_shared[idx]

            if tid == 0:
                A_shared[N - 1] = 0

            # Down-sweep phase
            for i in T.serial(steps):
                offset = N // (1 << (i + 1))
                idx = tid * offset * 2 + offset - 1
                # if tid < N // (2 * offset) and idx + offset < N:
                tmp = T.alloc_local([1], dtype)

                if idx + offset < N:
                    tmp[0] = A_shared[idx + offset]
                    A_shared[idx + offset] += A_shared[idx]
                    A_shared[idx] = tmp[0]

            T.copy(A_shared, B[bx * blk_m:(bx + 1) * blk_m, :])

    return main




def reference_program(x):
    return torch.cumsum(x, dim=1)



import re

class ScopeTracker:
    def __init__(self):
        self.scope_stack = []  # (scope_type, entry_brace_level)
        self.brace_level = 0
        self.sync_pattern = re.compile(r'__syncthreads\s*$\s*$\s*;')

    def update_scope(self, line: str):
        # 过滤注释
        code_line = re.sub(r'//.*', '', line).strip()
        if not code_line:
            return

        # 更新大括号层级
        delta_braces = code_line.count('{') - code_line.count('}')
        new_brace_level = self.brace_level + delta_braces
        
        # 检测 if 语句 (支持跨行)
        if re.search(r'\bif\s*\(', code_line):
            # 记录进入 if 时的层级
            entry_level = self.brace_level - code_line.count('{')
            self.scope_stack.append(('if', entry_level))
        
        # 更新层级
        self.brace_level = new_brace_level
        
        # 弹出过期作用域
        while self.scope_stack:
            _, entry_level = self.scope_stack[-1]
            if self.brace_level <= entry_level:
                self.scope_stack.pop()
            else:
                break

    @property
    def in_if_scope(self):
        return any(t[0] == 'if' for t in self.scope_stack)

if __name__ == "__main__":

    @register_cuda_postproc_callback
    def tilelang_callback_cuda_postproc(code, _):
        tracker = ScopeTracker()
        output = []
        
        for idx, line in enumerate(code.split('\n')):
            tracker.update_scope(line)
            # 保留原行格式
            if tracker.sync_pattern.search(line) and tracker.in_if_scope:
                continue
            if idx == 30 or idx == 32:
                continue
            output.append(line)
        
        return '\n'.join(output)
    

        
    M, N, blk_m = 2, 512, 1
    program = prefix_sum_exclusive(M, N, blk_m)
    # print(program)
    kernel = tilelang.compile(program, out_idx=-1, target="cuda", execution_backend="cython")
    print(kernel.get_kernel_source())

    # A = torch.ones(M, N).cuda()
    # out = kernel(A)
    # out_ref = reference_program(A)
    # print("Reference Output:")
    # print(out_ref)
    # print("Kernel Output:")
    # print(out)
