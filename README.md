# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


# TASK 3.1: 

Parallel check output:
```
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/tungnguyen/Documents/cornell/MLE/mod3-tungdnguyen/minitorch/fast_ops.py 
(168)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/tungnguyen/Documents/cornell/MLE/mod3-tungdnguyen/minitorch/fast_ops.py (168) 
---------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                | 
        out: Storage,                                                                        | 
        out_shape: Shape,                                                                    | 
        out_strides: Strides,                                                                | 
        in_storage: Storage,                                                                 | 
        in_shape: Shape,                                                                     | 
        in_strides: Strides,                                                                 | 
    ) -> None:                                                                               | 
        is_stride_aligned =  stride_aligned(out_strides, in_strides, out_shape, in_shape)    | 
        if is_stride_aligned:                                                                | 
            for i in prange(len(out)):-------------------------------------------------------| #0
                out[i] = fn(in_storage[i])                                                   | 
        else:                                                                                | 
            for i in prange(len(out)):          ---------------------------------------------| #1
                out_index: Index  = np.empty(MAX_DIMS, dtype=np.int32)                       | 
                in_index: Index  = np.empty(MAX_DIMS, dtype=np.int32)                        | 
                to_index(i, out_shape, out_index)                                            | 
                broadcast_index(out_index, out_shape, in_shape, in_index)                    | 
                in_pos = index_to_position(in_index, in_strides)                             | 
                out_pos = index_to_position(out_index, out_strides)                          | 
                out[out_pos] = fn(in_storage[in_pos])                                        | 
        return None                                                                          | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/tungnguyen/Documents/cornell/MLE/mod3-tungdnguyen/minitorch/fast_ops.py 
(182) is hoisted out of the parallel loop labelled #1 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: out_index: Index  = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/tungnguyen/Documents/cornell/MLE/mod3-tungdnguyen/minitorch/fast_ops.py 
(183) is hoisted out of the parallel loop labelled #1 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: in_index: Index  = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/tungnguyen/Documents/cornell/MLE/mod3-tungdnguyen/minitorch/fast_ops.py 
(217)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/tungnguyen/Documents/cornell/MLE/mod3-tungdnguyen/minitorch/fast_ops.py (217) 
----------------------------------------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                                                                             | 
        out: Storage,                                                                                                                                     | 
        out_shape: Shape,                                                                                                                                 | 
        out_strides: Strides,                                                                                                                             | 
        a_storage: Storage,                                                                                                                               | 
        a_shape: Shape,                                                                                                                                   | 
        a_strides: Strides,                                                                                                                               | 
        b_storage: Storage,                                                                                                                               | 
        b_shape: Shape,                                                                                                                                   | 
        b_strides: Strides,                                                                                                                               | 
    ) -> None:                                                                                                                                            | 
        is_stride_aligned =  stride_aligned(out_strides, a_strides, out_shape, a_shape) and stride_aligned(out_strides, b_strides, out_shape, b_shape)    | 
        if is_stride_aligned:                                                                                                                             | 
            for i in prange(len(out)):--------------------------------------------------------------------------------------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                                                                                                   | 
        else:                                                                                                                                             | 
            for i in prange(len(out)):--------------------------------------------------------------------------------------------------------------------| #3
                out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)                                                                                     | 
                a_index: Index  = np.empty(MAX_DIMS, dtype=np.int32)                                                                                      | 
                b_index: Index  = np.empty(MAX_DIMS, dtype=np.int32)                                                                                      | 
                to_index(i, out_shape, out_index)                                                                                                         | 
                broadcast_index(out_index, out_shape, a_shape, a_index)                                                                                   | 
                broadcast_index(out_index, out_shape, b_shape, b_index)                                                                                   | 
                a_pos = index_to_position(a_index, a_strides)                                                                                             | 
                b_pos = index_to_position(b_index, b_strides)                                                                                             | 
                out_pos = index_to_position(out_index, out_strides)                                                                                       | 
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])                                                                                     | 
        return None                                                                                                                                       | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/tungnguyen/Documents/cornell/MLE/mod3-tungdnguyen/minitorch/fast_ops.py 
(234) is hoisted out of the parallel loop labelled #3 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/tungnguyen/Documents/cornell/MLE/mod3-tungdnguyen/minitorch/fast_ops.py 
(235) is hoisted out of the parallel loop labelled #3 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: a_index: Index  = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/tungnguyen/Documents/cornell/MLE/mod3-tungdnguyen/minitorch/fast_ops.py 
(236) is hoisted out of the parallel loop labelled #3 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: b_index: Index  = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/tungnguyen/Documents/cornell/MLE/mod3-tungdnguyen/minitorch/fast_ops.py 
(269)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/tungnguyen/Documents/cornell/MLE/mod3-tungdnguyen/minitorch/fast_ops.py (269) 
----------------------------------------------------------------------|loop #ID
    def _reduce(                                                      | 
        out: Storage,                                                 | 
        out_shape: Shape,                                             | 
        out_strides: Strides,                                         | 
        a_storage: Storage,                                           | 
        a_shape: Shape,                                               | 
        a_strides: Strides,                                           | 
        reduce_dim: int,                                              | 
    ) -> None:                                                        | 
        reduce_dim_stride = a_strides[reduce_dim]                     | 
        a_shape_reduce_dim = a_shape[reduce_dim]                      | 
        for i in prange(len(out)):------------------------------------| #4
            out_index: Index  = np.empty(MAX_DIMS, dtype=np.int32)    | 
            a_index: Index  = np.empty(MAX_DIMS, dtype=np.int32)      | 
            out_shape[reduce_dim] = 1                                 | 
            to_index(i, out_shape, out_index)                         | 
            out_pos = index_to_position(out_index, out_strides)       | 
            a_index = out_index.copy()                                | 
            a_pos_initial = index_to_position(a_index, a_strides)     | 
            for j in range(0, a_shape_reduce_dim):                    | 
                a_pos = int(a_pos_initial + j*reduce_dim_stride)      | 
                out[out_pos] = fn(out[out_pos], a_storage[a_pos])     | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/tungnguyen/Documents/cornell/MLE/mod3-tungdnguyen/minitorch/fast_ops.py 
(281) is hoisted out of the parallel loop labelled #4 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: out_index: Index  = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
```

# Task 3.2

Output of parallel check:
```
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/tungnguyen/Documents/cornell/MLE/mod3-tungdnguyen/minitorch/fast_ops.py 
(296)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/tungnguyen/Documents/cornell/MLE/mod3-tungdnguyen/minitorch/fast_ops.py (296) 
-----------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                   | 
    out: Storage,                                                                              | 
    out_shape: Shape,                                                                          | 
    out_strides: Strides,                                                                      | 
    a_storage: Storage,                                                                        | 
    a_shape: Shape,                                                                            | 
    a_strides: Strides,                                                                        | 
    b_storage: Storage,                                                                        | 
    b_shape: Shape,                                                                            | 
    b_strides: Strides,                                                                        | 
) -> None:                                                                                     | 
    """NUMBA tensor matrix multiply function.                                                  | 
                                                                                               | 
    Should work for any tensor shapes that broadcast as long as                                | 
                                                                                               | 
    ```                                                                                        | 
    assert a_shape[-1] == b_shape[-2]                                                          | 
    ```                                                                                        | 
                                                                                               | 
    Optimizations:                                                                             | 
                                                                                               | 
    * Outer loop in parallel                                                                   | 
    * No index buffers or function calls                                                       | 
    * Inner loop should have no global writes, 1 multiply.                                     | 
                                                                                               | 
                                                                                               | 
    Args:                                                                                      | 
    ----                                                                                       | 
        out (Storage): storage for `out` tensor                                                | 
        out_shape (Shape): shape for `out` tensor                                              | 
        out_strides (Strides): strides for `out` tensor                                        | 
        a_storage (Storage): storage for `a` tensor                                            | 
        a_shape (Shape): shape for `a` tensor                                                  | 
        a_strides (Strides): strides for `a` tensor                                            | 
        b_storage (Storage): storage for `b` tensor                                            | 
        b_shape (Shape): shape for `b` tensor                                                  | 
        b_strides (Strides): strides for `b` tensor                                            | 
                                                                                               | 
    Returns:                                                                                   | 
    -------                                                                                    | 
        None : Fills in `out`                                                                  | 
                                                                                               | 
    """                                                                                        | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                     | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                     | 
                                                                                               | 
    for batch in prange(out_shape[0]):---------------------------------------------------------| #8
        for i in range(out_shape[1]):                                                          | 
            for j in range(out_shape[2]):                                                      | 
                value_at_ij = 0.0                                                              | 
                # number of cols in a = number of rows in b.                                   | 
                # Loops over each col in a belongs to batch, row i, col j.                     | 
                for col_no in range(a_shape[2]):                                               | 
                    a_pos = batch * a_batch_stride + i*a_strides[1] + col_no*a_strides[2]      | 
                    b_pos = batch * b_batch_stride + col_no * b_strides[1] + j*b_strides[2]    | 
                    value_at_ij += a_storage[a_pos] * b_storage[b_pos]                         | 
                out_pos = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]     | 
                out[out_pos] = value_at_ij                                                     | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #8).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

# TIMING REPORTS

Running timing.py we have: 

```
Running size 128
/usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
{'fast': 0.016060431798299152, 'gpu': 0.01486047108968099}
Running size 256
{'fast': 0.0965878168741862, 'gpu': 0.05833299954732259}
Running size 512
{'fast': 1.017174243927002, 'gpu': 0.22230259577433267}
Running size 1024
{'fast': 8.114876429239908, 'gpu': 1.0972613493601482}

Timing summary
Size: 64
    fast: 0.00383
    gpu: 0.00727
Size: 128
    fast: 0.01606
    gpu: 0.01486
Size: 256
    fast: 0.09659
    gpu: 0.05833
Size: 512
    fast: 1.01717
    gpu: 0.22230
Size: 1024
    fast: 8.11488
    gpu: 1.09726
```

