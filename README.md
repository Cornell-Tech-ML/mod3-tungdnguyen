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
(295)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/tungnguyen/Documents/cornell/MLE/mod3-tungdnguyen/minitorch/fast_ops.py (295)
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
    for batch in prange(out_shape[0]):---------------------------------------------------------| #5
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
loop(s) (originating from loops labelled: #5).
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

![Performance Comparison: CPU vs GPU](image.png)

# Dataset Reports

##  Smaller model

### HIDDEN = 10 for SIMPLE

#### CPU
41.5 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

```
Epoch  0  loss  7.483404331766127 correct 24
Epoch  10  loss  6.8448420704118424 correct 28
Epoch  20  loss  6.602892981822736 correct 39
Epoch  30  loss  6.218876130327848 correct 41
Epoch  40  loss  5.928031369346412 correct 43
Epoch  50  loss  5.147203098193613 correct 46
Epoch  60  loss  4.134719391977997 correct 47
Epoch  70  loss  4.388195208909912 correct 48
Epoch  80  loss  3.3243737255754136 correct 48
Epoch  90  loss  3.392702541951607 correct 48
Epoch  100  loss  2.389471956863857 correct 48
Epoch  110  loss  2.6434680739310803 correct 49
Epoch  120  loss  2.2150922016182677 correct 49
Epoch  130  loss  1.945748179340816 correct 50
Epoch  140  loss  2.460784470304225 correct 50
Epoch  150  loss  1.3880043418718053 correct 50
Epoch  160  loss  1.4957050298163757 correct 50
Epoch  170  loss  1.8937618777853324 correct 50
Epoch  180  loss  1.1774830397503278 correct 50
Epoch  190  loss  1.4126504371594686 correct 50
Epoch  200  loss  1.2886890703327687 correct 50
Epoch  210  loss  2.327680943594756 correct 50
Epoch  220  loss  0.7705574408259348 correct 50
Epoch  230  loss  1.3855869298874446 correct 50
Epoch  240  loss  1.4978230530070948 correct 50
Epoch  250  loss  0.9691843973877605 correct 50
Epoch  260  loss  0.7782353859378616 correct 50
Epoch  270  loss  0.35985377738107127 correct 50
Epoch  280  loss  0.936911972848123 correct 50
Epoch  290  loss  1.8770750204873945 correct 50
Epoch  300  loss  0.39874210599984683 correct 50
Epoch  310  loss  0.2836752439203043 correct 50
Epoch  320  loss  0.9939480823696953 correct 50
Epoch  330  loss  0.6461321457926915 correct 50
Epoch  340  loss  1.649781488857792 correct 50
Epoch  350  loss  0.9656828437110052 correct 50
Epoch  360  loss  1.4510002659176957 correct 50
Epoch  370  loss  0.9327991211876133 correct 50
Epoch  380  loss  0.4310815400270036 correct 50
Epoch  390  loss  0.42321897463106245 correct 50
Epoch  400  loss  1.35005760967803 correct 50
Epoch  410  loss  0.6265776505516398 correct 50
Epoch  420  loss  0.22143295261171114 correct 50
Epoch  430  loss  0.012600573010428618 correct 50
Epoch  440  loss  0.23079766266459303 correct 50
Epoch  450  loss  1.5553939531796273 correct 50
Epoch  460  loss  0.720663892104844 correct 50
Epoch  470  loss  1.423471886332606 correct 50
Epoch  480  loss  0.8870435205745876 correct 50
Epoch  490  loss  1.024682839509401 correct 50
```
#### GPU

11min 56s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

```
Epoch  0  loss  6.993476754714674 correct 24
Epoch  10  loss  6.352746927670055 correct 43
Epoch  20  loss  6.160830770051412 correct 40
Epoch  30  loss  5.119947005115567 correct 42
Epoch  40  loss  5.673172438058051 correct 45
Epoch  50  loss  3.9817099187527347 correct 48
Epoch  60  loss  3.1933980731388916 correct 49
Epoch  70  loss  3.911551137509905 correct 50
Epoch  80  loss  3.317160442410056 correct 50
Epoch  90  loss  2.8504492196551343 correct 50
Epoch  100  loss  2.709645693317059 correct 50
Epoch  110  loss  2.0320569211863377 correct 50
Epoch  120  loss  1.4566952287228179 correct 50
Epoch  130  loss  2.1494164782717258 correct 50
Epoch  140  loss  1.987649277558293 correct 50
Epoch  150  loss  0.9372778665325645 correct 50
Epoch  160  loss  1.588648007687075 correct 50
Epoch  170  loss  1.2581937837495236 correct 50
Epoch  180  loss  1.0652679270052334 correct 50
Epoch  190  loss  1.2868219758600028 correct 50
Epoch  200  loss  1.9004794209515987 correct 50
Epoch  210  loss  1.5516417137523235 correct 50
Epoch  220  loss  0.9931084905656081 correct 50
Epoch  230  loss  0.6883851963883321 correct 50
Epoch  240  loss  0.42228538092826917 correct 50
Epoch  250  loss  0.5947023046650619 correct 50
Epoch  260  loss  0.9505515029657386 correct 50
Epoch  270  loss  1.4479528023676056 correct 50
Epoch  280  loss  1.2749421551114084 correct 50
Epoch  290  loss  0.670858417673403 correct 50
Epoch  300  loss  1.0285644409366421 correct 50
Epoch  310  loss  1.086371179237435 correct 50
Epoch  320  loss  0.9164322562745063 correct 50
Epoch  330  loss  0.5472353956852799 correct 50
Epoch  340  loss  0.5735065329045202 correct 50
Epoch  350  loss  1.0524559012010841 correct 50
Epoch  360  loss  0.7141210681995358 correct 50
Epoch  370  loss  0.42969662904555356 correct 50
Epoch  380  loss  0.4300045319403469 correct 50
Epoch  390  loss  0.18419989165656933 correct 50
Epoch  400  loss  0.27741352024187416 correct 50
Epoch  410  loss  0.2743755043864088 correct 50
Epoch  420  loss  0.30553543646824477 correct 50
Epoch  430  loss  0.5484966144251509 correct 50
Epoch  440  loss  0.4063365606236342 correct 50
Epoch  450  loss  0.3706107415277497 correct 50
Epoch  460  loss  0.3303009302740909 correct 50
Epoch  470  loss  0.6788202350839774 correct 50
Epoch  480  loss  0.06778482035512697 correct 50
Epoch  490  loss  0.6853164226667834 correct 50
```
### HIDDEN = 100 for XOR
#### CPU

49 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

```
Epoch  0  loss  7.583412462350911 correct 11
Epoch  10  loss  6.605144766587477 correct 30
Epoch  20  loss  6.507064658533118 correct 33
Epoch  30  loss  6.181716783863916 correct 35
Epoch  40  loss  5.375613031844569 correct 37
Epoch  50  loss  5.780291872780515 correct 40
Epoch  60  loss  5.52174612748491 correct 40
Epoch  70  loss  5.874888404313531 correct 40
Epoch  80  loss  5.084720975449385 correct 40
Epoch  90  loss  4.893956063183717 correct 40
Epoch  100  loss  5.313877227239406 correct 40
Epoch  110  loss  5.486427144910571 correct 42
Epoch  120  loss  3.800658533783151 correct 42
Epoch  130  loss  4.948073337269821 correct 43
Epoch  140  loss  5.151419840198282 correct 44
Epoch  150  loss  4.340160542153836 correct 43
Epoch  160  loss  3.8779710070582927 correct 44
Epoch  170  loss  4.0632658041134455 correct 44
Epoch  180  loss  3.5693211582187585 correct 43
Epoch  190  loss  3.887979849860635 correct 44
Epoch  200  loss  2.641155028283863 correct 45
Epoch  210  loss  3.0224789088299415 correct 45
Epoch  220  loss  2.6915302678355535 correct 45
Epoch  230  loss  3.1347086758263454 correct 46
Epoch  240  loss  3.4454655268449352 correct 45
Epoch  250  loss  3.8901209007812674 correct 45
Epoch  260  loss  2.2649946425232814 correct 45
Epoch  270  loss  3.2756074351134905 correct 45
Epoch  280  loss  2.480963520890281 correct 46
Epoch  290  loss  3.0850170224154096 correct 46
Epoch  300  loss  3.3539348865787466 correct 46
Epoch  310  loss  2.9736408381534574 correct 47
Epoch  320  loss  3.422065297819108 correct 46
Epoch  330  loss  3.4406768879960308 correct 46
Epoch  340  loss  2.781816110128921 correct 47
Epoch  350  loss  3.5378861576867333 correct 47
Epoch  360  loss  3.4111103073858087 correct 47
Epoch  370  loss  3.2988434907516946 correct 47
Epoch  380  loss  2.1835097092802855 correct 47
Epoch  390  loss  1.4292821500171182 correct 47
Epoch  400  loss  2.401531099161597 correct 47
Epoch  410  loss  3.555555160984761 correct 47
Epoch  420  loss  2.24649239857824 correct 47
Epoch  430  loss  2.9073427763575936 correct 47
Epoch  440  loss  2.7324598013281562 correct 47
Epoch  450  loss  1.5221532652456997 correct 47
Epoch  460  loss  1.766938081480685 correct 47
Epoch  470  loss  2.4889104741719983 correct 47
Epoch  480  loss  1.433660680285786 correct 47
Epoch  490  loss  2.7632093388935326 correct 48
```

#### GPU

11min 48s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

```
Epoch  0  loss  7.089421049764587 correct 28
Epoch  10  loss  6.951299710635933 correct 37
Epoch  20  loss  5.226097830367639 correct 38
Epoch  30  loss  5.503330861939762 correct 39
Epoch  40  loss  5.327609927092724 correct 39
Epoch  50  loss  4.858894986865994 correct 39
Epoch  60  loss  5.918956136680352 correct 39
Epoch  70  loss  4.450449887353273 correct 39
Epoch  80  loss  3.9396130982195485 correct 40
Epoch  90  loss  4.527238000613851 correct 42
Epoch  100  loss  3.6976714022568222 correct 43
Epoch  110  loss  2.552198113640575 correct 41
Epoch  120  loss  3.9910108570697878 correct 42
Epoch  130  loss  4.037152886286446 correct 43
Epoch  140  loss  3.006674049761422 correct 44
Epoch  150  loss  3.7354410515323586 correct 44
Epoch  160  loss  3.6134727451236452 correct 44
Epoch  170  loss  3.923574140278213 correct 45
Epoch  180  loss  3.6744828251409123 correct 45
Epoch  190  loss  3.2796720189896638 correct 45
Epoch  200  loss  3.227380483390233 correct 45
Epoch  210  loss  3.668674673182611 correct 45
Epoch  220  loss  1.5669701792622526 correct 45
Epoch  230  loss  3.5159132097278203 correct 46
Epoch  240  loss  1.3561036806047195 correct 45
Epoch  250  loss  2.2940024822983833 correct 46
Epoch  260  loss  2.2948098859885095 correct 46
Epoch  270  loss  2.7408728481003184 correct 46
Epoch  280  loss  2.6645617842968883 correct 46
Epoch  290  loss  1.15464121041704 correct 47
Epoch  300  loss  2.062475626151511 correct 47
Epoch  310  loss  2.6792920323237523 correct 47
Epoch  320  loss  2.8195659584386674 correct 46
Epoch  330  loss  1.1476239996734336 correct 47
Epoch  340  loss  2.1980551209037933 correct 47
Epoch  350  loss  0.6986832799852485 correct 47
Epoch  360  loss  2.223299876283808 correct 47
Epoch  370  loss  2.1946930314568007 correct 47
Epoch  380  loss  1.7340918474155314 correct 47
Epoch  390  loss  0.9888000566187709 correct 47
Epoch  400  loss  1.3396817935701066 correct 47
Epoch  410  loss  2.3063976173227907 correct 47
Epoch  420  loss  1.1526287443358867 correct 47
Epoch  430  loss  2.23406626638251 correct 47
Epoch  440  loss  1.6834507220886212 correct 47
Epoch  450  loss  1.615496180307377 correct 47
Epoch  460  loss  2.150591617967106 correct 47
Epoch  470  loss  1.1396047631994632 correct 47
Epoch  480  loss  2.176907547326487 correct 47
Epoch  490  loss  2.1793950109543303 correct 47
```

### HIDDEN = 100 for Split
#### CPU

53.1 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

```
Epoch  0  loss  6.377646554786198 correct 25
Epoch  10  loss  6.235422836744814 correct 37
Epoch  20  loss  5.152381782816675 correct 38
Epoch  30  loss  5.651935968251932 correct 41
Epoch  40  loss  6.530192021231708 correct 38
Epoch  50  loss  5.288667986823268 correct 39
Epoch  60  loss  5.347132784628141 correct 38
Epoch  70  loss  5.014643546286866 correct 40
Epoch  80  loss  4.979047330942038 correct 41
Epoch  90  loss  5.356152646602814 correct 40
Epoch  100  loss  5.658333661303707 correct 40
Epoch  110  loss  5.325657199421452 correct 41
Epoch  120  loss  5.884163273924158 correct 41
Epoch  130  loss  5.490700832489042 correct 41
Epoch  140  loss  3.3695343045745645 correct 41
Epoch  150  loss  4.345735948574794 correct 41
Epoch  160  loss  5.020444409783333 correct 41
Epoch  170  loss  5.684987446089625 correct 41
Epoch  180  loss  3.296897604068014 correct 41
Epoch  190  loss  3.3029430498076064 correct 41
Epoch  200  loss  2.7861823912313946 correct 41
Epoch  210  loss  5.096023322292369 correct 41
Epoch  220  loss  2.5367534891800885 correct 41
Epoch  230  loss  3.371592595304398 correct 41
Epoch  240  loss  4.788920833040457 correct 41
Epoch  250  loss  4.276230590582737 correct 41
Epoch  260  loss  3.9277480714709725 correct 42
Epoch  270  loss  3.0381289484592635 correct 42
Epoch  280  loss  2.0002152700562683 correct 44
Epoch  290  loss  4.370978225734692 correct 44
Epoch  300  loss  3.1056634649642034 correct 45
Epoch  310  loss  3.0813191491775043 correct 44
Epoch  320  loss  2.2629147186977288 correct 45
Epoch  330  loss  2.1566958360464423 correct 44
Epoch  340  loss  2.4160729086717687 correct 45
Epoch  350  loss  2.831693607732686 correct 45
Epoch  360  loss  2.5481266418025292 correct 45
Epoch  370  loss  3.1554347781796195 correct 46
Epoch  380  loss  2.682884511988155 correct 47
Epoch  390  loss  2.4923691663121352 correct 46
Epoch  400  loss  1.977268245131075 correct 47
Epoch  410  loss  1.5194609130844958 correct 47
Epoch  420  loss  2.882662907816533 correct 48
Epoch  430  loss  1.6196283328736019 correct 48
Epoch  440  loss  2.811049776912415 correct 48
Epoch  450  loss  1.562716502754944 correct 48
Epoch  460  loss  2.594028591951245 correct 48
Epoch  470  loss  1.9709486257623925 correct 49
Epoch  480  loss  2.686678909793894 correct 49
Epoch  490  loss  1.2001968569315153 correct 49
```

#### GPU

11min 53s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

```
Epoch  0  loss  6.6930802707113255 correct 30
Epoch  10  loss  6.038972684706414 correct 32
Epoch  20  loss  5.682664427427305 correct 36
Epoch  30  loss  7.490730398277062 correct 41
Epoch  40  loss  5.1687168241399215 correct 44
Epoch  50  loss  7.893966527230845 correct 44
Epoch  60  loss  4.757919680529004 correct 44
Epoch  70  loss  7.535883476107453 correct 43
Epoch  80  loss  3.780963849942431 correct 43
Epoch  90  loss  6.012600198223824 correct 43
Epoch  100  loss  6.498417354741198 correct 43
Epoch  110  loss  5.606429597350228 correct 43
Epoch  120  loss  4.803974680871708 correct 43
Epoch  130  loss  5.3319075824795075 correct 43
Epoch  140  loss  3.624970075072811 correct 43
Epoch  150  loss  3.3084170914172946 correct 43
Epoch  160  loss  5.411216112021606 correct 43
Epoch  170  loss  6.645352575746081 correct 43
Epoch  180  loss  5.331337655610458 correct 43
Epoch  190  loss  3.1567079888396257 correct 43
Epoch  200  loss  4.506989676137776 correct 43
Epoch  210  loss  3.631887459404876 correct 43
Epoch  220  loss  4.1840929361291375 correct 43
Epoch  230  loss  7.4090295070323435 correct 42
Epoch  240  loss  4.903064426391554 correct 42
Epoch  250  loss  6.493727150390624 correct 43
Epoch  260  loss  3.3122554787284333 correct 43
Epoch  270  loss  3.7861795444745985 correct 43
Epoch  280  loss  4.974799117667241 correct 43
Epoch  290  loss  4.131802449386985 correct 43
Epoch  300  loss  4.567889394216973 correct 43
Epoch  310  loss  2.8587594618931877 correct 43
Epoch  320  loss  5.048026227306921 correct 43
Epoch  330  loss  4.480149774336068 correct 43
Epoch  340  loss  3.4902234852339697 correct 43
Epoch  350  loss  3.256237717823784 correct 44
Epoch  360  loss  2.776354904711753 correct 45
Epoch  370  loss  3.2043314142272505 correct 45
Epoch  380  loss  2.9424731918157185 correct 45
Epoch  390  loss  1.2705573199773732 correct 45
Epoch  400  loss  3.0124828272705275 correct 45
Epoch  410  loss  2.3983356459702483 correct 45
Epoch  420  loss  2.26546712235256 correct 45
Epoch  430  loss  2.8381989610672127 correct 46
Epoch  440  loss  1.896244649879354 correct 46
Epoch  450  loss  3.124381252943537 correct 46
Epoch  460  loss  3.4287335231706604 correct 46
Epoch  470  loss  2.70418646252186 correct 46
Epoch  480  loss  3.027515578593645 correct 46
Epoch  490  loss  1.180861278381959 correct 46
```

## Bigger Model (HIDDEN = 200)

### HIDDEN = 200 for SIMPLE

#### CPU

1min 5s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

```
Epoch  0  loss  6.770000679782351 correct 39
Epoch  10  loss  5.849276094360011 correct 37
Epoch  20  loss  3.9437080276404544 correct 38
Epoch  30  loss  5.2623264036271165 correct 38
Epoch  40  loss  7.764518245668111 correct 38
Epoch  50  loss  5.846012148769668 correct 39
Epoch  60  loss  3.9974021798067794 correct 39
Epoch  70  loss  5.0088946863008115 correct 40
Epoch  80  loss  4.209146554113109 correct 41
Epoch  90  loss  3.2836392563144168 correct 42
Epoch  100  loss  4.532736328958693 correct 41
Epoch  110  loss  4.885429435363818 correct 42
Epoch  120  loss  2.670129127387753 correct 43
Epoch  130  loss  3.1552684950537806 correct 45
Epoch  140  loss  3.415814954215819 correct 46
Epoch  150  loss  3.658693802733835 correct 46
Epoch  160  loss  4.228524015383792 correct 47
Epoch  170  loss  3.2810400842040357 correct 47
Epoch  180  loss  3.3587229192953862 correct 46
Epoch  190  loss  4.31620446608944 correct 48
Epoch  200  loss  3.267662149899528 correct 48
Epoch  210  loss  2.2491425035624384 correct 45
Epoch  220  loss  2.5409143399308363 correct 48
Epoch  230  loss  1.8761677734494304 correct 47
Epoch  240  loss  2.1591951387880957 correct 48
Epoch  250  loss  2.2744242437991344 correct 46
Epoch  260  loss  2.4074443028800943 correct 46
Epoch  270  loss  1.959334944915488 correct 48
Epoch  280  loss  2.0085020877045863 correct 48
Epoch  290  loss  2.6347581910725766 correct 49
Epoch  300  loss  4.729948795834368 correct 49
Epoch  310  loss  2.147405463609506 correct 49
Epoch  320  loss  2.407292659822832 correct 48
Epoch  330  loss  1.940346551890634 correct 49
Epoch  340  loss  2.3177375773552518 correct 49
Epoch  350  loss  2.2176319746479463 correct 49
Epoch  360  loss  1.8191332262007218 correct 48
Epoch  370  loss  2.5612008083866717 correct 49
Epoch  380  loss  1.85794778687675 correct 49
Epoch  390  loss  2.028458935334294 correct 49
Epoch  400  loss  1.5503642207300508 correct 49
Epoch  410  loss  2.1208010440773433 correct 49
Epoch  420  loss  1.329743855215669 correct 49
Epoch  430  loss  1.896117097922403 correct 48
Epoch  440  loss  1.767168679034887 correct 49
Epoch  450  loss  2.881830577227438 correct 49
Epoch  460  loss  1.4593687954169159 correct 48
Epoch  470  loss  2.5214853764896596 correct 49
Epoch  480  loss  1.9374406241492674 correct 49
Epoch  490  loss  2.9650976546982486 correct 49
```

#### GPU

12min 27s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

```
Epoch  0  loss  6.681258199027674 correct 35
Epoch  10  loss  6.541202670385444 correct 36
Epoch  20  loss  5.976467665221939 correct 34
Epoch  30  loss  6.402332151470243 correct 40
Epoch  40  loss  6.337041905382911 correct 38
Epoch  50  loss  5.903244763318708 correct 40
Epoch  60  loss  5.060476479925579 correct 41
Epoch  70  loss  5.660292562932726 correct 43
Epoch  80  loss  5.313254882092373 correct 40
Epoch  90  loss  6.059441877058076 correct 43
Epoch  100  loss  5.6957010010849185 correct 42
Epoch  110  loss  5.037113745804632 correct 44
Epoch  120  loss  4.919130042153474 correct 44
Epoch  130  loss  4.397414556377087 correct 46
Epoch  140  loss  4.650417689998409 correct 44
Epoch  150  loss  3.6558909796599908 correct 46
Epoch  160  loss  5.184687648499734 correct 44
Epoch  170  loss  3.29827735982941 correct 44
Epoch  180  loss  4.314139934645128 correct 44
Epoch  190  loss  3.971475703255355 correct 45
Epoch  200  loss  3.47318330438815 correct 44
Epoch  210  loss  5.015605373394857 correct 45
Epoch  220  loss  2.93758478057758 correct 45
Epoch  230  loss  5.742534914709317 correct 47
Epoch  240  loss  3.7596849973700674 correct 47
Epoch  250  loss  2.3105190940207496 correct 46
Epoch  260  loss  3.5771993641085564 correct 47
Epoch  270  loss  3.3414425514983312 correct 46
Epoch  280  loss  3.720701725927076 correct 45
Epoch  290  loss  2.870990884679848 correct 48
Epoch  300  loss  3.610980291202563 correct 46
Epoch  310  loss  3.5324776654197674 correct 47
Epoch  320  loss  3.0839670538369615 correct 48
Epoch  330  loss  1.682744503269514 correct 48
Epoch  340  loss  2.7664992744632237 correct 48
Epoch  350  loss  1.7788908906595433 correct 48
Epoch  360  loss  2.8373232261835173 correct 49
Epoch  370  loss  2.3891955951767967 correct 49
Epoch  380  loss  1.7325657354339228 correct 47
Epoch  390  loss  1.6009547490618672 correct 47
Epoch  400  loss  1.666967507080412 correct 49
Epoch  410  loss  1.934184890295079 correct 46
Epoch  420  loss  2.8522085259063736 correct 47
Epoch  430  loss  1.6388276290264 correct 47
Epoch  440  loss  2.4898841504153744 correct 49
Epoch  450  loss  0.8075758766197875 correct 49
Epoch  460  loss  1.2629110207209215 correct 46
Epoch  470  loss  2.7258589586994724 correct 47
Epoch  480  loss  3.0773911930202664 correct 46
Epoch  490  loss  1.564583145380864 correct 46
```
