import numpy as np
from .tensor_data import (
    to_index,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)
from numba import njit, prange


# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def tensor_map(fn):
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

        * Main loop in parallel
        * All indices use numpy buffers
        * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        n = len(out)
        out_dims = len(out_shape)
        in_dims = len(in_shape)
        if out_dims == in_dims:
            eq_shape = True
            for i in range(out_dims):
                i1, i2 = out_shape[i], in_shape[i]
                if i1 != i2:
                    eq_shape = False
                    break
            eq_stride = True
            for i in range(out_dims):
                i1, i2 = out_strides[i], in_strides[i]
                if i1 != i2:
                    eq_stride = False
                    break
        else:
            eq_shape = False
            eq_stride = False
        for i in prange(n):
            out_index = np.zeros(out_dims)
            to_index(i,out_shape,out_index)
            out_idx = index_to_position(out_index, out_strides)
            if eq_shape and eq_stride:
                idx = out_idx
            else:                
                in_index = np.zeros(in_dims)
                if eq_shape:
                    to_index(out_idx,in_shape,in_index)
                else:
                    broadcast_index(out_index,out_shape,in_shape,in_index)
                idx = index_to_position(in_index,in_strides)
            x = in_storage[idx]
            out[out_idx] = fn(x)

    return njit(parallel=True)(_map)


def map(fn):
    """
    Higher-order tensor map function ::

      fn_map = map(fn)
      fn_map(a, out)
      out

    Args:
        fn: function from float-to-float to apply.
        a (:class:`Tensor`): tensor to map over
        out (:class:`Tensor`): optional, tensor data to fill in,
               should broadcast with `a`

    Returns:
        :class:`Tensor` : new tensor
    """

    # This line JIT compiles your tensor_map
    f = tensor_map(njit()(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

        * Main loop in parallel
        * All indices use numpy buffers
        * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.

    Returns:
        None : Fills in `out`
    """

    def _zip(out,out_shape,out_strides,a_storage,a_shape,a_strides,b_storage,b_shape,b_strides):
        n = len(out)
        out_dims = len(out_shape)
        in_dims_a = len(a_shape)
        in_dims_b = len(b_shape)

        if out_dims == in_dims_a and out_dims == in_dims_b:
            eq_shape = True
            for i in range(out_dims):
                i1, i2, i3 = out_shape[i], a_shape[i], b_shape[i]
                if i1 != i2 or i1 != i3:
                    eq_shape = False
                    break
            eq_stride = True
            for i in range(out_dims):
                i1, i2, i3 = out_strides[i], a_strides[i], b_strides[i]
                if i1 != i2 or i1 != i3:
                    eq_stride = False
                    break
        else:
            eq_shape = False
            eq_stride = False

        for i in prange(n):
            out_index = np.zeros(out_dims)
            to_index(i,out_shape,out_index)
            out_idx = index_to_position(out_index,out_strides)

            in_index_a = np.zeros(in_dims_a)
            if eq_shape:
                to_index(out_idx,a_shape,in_index_a)
            else:
                broadcast_index(out_index, out_shape, a_shape, in_index_a)
            idx_a = index_to_position(in_index_a, a_strides)
            x_a = a_storage[idx_a]

            in_index_b = np.zeros(in_dims_b)
            if eq_shape:
                to_index(out_idx,b_shape,in_index_b)
            else:
                broadcast_index(out_index, out_shape, b_shape, in_index_b)
            idx_b = index_to_position(in_index_b, b_strides)
            x_b = b_storage[idx_b]

            out[out_idx] = fn(x_a, x_b)

    return njit(parallel=True)(_zip)


def zip(fn):
    """
    Higher-order tensor zip function.

      fn_zip = zip(fn)
      c = fn_zip(a, b)

    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to zip over
        b (:class:`Tensor`): tensor to zip over

    Returns:
        :class:`Tensor` : new tensor data
    """
    f = tensor_zip(njit()(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

        * Main loop in parallel
        * All indices use numpy buffers
        * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`

    """

    def _reduce(out, out_shape, out_strides, a_storage, a_shape, a_strides, reduce_dim):
        n = len(out)
        out_dims = len(a_shape)
        for i in prange(n):
            out_index = np.zeros(out_dims)
            to_index(i,out_shape,out_index)
            out_idx = index_to_position(out_index,out_strides)

            reduce_dim_size = a_shape[reduce_dim]

            for j in range(reduce_dim_size):
                idx_a = out_index.copy()
                idx_a[reduce_dim] = j
                pos_a = index_to_position(idx_a, a_strides)
                out[out_idx] = fn(a_storage[pos_a], out[out_idx])

    return njit(parallel=True)(_reduce)


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      out = fn_reduce(a, dim)


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to reduce over
        dim (int): int of dim to reduce

    Returns:
        :class:`Tensor` : new tensor
    """

    f = tensor_reduce(njit()(fn))

    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = 1

        # Other values when not sum.
        out = a.zeros(tuple(out_shape))
        out._tensor._storage[:] = start

        f(*out.tuple(), *a.tuple(), dim)
        return out

    return ret


@njit(parallel=True, fastmath=True)
def tensor_matrix_multiply(out, out_shape, out_strides, a_storage, a_shape, a_strides, b_storage, b_shape, b_strides):
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as ::

        assert a_shape[-1] == b_shape[-2]

    Optimizations:

        * Outer loop in parallel
        * No index buffers or function calls
        * Inner loop should have no global writes, 1 multiply.


    Args:
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    n = len(out)

    for i in prange(n):
        out_index = out_shape.copy()
        temp_i = i + 0
        to_index(temp_i,out_shape,out_index)

        out_pos = index_to_position(out_index, out_strides)

        last_dim = a_shape[-1]

        for j in range(last_dim):
            temp_j = j + 0

            a_index = a_shape.copy()
            a_tmp_index = out_index.copy()
            a_tmp_index[-1] = temp_j
            broadcast_index(a_tmp_index,out_shape,a_shape,a_index)
            a_pos = index_to_position(a_index,a_strides)

            b_index = b_shape.copy()
            b_tmp_index = out_index.copy()
            b_tmp_index[-2] = temp_j
            broadcast_index(b_tmp_index,out_shape,b_shape,b_index)
            b_pos = index_to_position(b_index,b_strides)

            out[out_pos] += (a_storage[a_pos]*b_storage[b_pos])


def matrix_multiply(a, b):
    """
    Batched tensor matrix multiply ::

        for n:
          for i:
            for j:
              for k:
                out[n, i, j] += a[n, i, k] * b[n, k, j]

    Where n indicates an optional broadcasted batched dimension.

    Should work for tensor shapes of 3 dims ::

        assert a.shape[-1] == b.shape[-2]

    Args:
        a (:class:`Tensor`): tensor data a
        b (:class:`Tensor`): tensor data b

    Returns:
        :class:`Tensor` : new tensor data
    """

    # Make these always be a 3 dimensional multiply
    both_2d = 0
    if len(a.shape) == 2:
        a = a.contiguous().view(1, a.shape[0], a.shape[1])
        both_2d += 1
    if len(b.shape) == 2:
        b = b.contiguous().view(1, b.shape[0], b.shape[1])
        both_2d += 1
    both_2d = both_2d == 2

    ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
    ls.append(a.shape[-2])
    ls.append(b.shape[-1])
    assert a.shape[-1] == b.shape[-2]
    out = a.zeros(tuple(ls))

    tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

    # Undo 3d if we added it.
    if both_2d:
        out = out.view(out.shape[1], out.shape[2])
    return out


class FastOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply
