from .fast_ops import FastOps
from .tensor_functions import rand, Function
from . import operators


def tile(input, kernel):
    """
    Reshape an image tensor for 2D pooling

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        (:class:`Tensor`, int, int) : Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    tile_h = height // kh
    tile_w = width // kw

    input = input.contiguous()
    input = input.view(batch, channel, tile_h, kh, tile_w, kw)
    input = input.permute(0,1,2,4,3,5)
    input = input.contiguous()
    input = input.view(batch, channel, tile_h, tile_w, kw*kh)

    return input, tile_h, tile_w


def avgpool2d(input, kernel):
    """
    Tiled average pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    input, tile_h, tile_w = tile(input, kernel)
    input = input.mean(4)
    input = input.view(batch, channel, tile_h, tile_w)
    return input


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input, dim):
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx, input, dim):
        ctx.save_for_backward(input, dim)
        return max_reduce(input, dim)

    @staticmethod
    def backward(ctx, grad_output):
        "Backward of max should be argmax (see above)"
        input, dim = ctx.saved_values
        return grad_output*argmax(input, dim)


max = Max.apply


def softmax(input, dim):
    r"""
    Compute the softmax as a tensor.

    .. math::

        z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply softmax

    Returns:
        :class:`Tensor` : softmax tensor
    """
    # An elements can have terms on the demoninator from any of axis which is determined by dim.
    input = input.exp()
    sum_along_axis = input.sum(dim)
    return input/sum_along_axis


def logsoftmax(input, dim):
    r"""
    Compute the log of the softmax as a tensor.

    .. math::

        z_i = x_i - \log \sum_i e^{x_i}

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply log-softmax

    Returns:
        :class:`Tensor` : log of softmax tensor
    """
    return softmax(input,dim).log()


def maxpool2d(input, kernel):
    """
    Tiled max pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    input, tile_h, tile_w = tile(input, kernel)
    input = max(input, 4)
    input = input.view(batch, channel, tile_h, tile_w)
    return input


def dropout(input, rate, ignore=False):
    """
    Dropout positions based on random noise.

    Args:
        input (:class:`Tensor`): input tensor
        rate (float): probability [0, 1) of dropping out each position
        ignore (bool): skip dropout, i.e. do nothing at all

    Returns:
        :class:`Tensor` : tensor with randoom positions dropped out
    """
    if not ignore:
        bit_tensor = rand(input.shape, input.backend) > rate
        input = bit_tensor * input
    return input