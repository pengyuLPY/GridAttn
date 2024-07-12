
def shuffle_and_recovery(grid_stride):
    index_shuffle = [i for i in range(grid_stride)]
    index_recovery = [i for i in range(grid_stride)]

    random.shuffle(index_shuffle)
    for i in range(grid_stride):
        index_recovery[index_shuffle[i]] = i

    index_shuffle = np.array(index_shuffle)
    index_recovery = np.array(index_recovery)

    return index_shuffle, index_recovery

def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W)


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()



def nlc_to_grid(x, hw_shape, grid_stride=1, w_index=None, h_index=None):
    N, L, C = x.shape
    H, W = hw_shape
    x = nlc_to_nchw(x, hw_shape)
    assert H % grid_stride == 0 and W % grid_stride == 0, 'W:{}, H:{}, G:{}'.format(W, H, grid_stride)

    x = x.reshape(N, C, H, W // grid_stride, grid_stride)
    x = x.permute(0, 4, 1, 3, 2)  # N, G, C, W, H
    x = x.reshape(N, grid_stride, C, W // grid_stride, H // grid_stride, grid_stride)
    x = x.permute(0, 1, 5, 2, 4, 3)  # N, G, G, C, H, W

    if w_index is not None:
        x[:, :, :, :, :, 1:] = x[:, w_index, :, :, :, 1:]
    if h_index is not None:
        x[:, :, :, :, 1:, :] = x[:, :, h_index, :, 1:, :]
    x = x.reshape(N * grid_stride * grid_stride, C, H // grid_stride, W // grid_stride)

    x = nchw_to_nlc(x)
    return x, (H // grid_stride, W // grid_stride)


def grid_to_nlc(x, hw_shape, grid_stride=1, w_index=None, h_index=None):
    N, L, C = x.shape
    H, W = hw_shape
    x = nlc_to_nchw(x, hw_shape)
    assert N % (grid_stride * grid_stride) == 0, 'N:{}, G:{}'.format(N, grid_stride)

    x = x.reshape(N // grid_stride // grid_stride, grid_stride, grid_stride, C, H, W)  # N, G, G, C, H, W

    if h_index is not None:
        x[:, :, :, :, 1:, :] = x[:, :, h_index, :, 1:, :]
    if w_index is not None:
        x[:, :, :, :, :, 1:] = x[:, w_index, :, :, :, 1:]

    x = x.permute(0, 1, 3, 5, 4, 2)
    x = x.reshape(N // grid_stride // grid_stride, grid_stride, C, W, H * grid_stride)
    x = x.permute(0, 2, 4, 3, 1)
    x = x.reshape(N // grid_stride // grid_stride, C, H * grid_stride, W * grid_stride)

    x = nchw_to_nlc(x)
    return x, (H * grid_stride, W * grid_stride)