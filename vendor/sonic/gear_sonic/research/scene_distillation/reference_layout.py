"""Decode the released SONIC G1 encoder packing without changing the native policy.

command_multi_future concatenates flattened q[10,29] and qdot[10,29], then
its observation reshapes this 580-vector to [10,58]. Relative orientation6 is
concatenated to each resulting block. Thus these blocks are NOT physical frames.
"""

import torch


def unpack_reference(packed):
    if packed.shape[-1] != 640:
        raise ValueError("Expected native packed 640D reference")
    blocks = packed.reshape(*packed.shape[:-1], 10, 64)
    qv = blocks[..., :58].reshape(*packed.shape[:-1], 580)
    q = qv[..., :290].reshape(*packed.shape[:-1], 10, 29)
    v = qv[..., 290:].reshape(*packed.shape[:-1], 10, 29)
    return torch.cat([q, v, blocks[..., 58:]], -1)


def pack_reference(frames):
    if frames.shape[-2:] != (10, 64):
        raise ValueError("Expected ten physical 64D frames")
    prefix = frames.shape[:-2]
    qv = torch.cat(
        [frames[..., :29].reshape(*prefix, 290), frames[..., 29:58].reshape(*prefix, 290)], -1
    ).reshape(*prefix, 10, 58)
    return torch.cat([qv, frames[..., 58:]], -1).reshape(*prefix, 640)
