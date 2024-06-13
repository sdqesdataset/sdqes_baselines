import numpy as np
import torch
from torch.utils.data._utils.collate import string_classes

# Fix imports, but keep backwards compatibility
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs, int_classes
else:
    import collections.abc as container_abcs
    int_classes = int

def collate_with_pad(batch, allow_pad=True, pad_right=True):
    r"""Puts each data field into a tensor with outer dimension batch size.
    Will pad with zeros if there are sequences of varying lenghts in the batch BUT ONLY IF seq is first dimension.
    Will pad on the right by default, except when `pad_right==False`.
    """
    # print(batch[0].keys())
    # print(batch[0].values())
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).view(-1, *list(elem.size()))
        ###########################################
        # NEW: if tensors are different lengths PAD
        ###########################################
        it = iter(batch)
        elem_size = torch.Tensor.size(next(it))
        if not all(torch.Tensor.size(elem) == elem_size for elem in it):
            if allow_pad:
                # TRY TO PAD along the first dimension
                max_tensor_len = max(map(lambda tensor: tensor.size(1), batch))
                stacked_padded_tensors = torch.zeros(len(batch), max_tensor_len, elem_size[0], *elem_size[2:])
                for idx_in_batch, tensor in enumerate(batch):
                    if not pad_right:
                        stacked_padded_tensors[idx_in_batch, -1*tensor.size(1):, ...] = tensor.permute((1,0,2,3))   # pad on the left
                    else:
                        stacked_padded_tensors[idx_in_batch, :tensor.size(1), ...] = tensor.permute((1,0,2,3))      # pad on the right
                return stacked_padded_tensors.permute((0,2,1,3,4))
            else:
                raise RuntimeError('each element in list of batch should be of equal size')
        ###########################################
        # END NEW
        ###########################################
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
#             # array of string classes and object
#             if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
#                 raise TypeError(f"Unsupported type: {elem_type}")

            return collate_with_pad([torch.as_tensor(b) for b in batch], pad_right=pad_right)
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_with_pad([d[key] for d in batch], pad_right=pad_right) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_with_pad(samples, pad_right=pad_right) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_with_pad(samples, pad_right=pad_right) for samples in transposed]

    raise TypeError(f"Unsupported type: {elem_type}")
