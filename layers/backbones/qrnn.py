# based on https://github.com/salesforce/pytorch-qrnn
import os
import torch
from torch import nn
from torch.utils.cpp_extension import load
import math
import CLIP.clip.model as clip_model
from collections import namedtuple
from torch.cuda.amp import custom_fwd, custom_bwd

import importlib.util

# check if the extension is has been pre-compiled
# TODO: solve environment issues
spec = importlib.util.find_spec('my_extension')
if spec is None:
    # compile the extension if it's not pre-compiled (eg. with `python setup.py install`)
    print('Did not find a pre-compiled extension for QRNN, compiling now...')
    qrnn = load(name='my_extension', sources=['layers/temporal_combination/qrnn_module.cu'])
else:
    # import the extension if it's been pre-compiled
    print('Found a pre-compiled extension for QRNN, importing now...')
    # Update the LD_LIBRARY_PATH environment variable
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    qrnn = importlib.import_module('my_extension')


class GPUForgetMult(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, f, x, hidden_init=None):
        seq_size, batch_size, hidden_size = f.size()
        result = f.new(seq_size + 1, batch_size, hidden_size)
        # We only zero the result array (result[0]) if we don't set a hidden initial state
        # All other values (result[1:]) are overwritten by default
        if hidden_init is not None: result[0, :, :] = hidden_init
        else: result = result.zero_()
        ##
        qrnn.recurrent_forget_mult(result, f, x, seq_size, batch_size, hidden_size)
        ctx.save_for_backward(f, x, hidden_init, result)
        return result[1:, :, :]

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_h):
        f, x, hidden_init, h = ctx.saved_tensors
        ###
        seq_size, batch_size, hidden_size = f.size()
        # Zeroing is not necessary as these will be overwritten
        grad_f = f.new(*f.size())
        grad_x = f.new(*f.size())
        grad_h_init = f.new(batch_size, hidden_size)
        ##
        qrnn.bwd_recurrent_forget_mult(h, f, x, grad_h, grad_f, grad_x, grad_h_init, seq_size, batch_size, hidden_size)
        if hidden_init is not None:
            return grad_f, grad_x, grad_h_init
        return grad_f, grad_x


class CPUForgetMult(nn.Module):
    @staticmethod
    def forward(f, x, hidden_init=None):
        result = []
        ###
        forgets = f.split(1, dim=0)
        prev_h = hidden_init
        for i, h in enumerate((f * x).split(1, dim=0)):
            if prev_h is not None: h = h + (1 - forgets[i]) * prev_h
            # h is (1, batch, hidden) when it needs to be (batch_hidden)
            # Calling squeeze will result in badness if batch size is 1
            h = h.view(h.size()[1:])
            result.append(h)
            prev_h = h
        ###
        return torch.stack(result)

class QRNNLayer(nn.Module):
    r"""Applies a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.

    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h. If not specified, the input size is used.
        window: Defines the size of the convolutional window (how many previous tokens to look when computing the QRNN values). Default: 1.
        output_gate: If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.

    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (batch, hidden_size): tensor containing the initial hidden state for the QRNN.

    Outputs: output, h_n
        - output (seq_len, batch, hidden_size): tensor containing the output of the QRNN for each timestep.
        - h_n (batch, hidden_size): tensor containing the hidden state for t=seq_len
    """

    def __init__(self, input_size, hidden_size=None, lookback_window=1, lookahead_window=0, output_gate=False, dilation=1):
        super().__init__()

        if output_gate: raise NotImplementedError()

        self.lookback_window = lookback_window
        self.lookahead_window = lookahead_window
        self.hidden_size = hidden_size if hidden_size else input_size
        self.output_gate = output_gate
        assert dilation == 1, 'Dilation is not yet supported'

        self.conv1d_f = nn.Conv1d(
            in_channels=input_size,
            out_channels=self.hidden_size,
            kernel_size=lookback_window + 1 + lookahead_window,
            stride=1,
            dilation=dilation,
        )   # expects batch_size, in_channels, seq_len
        self.conv1d_z = nn.Conv1d(
            in_channels=input_size,
            out_channels=self.hidden_size,
            kernel_size=lookback_window + 1 + lookahead_window,
            stride=1,
            dilation=dilation,
        )   # expects batch_size, in_channels, seq_len

        self.gelu = clip_model.QuickGELU()

        if os.environ.get('STREAM_VAL', False):
            # Buffers to store the last lookback_window frames and hidden state
            self.input_buffer = None
            self.hidden_buffer = None

    def reset_buffers(self):
        self.input_buffer = None
        self.hidden_buffer = None

    def forward(self, X, hidden=None):
        # X [seq_len, batch_size, input_size]
        seq_len, batch_size, in_channels = X.size()
        save_memory = os.environ.get('STREAM_VAL', False)

        if save_memory:
            assert self.lookahead_window == 0, 'Lookahead window is not supported when running in stream mode'

            if self.input_buffer is None:   # first iteration (beginning of video), initialize the buffer
                self.input_buffer = torch.zeros((self.lookback_window, batch_size, in_channels), device=X.device)

            X = torch.cat((self.input_buffer, X), dim=0)
            self.input_buffer = X[-self.lookback_window:, :, :] # Save the last lookback_window frames for the next iteration

            if self.hidden_buffer is None:
                self.hidden_buffer = torch.zeros((1, batch_size, self.hidden_size), device=X.device)

            hidden = self.hidden_buffer

            X_pad = X.permute(1, 2, 0)  # [batch, input_size, seq_len + lookback_window]
        else:
            X_pad = nn.functional.pad(
                X,
                (0, 0, 0, 0, self.lookback_window, self.lookahead_window),
                "constant",
                0
            )     # => [seq_len + lookback_window + lookahead_window, batch, input_size]
            X_pad = X_pad.permute(1, 2, 0)   # => [batch, input_size, seq_len + lookback_window + lookahead_window]

        # Convert the tensor back to (seq_len, batch, len([Z, F]) * hidden_size)
        if self.output_gate:
            raise NotImplementedError()
            Z, F, O = Y.chunk(3, dim=2)
        else:
            # compute the output logits
            Z = self.conv1d_z(X_pad)                # => [batch, hidden_size, seq_len]
            Z = Z.permute(2, 0, 1)              # => [seq_len, batch, hidden_size]
            F = self.conv1d_f(X_pad)                # => [batch, hidden_size, seq_len]
            F = F.permute(2, 0, 1)              # => [seq_len, batch, hidden_size]

        Z = self.gelu(Z)      # [seq_len, batch, hidden_size]
        F = torch.sigmoid(F)     # [seq_len, batch, hidden_size]

        Z = Z.contiguous()
        F = F.contiguous()

        if X.is_cuda:
            if hidden is None:
                C = GPUForgetMult.apply(F, Z)
            else:
                C = GPUForgetMult.apply(F, Z, hidden)
        else:
            C = CPUForgetMult.forward(F, Z, hidden)

        if self.output_gate:
            raise NotImplementedError()
            H = torch.sigmoid(O) * C
        else:
            H = C

        if save_memory:
            self.hidden_buffer = C[-1:, :, :]   # Save the last hidden state for the next iteration

        return H, C[-1:, :, :]

class QRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, layers=None, 
                 dilation=1, **kwargs):
        assert batch_first == False, 'Batch first mode is not yet supported'
        assert bias == True, 'Removing underlying bias is not yet supported'

        super().__init__()

        self.num_directions = 2 if bidirectional else 1

        self.layers = torch.nn.ModuleList(
            [
                QRNNLayer(
                    input_size=input_size if l == 0 else hidden_size,
                    hidden_size=hidden_size,
                    dilation=dilation,
                    **kwargs
                )
                for l in range(num_layers)
            ]
        )

        if bidirectional:
            self.layers.extend(
                [
                    QRNNLayer(
                        input_size=input_size if l == 0 else hidden_size,
                        hidden_size=hidden_size,
                        dilation=dilation,
                        **kwargs
                    )
                    for l in range(num_layers)
                ]
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = len(layers) if layers else num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

    def reset(self):
        [layer.reset() for layer in self.layers]

    def forward(self, input, hidden=None):
        next_hidden = []

        for i in range(0, self.num_layers * self.num_directions, self.num_directions):
            layer_forward = self.layers[i]
            input_forward, hn_forward = layer_forward(
                input, None if hidden is None else hidden[i]
            )

            if self.bidirectional:
                layer_backward = self.layers[self.num_layers + i]
                input_backward, hn_backward = layer_backward(
                    input.flip(0), None if hidden is None else hidden[self.num_layers + i]
                )
                input_backward = input_backward.flip(0)

                input = torch.cat([input_forward, input_backward], dim=2)
                next_hidden.extend([hn_forward, hn_backward])
            else:
                input = input_forward
                next_hidden.append(hn_forward)

            if i < (self.num_layers * self.num_directions) - self.num_directions:
                input = torch.nn.functional.dropout(
                    input, p=self.dropout, training=self.training, inplace=False
                )

        next_hidden = torch.cat(next_hidden, 0).view(
            self.num_layers * self.num_directions, *next_hidden[0].size()[-2:]
        )

        return input, next_hidden