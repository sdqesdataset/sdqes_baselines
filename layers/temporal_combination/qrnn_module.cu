#include <torch/extension.h>


__global__ void recurrent_forget_mult_kernel(float *dst, const float *f, const float *x, int SEQ, int BATCH, int HIDDEN)
{
  /*
  Note: destination is assumed to be one timestep longer than f or x where dst[0] = h_{-1}
  This means dst array has a separate index than that of f or x
  */
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = blockIdx.y * blockDim.y + threadIdx.y;
  if(hid >= HIDDEN || bid >= BATCH)
     return;
  //
  for (int ts = 0 + 1; ts < SEQ + 1; ts++) {
     // Good sanity check for debugging - only perform additions to a zeroed chunk of memory
     // Addition seems atomic or near atomic - you should get incorrect answers if doubling up via threads
     // Note: the index i needs to be offset by one as f[0] (f_t) is used for dst[1] (h_t) etc

     // To move timesteps, we step HIDDEN * BATCH
     // To move batches, we move HIDDEN
     // To move neurons, we move +- 1
     // Note: dst[dst_i] = ts * 100 + bid * 10 + hid; is useful for debugging

     int i           = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_i       = (ts - 0) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_iminus1 = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     dst[dst_i]      = f[i] * x[i];
     dst[dst_i]      += (1 - f[i]) * dst[dst_iminus1];
  }
}


__global__ void bwd_recurrent_forget_mult_kernel(const float *h, const float *f, const float *x, const float *gh, float *gf, float *gx, float *ghinit, int SEQ, int BATCH, int HIDDEN)
{
  /*
  Note: h is assumed to be one timestep longer than f, x, gf, gx, or gh where dst[0] = h_{-1}
  This means dst array has a separate index than that of f or x
  */
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = blockIdx.y * blockDim.y + threadIdx.y;
  if(hid >= HIDDEN || bid >= BATCH)
     return;
  //
  double running_f = 0;
  for (int ts = SEQ - 1 + 1; ts >= 0 + 1; ts--) {
     int i           = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_i       = (ts - 0) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_iminus1 = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     //
     running_f       += gh[dst_iminus1];
     // Gradient of X
     gx[i]           = f[i] * running_f;
     // Gradient of F
     gf[i]           = (x[i] - h[dst_iminus1]) * running_f;
     //
     // The line below is likely more numerically stable than (1 - f[i]) * running_f;
     running_f       = running_f - f[i] * running_f;
  }
  ghinit[bid * HIDDEN + hid] = running_f;
}


void recurrent_forget_mult(torch::Tensor dst, const torch::Tensor f, const torch::Tensor x, int SEQ, int BATCH, int HIDDEN){
    // tensor shape is seq_len x batch_size x hidden_size
    // block size is min of 512 and hidden_size
    const int block_size = std::min(HIDDEN, 512);
    // ceil of hidden_size / block_size
    const int num_blocks = (HIDDEN + block_size - 1) / block_size;
    recurrent_forget_mult_kernel<<<dim3(num_blocks, BATCH), block_size>>>(dst.data_ptr<float>(), f.data_ptr<float>(), x.data_ptr<float>(), SEQ, BATCH, HIDDEN);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in QRNN kernel: %s\n", cudaGetErrorString(err));
    }
}


void bwd_recurrent_forget_mult(const torch::Tensor h, const torch::Tensor f, const torch::Tensor x, const torch::Tensor gh, torch::Tensor gf, torch::Tensor gx, torch::Tensor ghinit, int SEQ, int BATCH, int HIDDEN){
    // tensor shape is seq_len x batch_size x hidden_size
    // block size is min of 512 and hidden_size
    const int block_size = std::min(HIDDEN, 512);
    // ceil of hidden_size / block_size
    const int num_blocks = (HIDDEN + block_size - 1) / block_size;
    bwd_recurrent_forget_mult_kernel<<<dim3(num_blocks, BATCH), block_size>>>(h.data_ptr<float>(), f.data_ptr<float>(), x.data_ptr<float>(), gh.data_ptr<float>(), gf.data_ptr<float>(), gx.data_ptr<float>(), ghinit.data_ptr<float>(), SEQ, BATCH, HIDDEN);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in QRNN kernel: %s\n", cudaGetErrorString(err));
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("recurrent_forget_mult", &recurrent_forget_mult, "Forward pass");
    m.def("bwd_recurrent_forget_mult", &bwd_recurrent_forget_mult, "Backward pass");
}