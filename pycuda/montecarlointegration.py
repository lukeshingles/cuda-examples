import pycuda.autoinit
import pycuda.driver as drv
import pycuda.curandom as curand
import pycuda.reduction
# from pycuda.curandom import XORWOWRandomNumberGenerator
import numpy as np
from pycuda.compiler import SourceModule
# from pycuda import gpuarray
import timeit

# nvalues = 10 ** 7

block_size = 128
nblocks = 10000
nvalues = nblocks * block_size

krnl = pycuda.reduction.ReductionKernel(np.float32, neutral="0",
        reduce_expr="a+b", map_expr="x[i] * x[i] + y[i] * y[i] < 1. ? 1 : 0",
        arguments="float *x, float *y")

rng = curand.XORWOWRandomNumberGenerator()

mod = SourceModule("""
__global__ void gpucounts(float *a, float *b, int *counts)
{
  __shared__ int hitormiss[128];

  unsigned int tid = threadIdx.x;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;

  if ((a[i] * a[i] + b[i] * b[i]) < 1.)
  {
    hitormiss[tid] = 1;
  }
  else
  {
    hitormiss[tid] = 0;
  }
  __syncthreads();

  if (tid == 0)
  {
    for (unsigned int s = 0; s < blockDim.x; s++)
    {
      counts[blockIdx.x] += hitormiss[s];
    }
  }
  __syncthreads();
}
""")

gpucounts = mod.get_function("gpucounts")

def get_pi_cuda():
    a_device = rng.gen_uniform((nvalues,), dtype=np.float32)
    b_device = rng.gen_uniform((nvalues,), dtype=np.float32)

    dest = np.zeros(nvalues, dtype=np.int32)
    counts = np.zeros(nblocks, dtype=np.int32)
    # counts = np.int32(0)

    gpucounts(a_device, b_device, drv.Out(counts), grid=(nblocks, 1), block=(block_size, 1, 1))
    # print(np.sum(c), np.sum(dest))
    hitcount = np.sum(counts)
    pi_est = hitcount / nvalues * 4

    print(f'CUDA result: {pi_est}')


def get_pi_cuda_redkern():
    a_device = rng.gen_uniform((nvalues,), dtype=np.float32)
    b_device = rng.gen_uniform((nvalues,), dtype=np.float32)

    # a_device = curand.rand(nvalues, dtype=np.float32)
    # b_device = curand.rand(nvalues, dtype=np.float32)
    hitcount = krnl(a_device, b_device).get()

    pi_est = hitcount / nvalues * 4
    print(f'CUDA ReductionKernel result: {pi_est}')


def get_pi_numpy():
    data = np.random.rand(nvalues, 2)
    inside = len(np.argwhere(np.linalg.norm(data, axis=1) < 1))

    pi_est = float(inside) / nvalues * 4.
    print(f'numpy result: {pi_est}')

ntrials = 10
print(f'CUDA Kernel took {timeit.timeit(get_pi_cuda, number=ntrials)} s')
print(f'CUDA ReductionKernel took {timeit.timeit(get_pi_cuda_redkern, number=ntrials)} s')
print(f'numpy took {timeit.timeit(get_pi_numpy, number=ntrials)} s')
