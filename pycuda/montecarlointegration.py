import pycuda.autoinit
import pycuda.driver as drv
import pycuda.curandom as curand
import pycuda.reduction
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
  __shared__ bool hitormiss[128];

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
    counts[blockIdx.x] = hitormiss[0] ? 1 : 0;
    for (unsigned int s = 1; s < blockDim.x; s++)
    {
      if (hitormiss[s])
        counts[blockIdx.x] += 1;
    }
  }
  __syncthreads();
}
""")

gpucounts = mod.get_function("gpucounts")

def get_pi_cuda():
    a_device = rng.gen_uniform((nvalues,), dtype=np.float32)
    b_device = rng.gen_uniform((nvalues,), dtype=np.float32)

    counts = np.empty(nblocks, dtype=np.int32)

    gpucounts(a_device, b_device, drv.Out(counts), grid=(nblocks, 1), block=(block_size, 1, 1))
    hitcount = np.sum(counts)
    return hitcount


def get_pi_cuda_redkern():
    a_device = rng.gen_uniform((nvalues,), dtype=np.float32)
    b_device = rng.gen_uniform((nvalues,), dtype=np.float32)

    # a_device = curand.rand(nvalues, dtype=np.float32)
    # b_device = curand.rand(nvalues, dtype=np.float32)
    hitcount = krnl(a_device, b_device).get()
    return hitcount


def get_pi_numpy():
    data = np.random.rand(nvalues, 2)
    hitcount = len(np.argwhere(np.linalg.norm(data, axis=1) < 1))
    return hitcount

start = drv.Event()
end = drv.Event()
n_iter = 50

for f, label in [(get_pi_cuda, 'CUDA Kernel'), (get_pi_cuda_redkern, 'CUDA ReductionKernel'), (get_pi_numpy, 'numpy')]:
    start.record()
    start.synchronize()
    hitcount = 0
    for i in range(n_iter):
        hitcount += f()

    end.record()
    end.synchronize()
    timeseconds = start.time_till(end) * 1e-3
    pi_est = hitcount / nvalues / n_iter * 4

    print(f'{label} after {n_iter} iterations:')
    print(f'  pi_est {pi_est:.8f} time {timeseconds:3.4f} s ({n_iter * nvalues / timeseconds:.1e} evals/sec)')
