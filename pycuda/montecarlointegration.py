import pycuda.autoinit
import pycuda.driver as drv
import pycuda.curandom as curand
import pycuda.reduction
import numpy as np
from pycuda.compiler import SourceModule
# from pycuda import gpuarray
import timeit

block_size = 128
nblocks = 100000
nvalues = nblocks * block_size

mcreduction = pycuda.reduction.ReductionKernel(
    np.int32, neutral="0", reduce_expr="a + b", map_expr="x[i] * x[i] + y[i] * y[i] < 1 ? 1 : 0",
    arguments="const double *x, const double *y")

rng = curand.XORWOWRandomNumberGenerator()

mod = SourceModule("""
__global__ void gpucounts(double *a, double *b, int *counts)
{
  __shared__ bool belowfunc[128];

  unsigned int tid = threadIdx.x;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;

  belowfunc[tid] = ((a[i] * a[i] + b[i] * b[i]) < 1.);

  __syncthreads();

  if (tid == 0)
  {
    counts[blockIdx.x] = 0;
    for (unsigned int s = 0; s < blockDim.x; s++)
    {
      if (belowfunc[s])
        counts[blockIdx.x] += 1;
    }
  }
  __syncthreads();
}
""")

gpucounts = mod.get_function("gpucounts")

def get_pi_cuda():
    a_device = rng.gen_uniform((nvalues,), dtype=np.float64)
    b_device = rng.gen_uniform((nvalues,), dtype=np.float64)

    counts = np.empty(nblocks, dtype=np.int32)

    gpucounts(a_device, b_device, drv.Out(counts), grid=(nblocks, 1), block=(block_size, 1, 1))
    hitcount = np.sum(counts)
    return hitcount


def get_pi_cuda_redkern():
    a_device = rng.gen_uniform((nvalues,), dtype=np.float64)
    b_device = rng.gen_uniform((nvalues,), dtype=np.float64)

    # a_device = curand.rand(nvalues, dtype=np.float32)
    # b_device = curand.rand(nvalues, dtype=np.float32)

    hitcount = mcreduction(a_device, b_device).get()
    return hitcount


def get_pi_numpy():
    data = np.random.rand(nvalues, 2)
    hitcount = len(np.argwhere(np.linalg.norm(data, axis=1) < 1))
    return hitcount


start = drv.Event()
end = drv.Event()
n_iter = 10

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
    print(f'  pi_est {pi_est:.8f} time {timeseconds:8.4f} s ({n_iter * nvalues / timeseconds:.2e} evals/sec)')
