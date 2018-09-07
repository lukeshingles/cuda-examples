# cuda-examples
NVIDIA CUDA examples in C and Python (pycuda)

## Example output from Monte Carlo estimator for Pi
```
CUDA Kernel after 10 iterations:
  pi_est 3.14150703 time   1.2310 s (1.04e+08 evals/sec)
CUDA ReductionKernel after 10 iterations:
  pi_est 3.14161831 time   0.9589 s (1.33e+08 evals/sec)
numpy after 10 iterations:
  pi_est 3.14141781 time  11.5646 s (1.11e+07 evals/sec)
```