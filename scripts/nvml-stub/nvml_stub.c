// nvml_stub.c — LD_PRELOAD stub that hides NVIDIA GPUs from NVML.
//
// When preloaded, nvmlInit returns an error and nvmlDeviceGetCount returns 0,
// which prevents vLLM's CUDA platform plugin from activating on hybrid
// AMD+NVIDIA systems while leaving ROCm device probing untouched.
//
// NVML C API: https://docs.nvidia.com/deploy/nvml-api/

#define NVML_ERROR_UNINITIALIZED 1

int nvmlInit(void) {
    return NVML_ERROR_UNINITIALIZED;
}

int nvmlInit_v2(void) {
    return NVML_ERROR_UNINITIALIZED;
}

int nvmlInitWithFlags(unsigned int flags) {
    (void)flags;
    return NVML_ERROR_UNINITIALIZED;
}

int nvmlDeviceGetCount(unsigned int *deviceCount) {
    if (deviceCount)
        *deviceCount = 0;
    return NVML_ERROR_UNINITIALIZED;
}
