// Device-link anchor for test executables.
//
// bsmp builds its CUDA sources with CUDA_SEPARABLE_COMPILATION ON (-rdc=true),
// so any executable that pulls kernel objects out of the bsmp static library
// needs a device-link step. CMake only runs the device linker for targets that
// have at least one .cu source; adding this empty translation unit to a test
// executable forces that step and resolves the __cudaRegisterLinkedBinary
// symbols without touching the (CUDA-free) test logic itself.