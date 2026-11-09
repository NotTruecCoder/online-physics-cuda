#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

// CUDA WITHOUT THE SDK. The driver (nvcuda.dll) comes with every NVIDIA driver and NVRTC (the run-time compiler) is a
// redistributable DLL, so the engine loads both at run time: it builds and starts on a machine with no CUDA toolkit,
// and on one without an NVIDIA card it says why and runs the law elsewhere. Only the few entry points used here are
// declared, with the ABI of the driver API (cuda.h, CUDA 12).
//
// NVRTC is looked for in FIELD_NVRTC (a DLL or a folder), %CUDA_PATH%\bin, next to the executable, the folder CMake
// found at configure time (FIELD_NVRTC_HINT), and PATH.
namespace fieldengine::field::cuda {

using CUresult = int;
using CUdevice = int;
using CUdeviceptr = unsigned long long;
struct CUctx_st;
struct CUmod_st;
struct CUfunc_st;
struct CUstream_st;
struct CUevent_st;
using CUcontext = CUctx_st*;
using CUmodule = CUmod_st*;
using CUfunction = CUfunc_st*;
using CUstream = CUstream_st*;
using CUevent = CUevent_st*;
struct nvrtcProgram_st;
using nvrtcProgram = nvrtcProgram_st*;

struct Api {
    CUresult (*cuInit)(unsigned) = nullptr;
    CUresult (*cuDeviceGetCount)(int*) = nullptr;
    CUresult (*cuDeviceGet)(CUdevice*, int) = nullptr;
    CUresult (*cuDeviceGetName)(char*, int, CUdevice) = nullptr;
    CUresult (*cuDeviceGetAttribute)(int*, int, CUdevice) = nullptr;
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
    CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
    CUresult (*cuCtxSynchronize)() = nullptr;
    CUresult (*cuModuleLoadDataEx)(CUmodule*, const void*, unsigned, int*, void**) = nullptr;
    CUresult (*cuModuleUnload)(CUmodule) = nullptr;
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult (*cuModuleGetGlobal)(CUdeviceptr*, std::size_t*, CUmodule, const char*) = nullptr;
    CUresult (*cuFuncSetAttribute)(CUfunction, int, int) = nullptr;
    CUresult (*cuMemAlloc)(CUdeviceptr*, std::size_t) = nullptr;
    CUresult (*cuMemFree)(CUdeviceptr) = nullptr;
    CUresult (*cuMemAllocHost)(void**, std::size_t) = nullptr;
    CUresult (*cuMemFreeHost)(void*) = nullptr;
    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr, const void*, std::size_t, CUstream) = nullptr;
    CUresult (*cuMemcpyDtoHAsync)(void*, CUdeviceptr, std::size_t, CUstream) = nullptr;
    CUresult (*cuMemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, std::size_t, CUstream) = nullptr;
    CUresult (*cuMemsetD32Async)(CUdeviceptr, unsigned, std::size_t, CUstream) = nullptr;
    CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream,
        void**, void**) = nullptr;
    CUresult (*cuStreamCreate)(CUstream*, unsigned) = nullptr;
    CUresult (*cuStreamSynchronize)(CUstream) = nullptr;
    CUresult (*cuEventCreate)(CUevent*, unsigned) = nullptr;
    CUresult (*cuEventRecord)(CUevent, CUstream) = nullptr;
    CUresult (*cuEventSynchronize)(CUevent) = nullptr;
    CUresult (*cuEventElapsedTime)(float*, CUevent, CUevent) = nullptr;
    CUresult (*cuGetErrorString)(CUresult, const char**) = nullptr;

    int (*nvrtcVersion)(int*, int*) = nullptr;
    int (*nvrtcCreateProgram)(nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*) = nullptr;
    int (*nvrtcCompileProgram)(nvrtcProgram, int, const char* const*) = nullptr;
    int (*nvrtcGetProgramLogSize)(nvrtcProgram, std::size_t*) = nullptr;
    int (*nvrtcGetProgramLog)(nvrtcProgram, char*) = nullptr;
    int (*nvrtcGetCUBINSize)(nvrtcProgram, std::size_t*) = nullptr;
    int (*nvrtcGetCUBIN)(nvrtcProgram, char*) = nullptr;
    int (*nvrtcDestroyProgram)(nvrtcProgram*) = nullptr;
    const char* (*nvrtcGetErrorString)(int) = nullptr;
};

// Attribute numbers of the driver API.
inline constexpr int kAttrComputeMajor = 75;
inline constexpr int kAttrComputeMinor = 76;
inline constexpr int kAttrMultiprocessors = 16;
inline constexpr int kFuncMaxDynamicShared = 8;

// The one CUDA device of the process: the first card, its primary context, one stream.
struct Device {
    Api api;
    bool ok = false;
    std::string why;   // when not ok
    std::string name;
    std::string nvrtc; // the DLL it compiles with
    int major = 0;
    int minor = 0;
    int sms = 0;
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUstream stream = nullptr;

    // Makes the context current on the calling thread (the law runs on a worker thread).
    void bind() const;
    [[nodiscard]] std::string error(CUresult r) const;
};

Device& device();

// Compiles CUDA C to a CUBIN for this card, or takes it from the disk cache (keyed by the text, the options and the
// card); empty with `log` set when it does not compile.
std::vector<char> compile(const std::string& source, const std::string& name, const std::vector<std::string>& options,
    std::string& log, bool& from_cache);

// A loaded module and its kernels by name.
class Module {
public:
    Module() = default;
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&& other) noexcept;
    Module& operator=(Module&& other) noexcept;
    ~Module();

    bool load(const std::vector<char>& cubin, std::string& why);
    CUfunction function(const char* name) const; // null if none
    CUdeviceptr global(const char* name, std::size_t* bytes = nullptr) const;
    [[nodiscard]] bool loaded() const { return module_ != nullptr; }

private:
    CUmodule module_ = nullptr;
};

// Device memory that frees itself.
class Buffer {
public:
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    ~Buffer();
    bool allocate(std::size_t bytes);
    [[nodiscard]] CUdeviceptr ptr() const { return ptr_; }
    [[nodiscard]] std::size_t bytes() const { return bytes_; }
    [[nodiscard]] CUdeviceptr at_float(std::size_t index) const { return ptr_ + index * 4; }

private:
    CUdeviceptr ptr_ = 0;
    std::size_t bytes_ = 0;
};

// Page-locked host memory: what copies to and from the card go through.
class Pinned {
public:
    Pinned() = default;
    Pinned(const Pinned&) = delete;
    Pinned& operator=(const Pinned&) = delete;
    ~Pinned();
    bool allocate(std::size_t bytes);
    [[nodiscard]] void* data() const { return data_; }
    [[nodiscard]] std::size_t bytes() const { return bytes_; }

private:
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
};

// Launches `f` over a grid of blocks with the given arguments (pointers to each argument's value).
CUresult launch(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shared,
    std::initializer_list<void*> args);

// Floats as IEEE half bits, rounded to nearest even (the host side of tensor-core weights).
std::uint16_t to_half(float v);

} // namespace fieldengine::field::cuda
