#include "field/cuda/CudaApi.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifndef FIELD_NVRTC_HINT
#define FIELD_NVRTC_HINT ""
#endif

namespace fieldengine::field::cuda {

namespace {

#ifdef _WIN32
using Library = HMODULE;
Library open_library(const std::string& path)
{
    return LoadLibraryExA(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
}
void* symbol(Library lib, const char* name)
{
    return reinterpret_cast<void*>(GetProcAddress(lib, name));
}
std::string exe_dir()
{
    char path[MAX_PATH + 1]{};
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    std::string p = path;
    const std::size_t slash = p.find_last_of("\\/");
    return slash == std::string::npos ? "." : p.substr(0, slash);
}
// nvrtc64_<version>_0.dll in a folder; the newest wins (not the .alt build).
std::string find_nvrtc_in(const std::string& dir)
{
    WIN32_FIND_DATAA found{};
    const HANDLE h = FindFirstFileA((dir + "\\nvrtc64_*.dll").c_str(), &found);
    if (h == INVALID_HANDLE_VALUE) {
        return {};
    }
    std::string best;
    do {
        const std::string name = found.cFileName;
        if (name.find(".alt.") == std::string::npos && name > best) {
            best = name;
        }
    } while (FindNextFileA(h, &found));
    FindClose(h);
    return best.empty() ? std::string{} : dir + "\\" + best;
}
std::string temp_dir()
{
    char temp[MAX_PATH + 1]{};
    GetTempPathA(MAX_PATH, temp);
    std::string dir = std::string(temp) + "field_cuda_cache";
    CreateDirectoryA(dir.c_str(), nullptr);
    return dir + "\\";
}
const char* kDriver = "nvcuda.dll";
#else
using Library = void*;
Library open_library(const std::string& path)
{
    return dlopen(path.c_str(), RTLD_NOW);
}
void* symbol(Library lib, const char* name)
{
    return dlsym(lib, name);
}
std::string exe_dir()
{
    return ".";
}
std::string find_nvrtc_in(const std::string& dir)
{
    const std::string p = dir + "/libnvrtc.so";
    std::ifstream in(p);
    return in ? p : std::string{};
}
std::string temp_dir()
{
    return "/tmp/";
}
const char* kDriver = "libcuda.so.1";
#endif

template <class Fn>
bool bind_symbol(Library lib, Fn& fn, const char* name, std::string& missing)
{
    fn = reinterpret_cast<Fn>(symbol(lib, name));
    if (fn == nullptr && missing.empty()) {
        missing = name;
    }
    return fn != nullptr;
}

std::string find_nvrtc()
{
    std::vector<std::string> dirs;
    if (const char* forced = std::getenv("FIELD_NVRTC")) {
        const std::string f = forced;
        if (f.size() > 4 && (f.substr(f.size() - 4) == ".dll" || f.find(".so") != std::string::npos)) {
            return f;
        }
        dirs.push_back(f);
    }
    if (const char* cuda = std::getenv("CUDA_PATH")) {
        dirs.push_back(std::string(cuda) + "/bin");
    }
    dirs.push_back(exe_dir());
    if (std::strlen(FIELD_NVRTC_HINT) > 0) {
        dirs.push_back(FIELD_NVRTC_HINT);
    }
    for (const std::string& d : dirs) {
        if (std::string p = find_nvrtc_in(d); !p.empty()) {
            return p;
        }
    }
    // Last, whatever the loader finds on PATH.
#ifdef _WIN32
    for (const char* name : {"nvrtc64_120_0.dll", "nvrtc64_130_0.dll", "nvrtc64_112_0.dll"}) {
        if (HMODULE h = LoadLibraryA(name)) {
            FreeLibrary(h);
            return name;
        }
    }
#endif
    return {};
}

std::uint64_t fnv(const std::string& text, std::uint64_t h = 1469598103934665603ull)
{
    for (const char c : text) {
        h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ull;
    }
    return h;
}

} // namespace

void Device::bind() const
{
    if (ok) {
        api.cuCtxSetCurrent(context);
    }
}

std::string Device::error(CUresult r) const
{
    const char* text = nullptr;
    if (api.cuGetErrorString != nullptr) {
        api.cuGetErrorString(r, &text);
    }
    return text != nullptr ? std::string(text) : "CUDA error " + std::to_string(r);
}

Device& device()
{
    static Device d;
    static std::once_flag once;
    std::call_once(once, [] {
        Api& a = d.api;
        const Library driver = open_library(kDriver);
        if (driver == nullptr) {
            d.why = "no NVIDIA driver (nvcuda.dll)";
            return;
        }
        std::string missing;
        bind_symbol(driver, a.cuInit, "cuInit", missing);
        bind_symbol(driver, a.cuDeviceGetCount, "cuDeviceGetCount", missing);
        bind_symbol(driver, a.cuDeviceGet, "cuDeviceGet", missing);
        bind_symbol(driver, a.cuDeviceGetName, "cuDeviceGetName", missing);
        bind_symbol(driver, a.cuDeviceGetAttribute, "cuDeviceGetAttribute", missing);
        bind_symbol(driver, a.cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain", missing);
        bind_symbol(driver, a.cuCtxSetCurrent, "cuCtxSetCurrent", missing);
        bind_symbol(driver, a.cuCtxSynchronize, "cuCtxSynchronize", missing);
        bind_symbol(driver, a.cuModuleLoadDataEx, "cuModuleLoadDataEx", missing);
        bind_symbol(driver, a.cuModuleUnload, "cuModuleUnload", missing);
        bind_symbol(driver, a.cuModuleGetFunction, "cuModuleGetFunction", missing);
        bind_symbol(driver, a.cuModuleGetGlobal, "cuModuleGetGlobal_v2", missing);
        bind_symbol(driver, a.cuFuncSetAttribute, "cuFuncSetAttribute", missing);
        bind_symbol(driver, a.cuMemAlloc, "cuMemAlloc_v2", missing);
        bind_symbol(driver, a.cuMemFree, "cuMemFree_v2", missing);
        bind_symbol(driver, a.cuMemAllocHost, "cuMemAllocHost_v2", missing);
        bind_symbol(driver, a.cuMemFreeHost, "cuMemFreeHost", missing);
        bind_symbol(driver, a.cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2", missing);
        bind_symbol(driver, a.cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2", missing);
        bind_symbol(driver, a.cuMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2", missing);
        bind_symbol(driver, a.cuMemsetD32Async, "cuMemsetD32Async", missing);
        bind_symbol(driver, a.cuLaunchKernel, "cuLaunchKernel", missing);
        bind_symbol(driver, a.cuStreamCreate, "cuStreamCreate", missing);
        bind_symbol(driver, a.cuStreamSynchronize, "cuStreamSynchronize", missing);
        bind_symbol(driver, a.cuEventCreate, "cuEventCreate", missing);
        bind_symbol(driver, a.cuEventRecord, "cuEventRecord", missing);
        bind_symbol(driver, a.cuEventSynchronize, "cuEventSynchronize", missing);
        bind_symbol(driver, a.cuEventElapsedTime, "cuEventElapsedTime", missing);
        bind_symbol(driver, a.cuGetErrorString, "cuGetErrorString", missing);
        if (!missing.empty()) {
            d.why = std::string("the NVIDIA driver has no ") + missing;
            return;
        }
        int count = 0;
        if (a.cuInit(0) != 0 || a.cuDeviceGetCount(&count) != 0 || count < 1) {
            d.why = "no CUDA device";
            return;
        }
        a.cuDeviceGet(&d.device, 0);
        char name[256]{};
        a.cuDeviceGetName(name, 255, d.device);
        d.name = name;
        a.cuDeviceGetAttribute(&d.major, kAttrComputeMajor, d.device);
        a.cuDeviceGetAttribute(&d.minor, kAttrComputeMinor, d.device);
        a.cuDeviceGetAttribute(&d.sms, kAttrMultiprocessors, d.device);
        if (d.major < 7) {
            d.why = d.name + " is older than Volta (sm_70)";
            return;
        }

        d.nvrtc = find_nvrtc();
#ifdef _WIN32
        // NVRTC loads its builtins by name when it compiles: loaded first from its folder, that name is already taken.
        if (const std::size_t slash = d.nvrtc.find_last_of("\\/"); slash != std::string::npos) {
            const std::string dir = d.nvrtc.substr(0, slash);
            WIN32_FIND_DATAA found{};
            const HANDLE h = FindFirstFileA((dir + "\\nvrtc-builtins64_*.dll").c_str(), &found);
            if (h != INVALID_HANDLE_VALUE) {
                do {
                    open_library(dir + "\\" + found.cFileName);
                } while (FindNextFileA(h, &found));
                FindClose(h);
            }
        }
#endif
        const Library rtc = d.nvrtc.empty() ? nullptr : open_library(d.nvrtc);
        if (rtc == nullptr) {
            d.why = "no NVRTC (nvrtc64_*.dll): set FIELD_NVRTC, CUDA_PATH, or put it next to the executable";
            return;
        }
        bind_symbol(rtc, a.nvrtcVersion, "nvrtcVersion", missing);
        bind_symbol(rtc, a.nvrtcCreateProgram, "nvrtcCreateProgram", missing);
        bind_symbol(rtc, a.nvrtcCompileProgram, "nvrtcCompileProgram", missing);
        bind_symbol(rtc, a.nvrtcGetProgramLogSize, "nvrtcGetProgramLogSize", missing);
        bind_symbol(rtc, a.nvrtcGetProgramLog, "nvrtcGetProgramLog", missing);
        bind_symbol(rtc, a.nvrtcGetCUBINSize, "nvrtcGetCUBINSize", missing);
        bind_symbol(rtc, a.nvrtcGetCUBIN, "nvrtcGetCUBIN", missing);
        bind_symbol(rtc, a.nvrtcDestroyProgram, "nvrtcDestroyProgram", missing);
        bind_symbol(rtc, a.nvrtcGetErrorString, "nvrtcGetErrorString", missing);
        if (!missing.empty()) {
            d.why = d.nvrtc + " has no " + missing;
            return;
        }
        if (a.cuDevicePrimaryCtxRetain(&d.context, d.device) != 0 || a.cuCtxSetCurrent(d.context) != 0 ||
            a.cuStreamCreate(&d.stream, 1 /* non-blocking */) != 0) {
            d.why = "could not open a context on " + d.name;
            return;
        }
        d.ok = true;
    });
    return d;
}

std::vector<char> compile(const std::string& source, const std::string& name, const std::vector<std::string>& options,
    std::string& log, bool& from_cache)
{
    Device& d = device();
    from_cache = false;
    if (!d.ok) {
        log = d.why;
        return {};
    }
    std::vector<std::string> opts = options;
    opts.push_back("--gpu-architecture=sm_" + std::to_string(d.major) + std::to_string(d.minor));
    std::string key = source;
    for (const std::string& o : opts) {
        key += "\n//" + o;
    }
    key += "\n//" + d.name + "|" + d.nvrtc;
    std::ostringstream path;
    path << temp_dir() << std::hex << fnv(key) << ".cubin";
    if (std::ifstream in(path.str(), std::ios::binary | std::ios::ate); in) {
        const auto size = static_cast<std::size_t>(in.tellg());
        std::vector<char> cubin(size);
        in.seekg(0);
        if (size > 0 && in.read(cubin.data(), static_cast<std::streamsize>(size))) {
            from_cache = true;
            return cubin;
        }
    }
    const Api& a = d.api;
    nvrtcProgram program = nullptr;
    if (a.nvrtcCreateProgram(&program, source.c_str(), name.c_str(), 0, nullptr, nullptr) != 0) {
        log = "nvrtcCreateProgram failed";
        return {};
    }
    std::vector<const char*> raw;
    for (const std::string& o : opts) {
        raw.push_back(o.c_str());
    }
    const int result = a.nvrtcCompileProgram(program, static_cast<int>(raw.size()), raw.data());
    std::size_t log_size = 0;
    a.nvrtcGetProgramLogSize(program, &log_size);
    if (log_size > 1) {
        log.assign(log_size, '\0');
        a.nvrtcGetProgramLog(program, log.data());
    }
    std::vector<char> cubin;
    if (result == 0) {
        std::size_t size = 0;
        a.nvrtcGetCUBINSize(program, &size);
        cubin.resize(size);
        a.nvrtcGetCUBIN(program, cubin.data());
        std::ofstream(path.str(), std::ios::binary).write(cubin.data(), static_cast<std::streamsize>(cubin.size()));
    } else if (log.empty()) {
        log = a.nvrtcGetErrorString(result);
    }
    a.nvrtcDestroyProgram(&program);
    return cubin;
}

Module::Module(Module&& other) noexcept : module_(other.module_)
{
    other.module_ = nullptr;
}

Module& Module::operator=(Module&& other) noexcept
{
    if (this != &other) {
        if (module_ != nullptr) {
            device().api.cuModuleUnload(module_);
        }
        module_ = other.module_;
        other.module_ = nullptr;
    }
    return *this;
}

Module::~Module()
{
    if (module_ != nullptr && device().ok) {
        device().api.cuModuleUnload(module_);
    }
}

bool Module::load(const std::vector<char>& cubin, std::string& why)
{
    const Device& d = device();
    d.bind();
    const CUresult r = d.api.cuModuleLoadDataEx(&module_, cubin.data(), 0, nullptr, nullptr);
    if (r != 0) {
        why = "the module did not load: " + d.error(r);
        module_ = nullptr;
        return false;
    }
    return true;
}

CUfunction Module::function(const char* name) const
{
    CUfunction f = nullptr;
    device().api.cuModuleGetFunction(&f, module_, name);
    return f;
}

CUdeviceptr Module::global(const char* name, std::size_t* bytes) const
{
    CUdeviceptr p = 0;
    std::size_t size = 0;
    device().api.cuModuleGetGlobal(&p, &size, module_, name);
    if (bytes != nullptr) {
        *bytes = size;
    }
    return p;
}

Buffer::Buffer(Buffer&& other) noexcept : ptr_(other.ptr_), bytes_(other.bytes_)
{
    other.ptr_ = 0;
    other.bytes_ = 0;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept
{
    if (this != &other) {
        if (ptr_ != 0) {
            device().api.cuMemFree(ptr_);
        }
        ptr_ = other.ptr_;
        bytes_ = other.bytes_;
        other.ptr_ = 0;
        other.bytes_ = 0;
    }
    return *this;
}

Buffer::~Buffer()
{
    if (ptr_ != 0 && device().ok) {
        device().api.cuMemFree(ptr_);
    }
}

bool Buffer::allocate(std::size_t bytes)
{
    if (ptr_ != 0) {
        device().api.cuMemFree(ptr_);
        ptr_ = 0;
    }
    bytes_ = std::max<std::size_t>(bytes, 4);
    return device().api.cuMemAlloc(&ptr_, bytes_) == 0;
}

Pinned::~Pinned()
{
    if (data_ != nullptr && device().ok) {
        device().api.cuMemFreeHost(data_);
    }
}

bool Pinned::allocate(std::size_t bytes)
{
    if (data_ != nullptr) {
        device().api.cuMemFreeHost(data_);
        data_ = nullptr;
    }
    bytes_ = std::max<std::size_t>(bytes, 4);
    return device().api.cuMemAllocHost(&data_, bytes_) == 0;
}

CUresult launch(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shared,
    std::initializer_list<void*> args)
{
    const Device& d = device();
    std::vector<void*> params(args.begin(), args.end());
    return d.api.cuLaunchKernel(f, std::max(gx, 1u), std::max(gy, 1u), std::max(gz, 1u), bx, by, bz, shared, d.stream,
        params.empty() ? nullptr : params.data(), nullptr);
}

std::uint16_t to_half(float v)
{
    std::uint32_t x = 0;
    std::memcpy(&x, &v, 4);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    const int exponent = static_cast<int>((x >> 23) & 0xffu) - 127 + 15;
    std::uint32_t mantissa = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xffu) { // inf, nan
        return static_cast<std::uint16_t>(sign | 0x7c00u | (mantissa != 0 ? 0x200u : 0u));
    }
    if (exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        mantissa |= 0x800000u;
        const int shift = 14 - exponent;
        std::uint32_t half = mantissa >> shift;
        const std::uint32_t rest = mantissa & ((1u << shift) - 1u);
        const std::uint32_t mid = 1u << (shift - 1);
        if (rest > mid || (rest == mid && (half & 1u))) {
            ++half;
        }
        return static_cast<std::uint16_t>(sign | half);
    }
    std::uint32_t half = (static_cast<std::uint32_t>(exponent) << 10) | (mantissa >> 13);
    const std::uint32_t rest = mantissa & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (half & 1u))) {
        ++half; // may carry into the exponent: that is the right rounding
    }
    return static_cast<std::uint16_t>(sign | half);
}

} // namespace fieldengine::field::cuda
