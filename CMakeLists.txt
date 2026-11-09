cmake_minimum_required(VERSION 3.25)

project(FieldEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/MP /utf-8)
endif()

# The engine, lawc, the benchmarks and the engine's gates need only a C++20 compiler and OpenMP. The sandbox (a window
# over the generated world) also needs SDL3, Dear ImGui, bgfx and glm from vcpkg: FIELD_BUILD_SANDBOX=OFF skips it.
option(FIELD_BUILD_SANDBOX "Build the interactive sandbox and the renderer (needs SDL3, imgui, bgfx, glm)" ON)

find_package(OpenMP REQUIRED)
if(FIELD_BUILD_SANDBOX)
    find_package(SDL3 CONFIG REQUIRED)
    find_package(imgui CONFIG REQUIRED)
    find_package(bgfx CONFIG REQUIRED)
    find_package(glm CONFIG REQUIRED)
    include("${bgfx_DIR}/bgfxToolUtils.cmake")
endif()

# --- The field engine -------------------------------------------------------------------------------
# The world as scalar fields on one voxel lattice, its laws compiled from text (assets/law), integrated on
# the CPU (the reference), as CUDA kernels generated from the law's IR and compiled by NVRTC at run time, or as HLSL
# compute shaders; networks inside the law (mlp, flux, ldiffuse) on tensor cores; bodies, rays and the picture reading
# one surface; a generated continent streamed around the viewer; creatures and people whose minds are lines of the same
# law, or a world model they learn. No window and no renderer, so it is tested on its own (src/field/FIELD.md).
file(GLOB_RECURSE FIELD_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/field/*.cpp" "${CMAKE_SOURCE_DIR}/src/field/*.hpp")
add_library(field_engine STATIC ${FIELD_SOURCES})
target_include_directories(field_engine PUBLIC src)

# CUDA needs no SDK to build: the driver and NVRTC are loaded at run time (src/field/cuda/CudaApi.hpp). Where NVRTC
# lives is looked up at run time too; a folder found here is one more place to look (CUDA_PATH, or the one a PyTorch
# install ships).
set(FIELD_NVRTC_HINT "" CACHE PATH "A folder with nvrtc64_*.dll; found when empty")
if(NOT FIELD_NVRTC_HINT)
    if(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}/bin")
        file(GLOB _nvrtc "$ENV{CUDA_PATH}/bin/nvrtc64_*.dll")
        if(_nvrtc)
            file(TO_CMAKE_PATH "$ENV{CUDA_PATH}/bin" _hint)
            set(FIELD_NVRTC_HINT "${_hint}" CACHE PATH "A folder with nvrtc64_*.dll; found when empty" FORCE)
        endif()
    endif()
endif()
if(NOT FIELD_NVRTC_HINT)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        execute_process(COMMAND "${Python3_EXECUTABLE}" -c
            "import importlib.util as u, os; s = u.find_spec('torch'); print(os.path.join(os.path.dirname(s.origin), 'lib') if s else '')"
            OUTPUT_VARIABLE _torch_lib OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        file(TO_CMAKE_PATH "${_torch_lib}" _torch_lib)
        if(_torch_lib)
            file(GLOB _nvrtc "${_torch_lib}/nvrtc64_*.dll")
            if(_nvrtc)
                set(FIELD_NVRTC_HINT "${_torch_lib}" CACHE PATH "A folder with nvrtc64_*.dll; found when empty" FORCE)
            endif()
        endif()
    endif()
endif()
message(STATUS "NVRTC hint: '${FIELD_NVRTC_HINT}' (run time also looks at FIELD_NVRTC, CUDA_PATH, next to the executable, PATH)")
target_compile_definitions(field_engine PRIVATE FIELD_NVRTC_HINT="${FIELD_NVRTC_HINT}")
# Cells are independent in every elementwise op and stencil: the solver cuts the lattice into fixed slabs
# and runs them on all cores; reductions sum fixed blocks in a fixed order, so the bits do not depend on
# the thread count.
target_link_libraries(field_engine PUBLIC OpenMP::OpenMP_CXX)
if(WIN32)
    # The law as compute shaders, generated from its text at run time (src/field/GpuSolver.hpp).
    target_link_libraries(field_engine PRIVATE d3d11 dxgi d3dcompiler)
endif()

if(FIELD_BUILD_SANDBOX)
# --- Render -----------------------------------------------------------------------------------------
# Camera math, units, post-process parameters: testable without a GPU.
add_library(field_render_core STATIC
    src/render/RenderTypes.hpp
    src/render/Units.hpp
    src/render/PostProcess.hpp
    src/render/Camera.cpp
    src/render/Camera.hpp
)
target_include_directories(field_render_core PUBLIC src)
target_link_libraries(field_render_core PUBLIC glm::glm)

# The bgfx backend: shaders compiled at build time into headers embedded in the binary (src/render/SHADERS.md).
set(SHADER_DIR "${CMAKE_SOURCE_DIR}/assets/shaders")
set(SHADER_OUT "${CMAKE_BINARY_DIR}/generated/shaders")
function(field_shader TYPE NAME VARYING OUT_VAR)
    bgfx_compile_shaders(
        TYPE ${TYPE}
        SHADERS "${SHADER_DIR}/${NAME}.sc"
        VARYING_DEF "${SHADER_DIR}/${VARYING}"
        OUTPUT_DIR "${SHADER_OUT}"
        OUT_FILES_VAR headers
        AS_HEADERS
    )
    set(${OUT_VAR} ${${OUT_VAR}} ${headers} PARENT_SCOPE)
endfunction()
set(SHADER_HEADERS "")
field_shader(VERTEX vs_basic varying.def.sc SHADER_HEADERS)
field_shader(FRAGMENT fs_basic varying.def.sc SHADER_HEADERS)
field_shader(VERTEX vs_imgui varying_imgui.def.sc SHADER_HEADERS)
field_shader(FRAGMENT fs_imgui varying_imgui.def.sc SHADER_HEADERS)
field_shader(VERTEX vs_post varying_post.def.sc SHADER_HEADERS)
field_shader(FRAGMENT fs_post varying_post.def.sc SHADER_HEADERS)

add_library(field_render STATIC
    src/render/BgfxRenderer.cpp
    src/render/BgfxRenderer.hpp
    src/render/ImGuiBgfx.cpp
    src/render/ImGuiBgfx.hpp
    ${SHADER_HEADERS}
)
target_include_directories(field_render PUBLIC src "${SHADER_OUT}")
target_link_libraries(field_render PUBLIC field_render_core bgfx::bgfx bgfx::bx bgfx::bimg bgfx::bimg_decode imgui::imgui glm::glm)

# --- The sandbox ------------------------------------------------------------------------------------
# A free camera over the generated world, and tools that act on it through the engine's public surface.
add_executable(sandbox
    src/sandbox/main.cpp
    src/sandbox/Sandbox.cpp
    src/sandbox/Sandbox.hpp
    src/sandbox/Draw.cpp
    src/sandbox/Panels.cpp
    src/sandbox/WorldView.cpp
    src/sandbox/WorldView.hpp
    src/sandbox/Effects.cpp
    src/sandbox/Effects.hpp
)
target_link_libraries(sandbox PRIVATE SDL3::SDL3 imgui::imgui field_engine field_render)
endif()

# The runtime data (laws, worlds, models) next to the binaries, so the build directory runs as it is.
add_custom_target(copy_assets ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/assets" "${CMAKE_BINARY_DIR}/assets"
    COMMENT "Copying runtime assets"
)
if(TARGET sandbox)
    add_dependencies(sandbox copy_assets)
endif()

# --- lawc -------------------------------------------------------------------------------------------
# The law compiler on the command line: the IR (text or JSON for tools/lawjax), the generated CUDA, steps of a scene on
# any backend with their profile, and the states they reach (for the JAX parity and the trainers).
add_executable(lawc tools/lawc/main.cpp)
target_link_libraries(lawc PRIVATE field_engine)
add_dependencies(lawc copy_assets)

# --- Benchmarks -------------------------------------------------------------------------------------
# bench/README.md: what each measures. The Python drivers (bench/*.py) call these and lawc.
add_executable(bench_mind bench/mind_bench.cpp)
target_link_libraries(bench_mind PRIVATE field_engine)
add_dependencies(bench_mind copy_assets)
add_executable(bench_stencil bench/stencil_bench.cpp)
target_link_libraries(bench_stencil PRIVATE field_engine)
add_dependencies(bench_stencil copy_assets)

# --- Gates ------------------------------------------------------------------------------------------
# Every src/tests/*_smoke.cpp is a gate: a must-pass check of one claim the engine makes, each with a control
# that shows the check can fail. A file is all it takes.
include(CTest)
file(GLOB GATES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/tests/*_smoke.cpp")
foreach(gate_path IN LISTS GATES)
    get_filename_component(gate_file "${gate_path}" NAME_WE)
    string(REGEX REPLACE "_smoke$" "" gate_name "${gate_file}")
    if(NOT FIELD_BUILD_SANDBOX AND gate_name MATCHES "^render_")
        continue()
    endif()
    add_executable(gate_${gate_name} "${gate_path}")
    target_link_libraries(gate_${gate_name} PRIVATE field_engine)
    if(gate_name STREQUAL "render_core")
        target_link_libraries(gate_${gate_name} PRIVATE field_render_core)
    elseif(gate_name STREQUAL "render_gpu")
        target_link_libraries(gate_${gate_name} PRIVATE field_render)
    endif()
    add_dependencies(gate_${gate_name} copy_assets)
    if(BUILD_TESTING)
        add_test(NAME ${gate_name} COMMAND gate_${gate_name})
        set_tests_properties(${gate_name} PROPERTIES WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
    endif()
endforeach()
