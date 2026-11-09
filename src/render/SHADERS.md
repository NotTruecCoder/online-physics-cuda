# Shaders

bgfx-flavoured GLSL (`.sc`) lives in `assets/shaders/` and is compiled at build time into headers embedded in
the binary: `bgfx_compile_shaders(... AS_HEADERS)` in `CMakeLists.txt`, one blob per renderer profile, picked at
run time by `bgfx::createEmbeddedShader`. The Noop backend has no profile, so draws there are no-ops.

Programs: `basic` (coloured geometry), `tex` (sprites), `post` (the optional pixel-art pass), `imgui`.

To add one: put `vs_foo.sc` / `fs_foo.sc` there, add `cl_shader` lines to `CMakeLists.txt`, include the generated
headers and add `BGFX_EMBEDDED_SHADER` entries in the renderer.

The law's compute shaders are another matter: they are generated from the law text at run time
(`src/field/GpuSolver.cpp`).
