// Gate: A LEARNED SCHEME RUNS UNDER GUARD. heat.law's learned conduction takes twice the classical scheme's stable step
// in one pass. Untrained it is the explicit scheme at that step, which blows up: every step that breaks T's range is
// undone and taken classically, so the world is what the classical law makes of it, and the profile says how often.
// Off, nothing learned runs. Same on the CPU and in CUDA.
//
// Falsified by (each run once): the snapshot not restored on a fallback (T leaves its range); fallbacks not counted
// (the profile check fails); learned max_dt ignored (the untrained scheme takes substeps and never breaks a contract).
#include "tests/Gate.hpp"

#include "field/Law.hpp"
#include "field/Scene.hpp"
#include "field/Solver.hpp"
#include "field/cuda/CudaSolver.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>

using namespace fieldengine::field;
using fieldengine::gate::check;

int main()
{
    std::optional<LawSource> source = read_law_file("assets/law/bench/heat.law");
    check(source.has_value(), "heat.law is there");
    if (!source) {
        return fieldengine::gate::report("guarded");
    }
    // Untrained on purpose: the net file is not read.
    std::string text = source->text;
    const std::size_t at = text.find("act tanh");
    text.insert(at + 8, " file none");
    CompileResult compiled = compile_law({LawSource{"heat", text}});
    check(compiled.law.has_value(), "heat.law compiles");
    Scene scene;
    std::string error;
    check(load_scene("assets/scenes/heat.scene", scene, error), "the heat scene loads");
    const double dt = 0.3; // twice the classical scheme's stable step for k = 1, h = 1

    for (const char* backend : {"cpu", "cuda"}) {
        if (std::string(backend) == "cuda" && !CudaSolver::unavailable().empty()) {
            continue;
        }
#ifdef _WIN32
        _putenv_s("FIELD_BACKEND", backend);
#else
        setenv("FIELD_BACKEND", backend, 1);
#endif
        Solver learned(*compiled.law, scene.grid);
        Solver classical(*compiled.law, scene.grid);
        apply_scene(scene, learned, error);
        apply_scene(scene, classical, error);
        if (std::string(backend) == "cuda") {
            std::string why;
            learned.use_gpu(why);
            classical.use_gpu(why);
        }
        learned.set_learned(true);
        int fallbacks = 0;
        int held = 0;
        for (int s = 0; s < 40; ++s) {
            learned.step(dt);
            classical.step(dt);
            fallbacks += learned.profile().fallbacks;
            held += learned.profile().terms[0].learned ? 1 : 0;
            check(learned.violations().empty(), std::string(backend) + ": no broken contract reaches the world");
        }
        double worst = 0.0;
        float least = 1e30f;
        float most = -1e30f;
        for (std::size_t i = 0; i < learned.fields().values(0).size(); ++i) {
            const float v = std::as_const(learned).fields().values(0)[i];
            least = std::min(least, v);
            most = std::max(most, v);
            worst = std::max(worst, static_cast<double>(std::abs(v - std::as_const(classical).fields().values(0)[i])));
        }
        std::cout << "  " << backend << ": untrained learned scheme at 2x the step: " << fallbacks << " of 40 steps undone, " << held
                  << " held; T in [" << least << ", " << most << "], worst off the classical run " << worst << " K\n";
        check(fallbacks > 0, std::string(backend) + ": the guard catches the unstable step");
        check(least >= 0.0f && most <= 6000.0f, std::string(backend) + ": T stays in its range");
        check(worst < 400.0, std::string(backend) + ": the world stays the classical one, near enough");
        Solver off(*compiled.law, scene.grid);
        apply_scene(scene, off, error);
        off.step(dt);
        check(off.profile().fallbacks == 0 && !off.profile().terms[0].learned, std::string(backend) + ": off, nothing learned runs");
    }
    return fieldengine::gate::report("guarded");
}
