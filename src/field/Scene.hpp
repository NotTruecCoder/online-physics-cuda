#pragma once

#include "field/Grid.hpp"
#include "field/Solver.hpp"

#include <string>
#include <vector>

// A STARTING STATE AS TEXT, for benchmarks, parity gates and the trainers (lawc --scene, tools/lawjax):
//
//   grid 96 96 1 0.5            # cells and metres per cell
//   law assets/law/bench/front.law
//   fill T 290                  # every cell
//   box fuel 1 0 0 0 96 96 1    # cells [x0, x1) x [y0, y1) x [z0, z1)
//   ball T 1200 10 48 0 4       # a ball of radius r cells around (x, y, z)
//   noise fuel 0.3 7            # adds a hash of the cell in [0, amp) (no RNG state: the same text, the same start)
//   param k_heat 0.5
namespace fieldengine::field {

struct Scene {
    GridShape grid{64, 64, 1, 1.0};
    std::vector<std::string> laws;
    struct Op {
        std::string kind; // fill, box, ball, noise
        std::string field;
        double value = 0.0;
        double args[6] = {0, 0, 0, 0, 0, 0};
    };
    std::vector<Op> ops;
    std::vector<std::pair<std::string, double>> params;
};

bool load_scene(const std::string& path, Scene& scene, std::string& error);
// Writes the scene's fields into the solver (by name; unknown names are an error) and its params.
bool apply_scene(const Scene& scene, Solver& solver, std::string& error);

// Raw float32 per field, linear order (x fastest), plus a JSON line of the grid: what tools/lawjax reads.
bool dump_fields(const FieldSet& fields, const std::string& dir, std::string& error);
// The other way: every DIR/FIELD.bin whose field the law has, of the lattice's size, replaces that field. Fields the
// directory lacks keep their values; a file of the wrong size is an error.
bool load_fields(FieldSet& fields, const std::string& dir, std::string& error);

} // namespace fieldengine::field
