// LAVA DOES NOT PUT THE WORLD OUT. A lot of it, at the long law step a slow frame asks for, on the card and on the CPU:
//   1. Every field stays a number. (A NaN in T spreads to all that conducts, and a colour made of it is black.)
//   2. No voxel of matter is drawn black that was not black before.
//   3. Lava goes on being lava-coloured, and what it leaves is the stone it was.
#include "field/Playground.hpp"
#include "field/Terrain.hpp"
#include "field/VoxelMesh.hpp"
#include "field/WorldMesher.hpp"
#include "tests/Gate.hpp"
#include "tests/Pours.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

using namespace fieldengine::field;
using fieldengine::gate::check;

namespace {

std::size_t black_cells(const Playground& world)
{
    const FieldSet& f = world.fields();
    const GridShape& g = f.shape();
    const MatterView view = MatterView::of(f);
    const VoxelPalette palette;
    std::size_t black = 0;
    for (int z = 0; z < g.nz; ++z) {
        for (int y = 0; y < g.ny; ++y) {
            for (int x = 0; x < g.nx; ++x) {
                const VoxelCell cell = classify(view, x, y, z, palette);
                if (cell.kind != VoxelCell::Air && cell.r < 12 && cell.g < 12 && cell.b < 12) {
                    ++black;
                }
            }
        }
    }
    return black;
}

void not_numbers(const Playground& world, const std::string& when)
{
    const FieldSet& f = world.fields();
    const Law& law = world.law();
    for (std::size_t id = 0; id < f.count(); ++id) {
        std::size_t bad = 0;
        float low = 1e30f;
        float high = -1e30f;
        for (const float v : f.values(static_cast<int>(id))) {
            bad += std::isfinite(v) ? 0 : 1;
            low = std::min(low, v);
            high = std::max(high, v);
        }
        if (bad > 0) {
            std::cout << "  " << when << ": " << law.fields[id].name << " has " << bad << " cells that are not numbers" << std::endl;
        }
        check(bad == 0, "1: " + when + ": " + law.fields[id].name + " stays a number");
    }
}

void run_on(bool gpu, int field_every)
{
    const std::string name = std::string(gpu ? "the card" : "the CPU") + ", a step of " + std::to_string(field_every) + " ticks";
    std::vector<std::string> errors;
    std::vector<LawSource> laws;
    std::string spells;
    std::string path;
    if (!Playground::read_default_law(laws, spells, path, errors)) {
        check(false, "the law reads");
        return;
    }
    Playground::Config config;
    config.open_world = true;
    config.gpu_fields = gpu;
    config.field_every = field_every;
    config.window_chunks = gpu ? 9 : 3;
    config.window_layers = 3;
    config.far_life = false;
    std::optional<Playground> w = Playground::create(laws, spells, errors, config);
    if (!w) {
        check(false, "the world makes");
        return;
    }
    w->build_open_world();
    const std::size_t black_before = black_cells(*w);
    const auto spell = [&](const char* wanted) -> const Pour* {
        for (const Pour& s : w->pours()) {
            if (s.name == wanted) {
                return &s;
            }
        }
        return nullptr;
    };
    const Pour* magma = spell("magma");
    const Pour* fireball = spell("fireball");
    const Pour* rain = spell("downpour");
    check(magma != nullptr && fireball != nullptr && rain != nullptr, "the spells are there");
    const FieldSet& f = w->fields();
    const GridShape& g = f.shape();
    double most_lava = 0.0;
    std::size_t black_most = 0;
    const Vec3 me = w->player().position;
    const int rounds = gpu ? 60 : 20;
    for (int round = 0; round < rounds; ++round) {
        const double a = 0.7 * round;
        const Vec3 top{me.x + 4.0 * std::cos(a), me.y + 4.0 * std::sin(a), g.nz * g.cell - 0.1};
        const Vec3 at = w->pick(top, Vec3{0.0, 0.0, -1.0}).value_or(me);
        w->pour(*fireball, at, Vec3{1.0, 0.0, 0.0});
        w->pour(*magma, at + Vec3{0.0, 0.0, 0.5}, Vec3{1.0, 0.0, 0.0});
        if (round % 5 == 4) {
            w->pour(*rain, at + Vec3{0.0, 0.0, 2.0}, Vec3{1.0, 0.0, 0.0}); // and lava meets water
        }
        for (int tick = 0; tick < 30; ++tick) {
            w->player().hp = w->player().hp_max;
            w->step(Control{});
        }
        most_lava = std::max(most_lava, w->total("lava"));
    }
    not_numbers(*w, name);
    black_most = black_cells(*w);
    std::cout << "  " << name << ": lava up to " << most_lava << "; black voxels " << black_before << " -> " << black_most << "; T up to ";
    double hottest = 0.0;
    for (const float v : f.values(f.find("T"))) {
        hottest = std::max(hottest, static_cast<double>(v));
    }
    std::cout << hottest << std::endl;
    check(most_lava > 20.0, "3: " + name + ": there was a lot of lava");
    check(black_most <= black_before + 50, "2: " + name + ": nothing went black");
    check(w->violations().empty(), "1: " + name + ": no contract broke");
    for (const Violation& v : w->violations()) {
        std::cout << "  " << v.term << " " << v.field << " " << v.detail << std::endl;
    }
}

// And when you walk away, and it goes on without you, one chunk at a time.
void far_away()
{
    std::vector<std::string> errors;
    std::vector<LawSource> laws;
    std::string spells;
    std::string path;
    Playground::read_default_law(laws, spells, path, errors);
    Playground::Config config;
    config.open_world = true;
    config.gpu_fields = true;
    config.field_every = 4;
    config.window_chunks = 5;
    config.window_layers = 3;
    std::optional<Playground> w = Playground::create(laws, spells, errors, config);
    w->build_open_world();
    const Pour* fireball = nullptr;
    for (const Pour& s : w->pours()) {
        fireball = s.name == "fireball" ? &s : fireball;
    }
    const FieldSet& f = w->fields();
    const GridShape& g = f.shape();
    const Vec3 me = w->player().position;
    const int ccx = w->window_cx() + 2;
    const int ccy = w->window_cy() + 2;
    const int ccz = w->window_cz();
    for (int round = 0; round < 30; ++round) {
        const double a = 0.7 * round;
        const Vec3 top{me.x + 4.0 * std::cos(a), me.y + 4.0 * std::sin(a), g.nz * g.cell - 0.1};
        w->pour(*fireball, w->pick(top, Vec3{0.0, 0.0, -1.0}).value_or(me), Vec3{1.0, 0.0, 0.0});
        for (int tick = 0; tick < 20; ++tick) {
            w->player().hp = w->player().hp_max;
            w->step(Control{});
        }
    }
    const double lava = w->total("lava");
    const TerrainConfig& t = w->config().terrain;
    const double chunk_m = t.chunk * t.cell;
    const int fx = ccx + 30;
    const int fy = ccy;
    const int fz = (terrain_top(t, fx * t.chunk, fy * t.chunk) + 1) / t.chunk;
    w->center_window_on(fx, fy, fz);
    w->player().position = Vec3{(fx + 0.5) * chunk_m, (fy + 0.5) * chunk_m, (fz + 1.5) * chunk_m} - w->origin();
    w->player().velocity = Vec3{};
    std::size_t most_busy = 0;
    for (int tick = 0; tick < 60 * 30; ++tick) {
        w->player().hp = w->player().hp_max;
        w->step(Control{});
        most_busy = std::max(most_busy, w->far_busy());
    }
    std::size_t bad = 0;
    std::size_t black = 0;
    double far_lava = 0.0;
    const VoxelPalette palette;
    const int n = t.chunk;
    for (int cz = ccz - 1; cz < ccz + 4; ++cz) {
        for (int cy = ccy - 4; cy <= ccy + 4; ++cy) {
            for (int cx = ccx - 4; cx <= ccx + 4; ++cx) {
                ChunkData data;
                w->read_chunk(cx, cy, cz, data);
                for (const char* name : {"T", "solid", "lava", "wood", "leaf", "material", "alight", "held", "shaken"}) {
                    if (const std::vector<float>* v = data.find(name)) {
                        for (const float x : *v) {
                            bad += std::isfinite(x) ? 0 : 1;
                        }
                    }
                }
                if (const std::vector<float>* v = data.find("lava")) {
                    for (const float x : *v) {
                        far_lava += x;
                    }
                }
                const MatterView view = MatterView::of(data, n, n);
                for (int z = 0; z < n; ++z) {
                    for (int y = 0; y < n; ++y) {
                        for (int x = 0; x < n; ++x) {
                            const VoxelCell cell = classify(view, x, y, z, palette);
                            black += cell.kind != VoxelCell::Air && cell.r < 12 && cell.g < 12 && cell.b < 12 ? 1 : 0;
                        }
                    }
                }
            }
        }
    }
    std::cout << "  left behind: lava " << lava << " -> " << far_lava << ", " << most_busy << " chunks busy at most, " << bad << " not numbers, " << black << " black" << std::endl;
    // And as the mesher draws it, at every level, both where it was left and where you are.
    for (const auto& [name, rx0, ry0] : {std::tuple<const char*, int, int>{"there", ccx, ccy}, std::tuple<const char*, int, int>{"here", fx, fy}}) {
        for (int level = 0; level <= 5; ++level) {
            const LodLevel l = lod_level(level);
            ChunkMesh mesh;
            const int rx = rx0 - ((rx0 % l.span) + l.span) % l.span;
            const int ry = ry0 - ((ry0 % l.span) + l.span) % l.span;
            WorldMesher::build_region(*w, level, rx, ry, mesh);
            std::size_t dark = 0;
            for (const MeshVertex& v : mesh.vertices) {
                dark += (v.abgr & 0xffu) < 10u && ((v.abgr >> 8) & 0xffu) < 10u && ((v.abgr >> 16) & 0xffu) < 10u ? 1 : 0;
            }
            std::cout << "  " << name << " level " << level << ": " << mesh.vertices.size() << " vertices, " << dark << " black" << std::endl;
            check(dark * 20 <= mesh.vertices.size(), std::string("2: as drawn: ") + name);
        }
    }
    check(far_lava < lava * 3.0 + 200.0, "3: left behind, it makes no matter out of nothing");
    check(bad == 0, "1: left behind: everything stays a number");
    check(black <= 50, "2: left behind: nothing went black");
}

} // namespace

int main()
{
    run_on(true, 4);
    run_on(true, 15);
    run_on(false, 15);
    far_away();
    return fieldengine::gate::report("voxel_lava");
}
