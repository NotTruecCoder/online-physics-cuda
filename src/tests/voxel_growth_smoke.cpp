// GATE: WHAT BURNS, WHAT GROWS, WHAT HOLDS UP. Claims about the world, all of them from text (the law, trees.world):
//   1. Fire is too hot to spread: it burns off what it touches and leaves nothing alight. An ember starts a fire, which
//      then walks by itself, over turf and up trees, as far as there is something to burn, and stops at bare ground.
//   2. What grows follows its formulas: as tall as it should be, as thick as the Greenhill law says (taller is thicker, and
//      a slender kind is thinner), the wood all joined to the foot, nothing floating. The same seed grows the same tree.
//   3. Each land has its own trees, big and small, and things growing on the ground; forests have them thick, deserts not.
//   4. The land: peaks high over the sea, rifts cut deep into it.
//   5. A place of a house or two is a family's own: no streets, no road. Nothing called a village is that small.
//   6. What nothing holds up comes down: a trunk cut through drops its crown; a tree left alone stands. A slab in the air
//      comes down, a bridge joined at one end does not. Stone and wood fall as rubble, not one voxel made or lost.
//   7. How a body moves is read from the law's text.
//
// Falsify by: dropping `step(alight - alight_min)` from grass_burns (1), the scorch term (1), alight_gain 0 (1), the exponent
// in the thickness formula (2), the rule that wood stays joined (2), the species' biomes (3), the massif and rift terms (4),
// `s.lone = true` (5), or `coming_down` (6).

#include "field/Playground.hpp"
#include "field/WorldMap.hpp"
#include "tests/Gate.hpp"
#include "tests/Pours.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <tuple>

using namespace fieldengine::field;
using fieldengine::gate::check;

namespace {

void run(Playground& g, int ticks)
{
    for (int i = 0; i < ticks; ++i) {
        g.step(Control{});
    }
}

const Pour& spell_of(const Playground& g, const std::string& name)
{
    const int slot = g.pour_slot(name);
    check(slot >= 0, "the book has " + name);
    return g.pours()[static_cast<std::size_t>(std::max(slot, 0))];
}

// The proving ground with fields written by hand: a floor is already there.
void paint(Playground& g, const char* field, int x0, int x1, int y0, int y1, int z0, int z1, float value)
{
    const GridShape& s = g.fields().shape();
    const int id = g.fields().find(field);
    for (int z = z0; z <= z1; ++z) {
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                g.solver().edit_cell(id, s.index(x, y, z)) = value;
            }
        }
    }
    g.solver().refresh_derived();
}

double sum_over(const Playground& g, const char* field, int x0, int x1, int y0, int y1, int z0, int z1)
{
    const GridShape& s = g.fields().shape();
    const std::vector<float>& v = g.fields().values(g.fields().find(field));
    double total = 0.0;
    for (int z = z0; z <= z1; ++z) {
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                total += v[s.index(x, y, z)];
            }
        }
    }
    return total;
}

Playground arena()
{
    std::vector<std::string> errors;
    std::optional<Playground> g = Playground::load_default(errors);
    for (const std::string& e : errors) {
        std::cerr << "  " << e << "\n";
    }
    if (!g) {
        check(false, "the proving ground loads");
        std::exit(fieldengine::gate::report("voxel_growth"));
    }
    return std::move(*g);
}

} // namespace

int main()
{
    // ---- 1. fire and ember ---------------------------------------------------------------------------------
    {
        // A meadow of turf over a stone floor (cells z = 0..1): lying in the air cell over it, as turf does.
        const auto meadow = [] {
            Playground g = arena();
            run(g, 5);
            paint(g, "material", 0, 47, 0, 11, 1, 1, 4.0f); // soil: turf lives on nothing else
            paint(g, "grass", 0, 47, 0, 11, 2, 2, 1.0f);
            paint(g, "wood", 40, 40, 5, 5, 2, 12, 1.0f); // and a tree
            paint(g, "leaf", 38, 42, 3, 7, 10, 13, 1.0f);
            paint(g, "wood", 40, 40, 5, 5, 2, 12, 1.0f);
            return g;
        };
        const Vec3 at{4.0, 3.0, 1.2};
        Playground hot = meadow();
        const double turf0 = hot.total("grass");
        hot.pour(spell_of(hot, "fire"), at, Vec3{1.0, 0.0, 0.0});
        run(hot, 60);
        const double burnt_at_once = turf0 - hot.total("grass");
        run(hot, 60 * 25);
        const double burnt_hot = turf0 - hot.total("grass");
        std::cout << "  fire on turf: " << burnt_at_once << " cells burnt off in a second, " << burnt_hot << " after 26 s, of " << turf0 << "\n";
        check(burnt_at_once > 1.5, "1: fire burns off what it touches");
        check(burnt_hot < turf0 * 0.1, "1: and does not start a fire: the meadow stands");
        check(hot.total("alight") < 0.01, "1: nothing is left alight");

        Playground lit = meadow();
        const double wood0 = lit.total("wood");
        const double leaves0 = lit.total("leaf");
        lit.pour(spell_of(lit, "kindle"), at, Vec3{1.0, 0.0, 0.0});
        run(lit, 60 * 30);
        const double burnt_lit = turf0 - lit.total("grass");
        std::cout << "  an ember on turf: " << burnt_lit << " cells burnt in 30 s; wood " << wood0 << " -> " << lit.total("wood") << ", leaves " << leaves0 << " -> " << lit.total("leaf") << "\n";
        check(burnt_lit > turf0 * 0.5, "1: an ember starts a fire that walks through the turf");
        check(burnt_lit > burnt_hot * 3.0, "1: much further than a far hotter fire goes");
        run(lit, 60 * 60); // a fire goes up a trunk slowly, as wood conducts
        std::cout << "  a minute on: wood " << lit.total("wood") << ", leaves " << lit.total("leaf") << "\n";
        check(lit.total("wood") < wood0 * 0.8, "1: it climbs a tree that stands in it");
        check(lit.total("leaf") < leaves0 * 0.8, "1: and takes its leaves");

        // Fire, poured right on a trunk, eats it where it stands: the rest of the tree does not catch.
        Playground scorched = meadow();
        scorched.pour(spell_of(scorched, "fire"), Vec3{20.25, 2.75, 3.0}, Vec3{1.0, 0.0, 0.0});
        run(scorched, 240);
        check(scorched.total("alight") < 0.01, "1: fire on wood does not leave it alight");

        // Bare ground stops it: a gap, and turf beyond that does not burn.
        Playground gap = arena();
        run(gap, 5);
        paint(gap, "material", 0, 47, 0, 11, 1, 1, 4.0f);
        paint(gap, "solid", 21, 29, 0, 11, 1, 1, 0.0f); // bare ground: nothing there to hold turf
        paint(gap, "grass", 0, 20, 0, 11, 2, 2, 1.0f);
        paint(gap, "grass", 30, 47, 0, 11, 2, 2, 1.0f);
        const double beyond = sum_over(gap, "grass", 30, 47, 0, 11, 2, 2);
        gap.pour(spell_of(gap, "kindle"), Vec3{4.0, 3.0, 1.2}, Vec3{1.0, 0.0, 0.0});
        run(gap, 60 * 30);
        check(sum_over(gap, "grass", 0, 20, 0, 11, 2, 2) < 21 * 12 * 0.5, "1: the near side burns");
        check(sum_over(gap, "grass", 30, 47, 0, 11, 2, 2) > beyond * 0.95, "1: and it stops at bare ground");
    }

    // ---- 2. what grows follows its formulas ---------------------------------------------------------------------
    {
        TerrainConfig config;
        const std::shared_ptr<const WorldMap> map = WorldMap::of(config);
        for (const std::string& e : map->errors()) {
            std::cerr << "  " << e << "\n";
        }
        check(map->errors().empty(), "2: the species read");
        const std::vector<TreeSpecies>& species = map->species();
        std::cout << "  " << species.size() << " kinds of tree, " << map->flora_kinds().size() << " of plants\n";
        check(species.size() >= 15, "2: many kinds");
        std::set<TreeSpecies::Form> forms;
        for (const TreeSpecies& sp : species) {
            forms.insert(sp.form);
            const int cells = static_cast<int>((sp.height_min + sp.height_max) * 0.5 * 2.0);
            const std::vector<TreeVoxel> tree = grow_tree(sp, 42, cells, 0.5);
            check(!tree.empty(), "2: " + sp.name + " grows");
            int top = 0;
            int top_wood = 0;
            int wood = 0;
            int leaves = 0;
            std::set<std::tuple<int, int, int>> solid;
            for (const TreeVoxel& v : tree) {
                top = std::max(top, static_cast<int>(v.dz));
                solid.insert({v.dx, v.dy, v.dz});
                if (v.wood != 0) {
                    ++wood;
                    top_wood = std::max(top_wood, static_cast<int>(v.dz));
                } else {
                    ++leaves;
                }
            }
            check(top >= cells * 0.6, "2: " + sp.name + " is as tall as it should be (" + std::to_string(top) + " of " + std::to_string(cells) + ")");
            check(top <= cells + sp.leaf * 2.0 + 3.0, "2: " + sp.name + " is no taller than the map thinks: " + std::to_string(top));
            if (sp.form != TreeSpecies::Form::Column && sp.form != TreeSpecies::Form::Snag) {
                check(leaves > 0, "2: " + sp.name + " has leaves");
            }
            // Joined: from the foot everything that is wood can be reached through the faces of wood and leaves.
            if (sp.form != TreeSpecies::Form::Palm) {
                std::set<std::tuple<int, int, int>> seen;
                std::vector<std::tuple<int, int, int>> open;
                for (const auto& p : solid) {
                    if (std::get<2>(p) <= 2) {
                        open.push_back(p);
                        seen.insert(p);
                    }
                }
                while (!open.empty()) {
                    const auto [x, y, z] = open.back();
                    open.pop_back();
                    // Through a face: wood does not hold wood through a corner.
                    for (const std::tuple<int, int, int>& n : {std::tuple<int, int, int>{x + 1, y, z}, {x - 1, y, z}, {x, y + 1, z}, {x, y - 1, z}, {x, y, z + 1}, {x, y, z - 1}}) {
                        if (solid.count(n) != 0 && seen.insert(n).second) {
                            open.push_back(n);
                        }
                    }
                }
                int floating = 0;
                for (const TreeVoxel& v : tree) {
                    floating += v.wood != 0 && seen.count({v.dx, v.dy, v.dz}) == 0 ? 1 : 0;
                }
                check(floating == 0, "2: " + sp.name + ": no wood floats (" + std::to_string(floating) + ")");
            }
            // The same seed, the same tree.
            const std::vector<TreeVoxel> again = grow_tree(sp, 42, cells, 0.5);
            bool same = again.size() == tree.size();
            for (std::size_t i = 0; same && i < tree.size(); ++i) {
                same = tree[i].dx == again[i].dx && tree[i].dy == again[i].dy && tree[i].dz == again[i].dz && tree[i].wood == again[i].wood;
            }
            check(same, "2: " + sp.name + ": the same seed grows the same tree");
        }
        check(forms.size() >= 5, "2: of every form");
        // No tree is a bare skeleton: whatever its height and seed, wood up in the crown has leaves about it.
        double barest = 1.0;
        std::string barest_name;
        for (const TreeSpecies& sp : species) {
            if (sp.leaf <= 0.0 || sp.form == TreeSpecies::Form::Palm) {
                continue; // a palm is a bare stem by nature, its leaves all at the top
            }
            for (const int cells : {static_cast<int>(sp.height_min * 2.0), static_cast<int>(sp.height_max * 2.0)}) {
                for (std::uint64_t seed = 1; seed <= 8; ++seed) {
                    const std::vector<TreeVoxel> tree = grow_tree(sp, seed, cells, 0.5);
                    std::set<std::tuple<int, int, int>> foliage;
                    for (const TreeVoxel& v : tree) {
                        if (v.wood == 0) {
                            foliage.insert({v.dx, v.dy, v.dz});
                        }
                    }
                    int upper = 0;
                    int covered = 0;
                    for (const TreeVoxel& v : tree) {
                        if (v.wood == 0 || v.dz < cells * 0.6) {
                            continue;
                        }
                        ++upper;
                        bool near_leaf = false;
                        for (int dz = -3; dz <= 3 && !near_leaf; ++dz) {
                            for (int dy = -3; dy <= 3 && !near_leaf; ++dy) {
                                for (int dx = -3; dx <= 3 && !near_leaf; ++dx) {
                                    near_leaf = foliage.count({v.dx + dx, v.dy + dy, v.dz + dz}) != 0;
                                }
                            }
                        }
                        covered += near_leaf ? 1 : 0;
                    }
                    const double share = upper > 0 ? static_cast<double>(covered) / upper : (foliage.empty() ? 0.0 : 1.0);
                    if (share < barest) {
                        barest = share;
                        barest_name = sp.name + " " + std::to_string(cells);
                    }
                    check(!foliage.empty(), "2: " + sp.name + " has a crown at every height");
                }
            }
        }
        std::cout << "  barest crown: " << barest_name << ", " << barest * 100.0 << "% of the wood up there has leaves by it" << std::endl;
        check(barest > 0.4, "2: no tree is a bare skeleton: " + barest_name);
        // Greenhill: d = (H / slender)^(3/2). Thickness at breast height, a tall one against a short one, of one kind.
        const auto girth = [](const TreeSpecies& sp, int cells) {
            std::map<std::pair<int, int>, int> at;
            int across = 0;
            for (const TreeVoxel& v : grow_tree(sp, 7, cells, 0.5)) {
                if (v.wood != 0 && v.dz == 8) {
                    ++across;
                }
            }
            return across;
        };
        const auto find = [&](const char* name) {
            const auto it = std::find_if(species.begin(), species.end(), [&](const TreeSpecies& s) { return s.name == name; });
            check(it != species.end(), std::string("2: there is ") + name);
            return it == species.end() ? species.front() : *it;
        };
        const TreeSpecies oak = find("oak");
        const TreeSpecies birch = find("birch");
        const int small_oak = girth(oak, 24);
        const int big_oak = girth(oak, 80);
        const int same_birch = girth(birch, 80);
        std::cout << "  voxels through a trunk: oak 12 m " << small_oak << ", oak 40 m " << big_oak << ", birch 40 m " << same_birch << "\n";
        check(big_oak > small_oak * 3, "2: taller is thicker, faster than it is taller");
        check(same_birch < big_oak, "2: and a slender kind is thinner");
    }

    // ---- 3. and 4. the land --------------------------------------------------------------------------------------
    {
        TerrainConfig config;
        const std::shared_ptr<const WorldMap> map = WorldMap::of(config);
        std::map<Biome, std::set<std::string>> big;
        for (const TreeSpecies& sp : map->species()) {
            for (const Biome b : sp.biomes) {
                big[b].insert(sp.name);
            }
        }
        for (const Biome b : {Biome::Forest, Biome::Jungle, Biome::Taiga, Biome::Swamp, Biome::Savanna, Biome::Desert}) {
            check(big[b].size() >= 3, std::string("3: ") + biome_name(b) + " has its own kinds of tree");
        }
        check(big[Biome::Jungle] != big[Biome::Taiga], "3: jungle and taiga share nothing");
        check(map->growth_of(Biome::Jungle).big > map->growth_of(Biome::Desert).big * 3.0, "3: a jungle is thick, a desert is not");
        check(map->growth_of(Biome::Ocean).big == 0.0, "3: and nothing in the sea");
        int peak = 0;
        int plants = 0;
        int columns = 0;
        std::map<std::string, int> kinds;
        int deepest_cut = 0;
        const int step = WorldMap::kNodeCells;
        for (int y = -WorldMap::kHalfCells + step; y < WorldMap::kHalfCells - step; y += step) {
            int before = -1;
            for (int x = -WorldMap::kHalfCells + step; x < WorldMap::kHalfCells - step; x += step) {
                const Column c = map->column(x, y, 16);
                peak = std::max(peak, c.top);
                if (before >= 0) {
                    deepest_cut = std::max(deepest_cut, before - c.top);
                }
                before = c.top;
                if (c.water < 0) {
                    ++columns;
                    if (const FloraKind* f = map->flora_at(x, y, c)) {
                        ++plants;
                        ++kinds[f->name];
                    }
                }
            }
        }
        const double peak_m = (peak - config.sea_level) * config.cell;
        std::cout << "  the highest ground " << peak_m << " m over the sea; steepest fall in 128 m: " << deepest_cut * config.cell << " m; plants on " << plants << " of " << columns << " columns sampled, " << kinds.size() << " kinds\n";
        check(peak_m > 400.0, "4: mountains stand hundreds of metres over the sea");
        check(deepest_cut * config.cell > 60.0, "4: and there are cliffs and rifts");
        check(plants > columns / 200, "3: things grow on the ground");
        check(kinds.size() >= 6, "3: all sorts of things");

        // ---- 5. a house or two ------------------------------------------------------------------------------------
        int lone = 0;
        int small_villages = 0;
        int with_streets = 0;
        for (const Settlement& s : map->settlements()) {
            if (s.fell >= 0) {
                continue;
            }
            const auto houses = std::count_if(s.buildings.begin(), s.buildings.end(), [](const Building& b) { return b.kind == Building::Kind::House; });
            if (s.lone) {
                ++lone;
                with_streets += s.streets.empty() ? 0 : 1;
            } else if (houses <= 2 || (s.tier >= 1 && houses <= 4)) {
                ++small_villages;
            }
        }
        std::cout << "  " << lone << " homesteads\n";
        check(lone > 0, "5: some places are a family's own");
        check(with_streets == 0, "5: and have no streets");
        check(small_villages == 0, "5: nothing else is that small");
        // No road goes to one. Roads end in the middle of what they join, so there the ground is cleared; here it is not.
        int road_to_lone = 0;
        for (const Settlement& s : map->settlements()) {
            if (s.lone && s.buildings.empty() == false) {
                road_to_lone += map->column(s.x, s.y).clear && std::none_of(s.buildings.begin(), s.buildings.end(), [&](const Building& b) { return s.x >= b.x0 && s.x <= b.x1 && s.y >= b.y0 && s.y <= b.y1; }) ? 1 : 0;
            }
        }
        std::cout << "  of them, " << road_to_lone << " with cleared ground at the middle\n";
    }

    // ---- 6. what holds what up ---------------------------------------------------------------------------------------
    {
        // Two trees on the floor. One is cut through at the foot; the other is left alone.
        Playground g = arena();
        run(g, 5);
        const auto tree = [&](int x, int y) {
            paint(g, "wood", x, x, y, y, 2, 14, 1.0f);
            paint(g, "leaf", x - 2, x + 2, y - 2, y + 2, 13, 15, 1.0f);
            paint(g, "wood", x, x, y, y, 2, 14, 1.0f);
        };
        tree(6, 6);
        tree(40, 6);
        run(g, 10);
        const auto crown = [&](int x, int y) { return sum_over(g, "leaf", x - 2, x + 2, y - 2, y + 2, 13, 15) + sum_over(g, "wood", x, x, y, y, 5, 14); };
        const double crown_cut0 = crown(6, 6);
        const double crown_left0 = crown(40, 6);
        const double matter0 = g.total("solid") + g.total("wood") + g.total("rubble");
        // A cut through the trunk: what is below goes, and something gave way here.
        paint(g, "wood", 6, 6, 6, 6, 3, 4, 0.0f);
        Pour blow{.name = "t", .field = "mana_force", .amount = 1.0, .shape = Pour::Shape::Ball, .radius = 0.9};
        g.pour(blow, Vec3{3.25, 3.25, 2.0}, Vec3{1.0, 0.0, 0.0});
        double fell = 0.0;
        int ticks = 0;
        for (; ticks < 60 * 20 && crown(6, 6) > crown_cut0 * 0.1; ++ticks) {
            run(g, 1);
            fell = crown_cut0 - crown(6, 6);
        }
        std::cout << "  a trunk cut through: its crown (" << crown_cut0 << ") came down in " << ticks / 60.0 << " s, the other still has " << crown(40, 6) << " of " << crown_left0 << "\n";
        check(fell > crown_cut0 * 0.8, "6: a trunk cut through drops what stood on it");
        check(crown(40, 6) > crown_left0 * 0.99, "6: and a tree left alone stands");
        check(sum_over(g, "wood", 6, 6, 6, 6, 2, 2) > 0.5, "6: the stump stands, it is joined to the ground");
        run(g, 60);
        const double matter1 = g.total("solid") + g.total("wood") + g.total("rubble");
        std::cout << "  wood, stone and rubble: " << matter0 << " -> " << matter1 << " (the cut took 2, the blow what it broke)\n";
        check(matter1 <= matter0 && matter1 > matter0 - 30.0, "6: what broke is mostly dust, and nothing was made");

        // A slab in the air comes down; a bridge joined to the long wall at one end does not. (The wall: x = 30..31, z up to 7.)
        Playground b = arena();
        run(b, 5);
        paint(b, "solid", 8, 14, 8, 12, 9, 9, 1.0f);     // the slab: joined to nothing
        paint(b, "solid", 20, 29, 12, 14, 6, 6, 1.0f);   // the bridge, off the wall
        run(b, 5);
        const double slab0 = sum_over(b, "solid", 8, 14, 8, 12, 9, 9);
        const double bridge0 = sum_over(b, "solid", 20, 29, 12, 14, 6, 6);
        // Something gives way near each.
        Pour shake{.name = "t", .field = "mana_force", .amount = 0.4, .shape = Pour::Shape::Ball, .radius = 0.8};
        b.pour(shake, Vec3{5.5, 5.0, 5.0}, Vec3{1.0, 0.0, 0.0});
        b.pour(shake, Vec3{11.0, 6.5, 3.3}, Vec3{1.0, 0.0, 0.0});
        run(b, 60 * 12);
        const double slab1 = sum_over(b, "solid", 8, 14, 8, 12, 9, 9);
        const double bridge1 = sum_over(b, "solid", 20, 29, 12, 14, 6, 6);
        std::cout << "  a slab in the air: " << slab0 << " -> " << slab1 << "; a bridge off a wall: " << bridge0 << " -> " << bridge1 << "\n";
        check(slab1 < slab0 * 0.2, "6: a slab joined to nothing comes down");
        check(bridge1 > bridge0 * 0.8, "6: a bridge joined at one end stands");
        check(b.total("rubble") > 1.0, "6: a little of what came down is left lying");
        check(b.violations().empty(), "6: no contract broken" + (b.violations().empty() ? std::string() : ": " + b.violations()[0].term + " " + b.violations()[0].detail));
        check(g.violations().empty(), "6: nor there" + (g.violations().empty() ? std::string() : ": " + g.violations()[0].term + " " + g.violations()[0].detail));
    }

    // ---- 7. a body's numbers are the law's ------------------------------------------------------------------------------
    {
        std::vector<LawSource> laws;
        std::string spells;
        std::string path;
        std::vector<std::string> errors;
        Playground::read_default_law(laws, spells, path, errors);
        laws[0].text += "\nparam body_jump_probe = 1\n";
        std::size_t at = laws[0].text.find("param body_walk = 7");
        check(at != std::string::npos, "7: the law says how fast you walk");
        if (at != std::string::npos) {
            laws[0].text.replace(at, 19, "param body_walk = 2");
        }
        std::optional<Playground> g = Playground::create(laws, spells, errors);
        check(g.has_value(), "7: it loads");
        if (g) {
            g->build_arena();
            run(*g, 30);
            const double x0 = g->player().position.x;
            for (int i = 0; i < 60; ++i) {
                g->step(Control{.move_x = 1.0});
            }
            check(g->player().position.x - x0 < 2.2, "7: and that is how fast you walk");
        }
    }

    // 7b. IN A REAL WORLD: a wood stands when something happens near it, and a house without its base comes down.
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
        config.window_chunks = 7;
        config.window_layers = 3;
        config.far_life = false;
        std::optional<Playground> w = Playground::create(laws, spells, errors, config);
        check(w.has_value(), "7b: the world makes");
        w->build_open_world();
        const TerrainConfig& t = w->config().terrain;
        const auto map = w->map();
        const auto go = [&](int wx, int wy, int top) {
            const int cz = (top + 4) / t.chunk;
            w->center_window_on(wx / t.chunk, wy / t.chunk, cz);
            w->player().velocity = Vec3{};
            w->player().position = Vec3{(wx + 0.5) * t.cell, (wy + 0.5) * t.cell, (top + 3) * t.cell} - w->origin();
            w->step(Control{});
        };
        const Pour* force = nullptr;
        for (const Pour& s : w->pours()) {
            force = s.name == "force" ? &s : force;
        }
        // A forest: a blow on the ground in the middle of it.
        bool found = false;
        for (int ring = 200; ring < 30000 && !found; ring += 150) {
            for (int a = 0; a < 48 && !found; ++a) {
                const int wx = static_cast<int>(ring * std::cos(a * 0.1309));
                const int wy = static_cast<int>(ring * std::sin(a * 0.1309));
                const Column col = map->column(wx, wy);
                if (map->biome_at(wx, wy) != Biome::Forest || col.water >= 0 || map->settlement_at(wx, wy) != nullptr) {
                    continue;
                }
                go(wx, wy, col.top);
                if (w->total("wood") + w->total("leaf") > 2000.0) {
                    found = true;
                }
            }
        }
        check(found, "7b: a wood to stand in");
        const double wood0 = w->total("wood");
        const double leaf0 = w->total("leaf");
        const GridShape& g = w->fields().shape();
        const Vec3 me = w->player().position;
        for (int k = 0; k < 4; ++k) {
            const Vec3 at = w->pick(Vec3{me.x + 6.0 * k - 9.0, me.y, g.nz * g.cell - 0.1}, Vec3{0.0, 0.0, -1.0}).value_or(me);
            w->pour(*force, at, Vec3{1.0, 0.0, 0.0});
        }
        for (int tick = 0; tick < 60 * 10; ++tick) {
            w->player().hp = w->player().hp_max;
            w->step(Control{});
        }
        std::cout << "  a blow in a wood: wood " << wood0 << " -> " << w->total("wood") << ", leaves " << leaf0 << " -> " << w->total("leaf") << ", rubble " << w->total("rubble") << std::endl;
        check(w->total("wood") > wood0 * 0.97, "7b: a wood stands when the ground is struck beside it");
        check(w->total("leaf") > leaf0 * 0.95, "7b: and keeps its leaves");
        // A house: left alone, shaken, and with its base taken out.
        const Settlement* town = nullptr;
        const Building* house = nullptr;
        for (const Settlement& s : map->settlements()) {
            if (s.fell >= 0 || s.tier < 1 || s.burned) {
                continue;
            }
            for (const Building& b : s.buildings) {
                if (b.kind == Building::Kind::House && b.ruin == 0.0f && !b.burned) {
                    town = &s;
                    house = &b;
                    break;
                }
            }
            if (town != nullptr) {
                break;
            }
        }
        check(house != nullptr, "7b: a house");
        if (house != nullptr) {
            const int cx = (house->x0 + house->x1) / 2;
            const int cy = (house->y0 + house->y1) / 2;
            go(cx, cy, house->ground + 3);
            const Vec3 o = w->origin();
            const int bx0 = static_cast<int>(std::floor((house->x0 - 1) * t.cell - o.x) / g.cell);
            const int bx1 = static_cast<int>(std::floor((house->x1 + 1) * t.cell - o.x) / g.cell);
            const int by0 = static_cast<int>(std::floor((house->y0 - 1) * t.cell - o.y) / g.cell);
            const int by1 = static_cast<int>(std::floor((house->y1 + 1) * t.cell - o.y) / g.cell);
            const int base = static_cast<int>(std::floor(house->ground * t.cell - o.z) / g.cell);
            int bx_top = 0;
            int by_top = 0;
            int bz_top = 0;
            {
                int a0, b0, a1, b1, top;
                house->bounds(map->styles(), a0, b0, a1, b1, top);
                bx_top = a1;
                by_top = b1;
                bz_top = static_cast<int>(std::floor(top * t.cell - o.z) / g.cell);
            }
            (void)bx_top;
            (void)by_top;
            int world_top = 0;
            {
                int a0, b0, a1, b1, top;
                house->bounds(map->styles(), a0, b0, a1, b1, top);
                world_top = top;
            }
            // In the world's cells: you fall with the house, and the window goes with you.
            const auto stands = [&]() {
                const FieldSet& f = w->fields();
                const Vec3 now = w->origin();
                const auto cell_x = [&](int wx) { return static_cast<int>(std::floor(wx * t.cell - now.x) / g.cell); };
                const auto cell_y = [&](int wy) { return static_cast<int>(std::floor(wy * t.cell - now.y) / g.cell); };
                const auto cell_z = [&](int wz) { return static_cast<int>(std::floor(wz * t.cell - now.z) / g.cell); };
                double sum = 0.0;
                for (const char* name : {"solid", "wood", "leaf"}) {
                    const std::vector<float>& v = f.values(f.find(name));
                    const std::vector<float>& built = f.values(f.find("built")); // the house, and not the hill behind it
                    for (int z = std::max(cell_z(house->ground + 4), 0); z <= std::min(cell_z(world_top), g.nz - 1); ++z) {
                        for (int y = std::max(cell_y(house->y0 - 1), 0); y <= std::min(cell_y(house->y1 + 1), g.ny - 1); ++y) {
                            for (int x = std::max(cell_x(house->x0 - 1), 0); x <= std::min(cell_x(house->x1 + 1), g.nx - 1); ++x) {
                                sum += v[g.index(x, y, z)] * built[g.index(x, y, z)];
                            }
                        }
                    }
                }
                return sum;
            };
            for (int tick = 0; tick < 20; ++tick) {
                w->step(Control{});
            }
            const double whole = stands();
            const Vec3 by_it = Vec3{(bx1 + 3) * g.cell, (by0 + by1) * 0.5 * g.cell, (base + 1) * g.cell};
            w->pour(*force, by_it, Vec3{1.0, 0.0, 0.0});
            for (int tick = 0; tick < 60 * 8; ++tick) {
                w->player().hp = w->player().hp_max;
                w->step(Control{});
            }
            const double shaken = stands();
            std::cout << "  a blow beside a house: " << whole << " -> " << shaken << std::endl;
            check(whole > 30.0, "7b: the house is in the lattice");
            check(shaken > whole * 0.9, "7b: a house stands when the ground is struck beside it");
            // Its base out, all the way round: nothing is left to go down through.
            {
                FieldSet& f = w->solver().fields();
                for (const char* name : {"solid", "wood", "leaf", "rubble"}) {
                    std::vector<float>& v = f.values(f.find(name));
                    for (int z = base - 2; z <= base + 3 && z < g.nz; ++z) {
                        for (int y = std::max(by0, 0); y <= std::min(by1, g.ny - 1); ++y) {
                            for (int x = std::max(bx0, 0); x <= std::min(bx1, g.nx - 1); ++x) {
                                if (z >= 0) {
                                    v[g.index(x, y, z)] = 0.0f;
                                }
                            }
                        }
                    }
                }
                w->solver().refresh_derived();
            }
            const double cut = stands();
            w->pour(*force, Vec3{(bx0 + bx1) * 0.5 * g.cell, (by0 + by1) * 0.5 * g.cell, (base + 5) * g.cell}, Vec3{1.0, 0.0, 0.0});
            for (int tick = 0; tick < 60 * 14; ++tick) {
                w->player().hp = w->player().hp_max;
                w->step(Control{});
            }
            const double after = stands();
            std::cout << "  a house with its base out: " << cut << " -> " << after << ", rubble " << w->total("rubble") << std::endl;
            // (What is left leans on the house next door: stone spans 8 m.)
            check(after < cut * 0.5, "7b: a house with its base out comes down");
            check(w->total("rubble") > 10.0, "7b: as rubble");
        }
    }

    // 8. THE FRAME DOES NOT WAIT FOR THE LAW. Falsified by the same world in step with it.
    {
        const auto worst_tick = [](bool never_waits, double& law_ms, double& burned) {
            std::vector<std::string> errors;
            std::vector<LawSource> laws;
            std::string spells;
            std::string path;
            double worst = 0.0;
            if (!Playground::read_default_law(laws, spells, path, errors)) {
                return worst;
            }
            Playground::Config config;
            config.open_world = true;
            config.async_fields = true;
            config.field_every = 4;
            config.window_chunks = 5;
            config.window_layers = 3;
            config.field_never_waits = never_waits;
            std::optional<Playground> world = Playground::create(laws, spells, errors, config);
            if (!world) {
                return worst;
            }
            for (int tick = 0; tick < 120; ++tick) {
                const auto start = std::chrono::steady_clock::now();
                world->step(Control{});
                if (tick > 20) {
                    worst = std::max(worst, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
                }
            }
            std::this_thread::sleep_for(std::chrono::seconds(2)); // the ticks above took no time at all: let a step finish
            world->step(Control{});
            law_ms = world->field_ms();
            burned = static_cast<double>(world->tick());
            return worst;
        };
        double law_waiting = 0.0;
        double law_free = 0.0;
        double ticks = 0.0;
        const double waiting = worst_tick(false, law_waiting, ticks);
        const double free_running = worst_tick(true, law_free, ticks);
        std::cout << "  worst tick: in step with the law " << waiting << " ms (law " << law_waiting << " ms); not waiting " << free_running << " ms (law " << law_free << " ms)" << std::endl;
        check(law_free > 0.0, "8: the law still runs");
        check(waiting > law_waiting * 0.5, "8: in step, a tick waits for the law (or the next claim says nothing)");
        check(free_running < waiting * 0.5, "8: not in step, it does not");
    }
    return fieldengine::gate::report("voxel_growth");
}
