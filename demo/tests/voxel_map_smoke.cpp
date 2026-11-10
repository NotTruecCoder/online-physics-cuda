// THE MAP IS A WORLD, AT EVERY SCALE. Claims about the layers of WorldMap and what goes on beyond the window:
//   1. The layers have what they promise: continents and island chains, climates, rivers, peoples with a history, roads.
//   2. It is a pure function: the same seed, the same bits, whatever was asked for first.
//   3. Water stays where it is put: every level is held in by what stands next to it, and the law leaves it there.
//   4. Every whole building can be walked: from the street, through its door, to every room on every floor.
//      The style file is checked when it is read. Houses stand under the law, and timber burns by it.
//   5. Each layer has its own far: fine noise and thin rivers drop out, big rivers stay, trees turn to canopy,
//      houses to roofs, towns to patches. Far columns stand where the near ones do.
//   6. height_bounds really bounds: no matter of the world lies outside it.
//   7. The window follows you up and down, and a body falling off a mountain lands, with no jump in the world.
//   8. Beyond the window: a fire goes on burning a tree and wakes the next chunk; creatures walk on and come back;
//      the land's law stays in bounds. All of it as deterministic as the rest.
//
// Falsified by: dropping the containment in tile() (3 fails); a door on the wrong side, no stair hole, partition
// without a gap (4 fails); keeping thin rivers at every factor, a far column ignoring buildings (5 fails);
// bounds without the features' margin (6 fails); not shifting bodies in z (7 fails); note_busy never noting,
// neighbours never woken, far bodies frozen (8 fails).
#include "world/Playground.hpp"
#include "tests/Gate.hpp"
#include "tests/Pours.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>

using namespace fieldengine::field;
using fieldengine::gate::check;

namespace {

std::optional<Playground> make_world(int window = 3, bool async = false)
{
    std::vector<std::string> errors;
    std::vector<LawSource> laws;
    std::string spells;
    std::string path;
    Playground::read_default_law(laws, spells, path, errors);
    Playground::Config c;
    c.open_world = true;
    c.window_chunks = window;
    c.field_every = 4;
    c.async_fields = async;
    std::optional<Playground> world = Playground::create(laws, spells, errors, c);
    for (const std::string& e : errors) {
        std::cerr << "  " << e << "\n";
    }
    if (world) {
        world->build_open_world();
    }
    return world;
}

// You, at the middle of a window on this chunk: otherwise the window walks straight back to where you were.
void go_to(Playground& world, int cx, int cy, int cz)
{
    world.center_window_on(cx, cy, cz);
    const TerrainConfig& t = world.config().terrain;
    const int wx = cx * t.chunk + t.chunk / 2;
    const int wy = cy * t.chunk + t.chunk / 2;
    const double top = (t.chunk * world.window_layers() + world.window_cz() * t.chunk) * t.cell;
    world.player().velocity = Vec3{};
    world.player().position = Vec3{(wx + 0.5) * t.cell, (wy + 0.5) * t.cell, std::min(top, (terrain_top(t, wx, wy) + 3) * t.cell)} - world.origin();
}

void check_layers(const WorldMap& map, const TerrainConfig& t)
{
    std::map<Biome, int> biomes;
    int land = 0;
    int samples = 0;
    int highest = 0;
    int lowest = t.nz;
    for (int y = -WorldMap::kHalfCells; y < WorldMap::kHalfCells; y += 1024) {
        for (int x = -WorldMap::kHalfCells; x < WorldMap::kHalfCells; x += 1024) {
            const Column c = map.column(x, y, 64);
            ++samples;
            land += c.water < 0 ? 1 : 0;
            ++biomes[map.biome_at(x, y)];
            highest = std::max(highest, c.top);
            lowest = std::min(lowest, c.top);
        }
    }
    std::cout << "  land " << 100 * land / samples << "%, ground from " << (lowest - t.sea_level) * t.cell << " m to +" << (highest - t.sea_level) * t.cell
              << " m around the sea, " << map.rivers() << " river reaches, " << map.roads() << " road reaches\n  biomes:";
    for (const auto& [biome, count] : biomes) {
        std::cout << " " << biome_name(biome) << " " << 100 * count / samples << "%";
    }
    std::cout << "\n";
    check(land * 100 > samples * 20 && land * 100 < samples * 70, "1: land and sea");
    // Separate land masses: a flood fill over the map, node by node (at 1 km).
    {
        const int side = 2 * WorldMap::kHalfCells / 2048;
        std::vector<char> dry(static_cast<std::size_t>(side) * side);
        for (int y = 0; y < side; ++y) {
            for (int x = 0; x < side; ++x) {
                dry[static_cast<std::size_t>(y * side + x)] = map.column(x * 2048 - WorldMap::kHalfCells, y * 2048 - WorldMap::kHalfCells, 64).water < 0 ? 1 : 0;
            }
        }
        std::vector<int> sizes;
        for (int start = 0; start < side * side; ++start) {
            if (!dry[static_cast<std::size_t>(start)]) {
                continue;
            }
            int size = 0;
            std::vector<int> open{start};
            dry[static_cast<std::size_t>(start)] = 0;
            while (!open.empty()) {
                const int n = open.back();
                open.pop_back();
                ++size;
                for (const int m : {n - 1, n + 1, n - side, n + side}) {
                    if (m >= 0 && m < side * side && dry[static_cast<std::size_t>(m)]) {
                        dry[static_cast<std::size_t>(m)] = 0;
                        open.push_back(m);
                    }
                }
            }
            sizes.push_back(size);
        }
        std::sort(sizes.rbegin(), sizes.rend());
        const int big = static_cast<int>(std::count_if(sizes.begin(), sizes.end(), [](int v) { return v >= 400; }));
        const int small = static_cast<int>(std::count_if(sizes.begin(), sizes.end(), [](int v) { return v >= 3 && v < 400; }));
        std::cout << "  land masses: " << big << " of 400 km2 or more (largest " << (sizes.empty() ? 0 : sizes[0]) << " km2), " << small << " islands\n";
        check(big >= 3, "1: more than one land to sail between");
        const long long all_land = std::accumulate(sizes.begin(), sizes.end(), 0LL);
        check(sizes.size() >= 2 && sizes[0] * 10 < all_land * 8 && sizes[1] >= 400, "1: and no one land is the whole world");
        check(all_land >= 20000, "1: tens of thousands of square kilometres of it");
        check(small >= 20, "1: and islands");
    }
    check((highest - t.sea_level) * t.cell > 100.0, "1: mountains worth the height of the world");
    check((lowest - t.sea_level) * t.cell < -15.0, "1: and a sea with a floor");
    check(highest + 40 < t.nz, "1: nothing reaches the ceiling");
    check(biomes.size() >= 9, "1: climates differ (" + std::to_string(biomes.size()) + " biomes)");
    check((biomes[Biome::Forest] + biomes[Biome::Grassland] + biomes[Biome::Jungle] + biomes[Biome::Taiga] + biomes[Biome::Savanna]) * 3 >= land, "1: rain falls: a third of the land is green");
    check(map.rivers_run_down(), "3: every river runs down");
    check(map.rivers() > 200, "1: rivers");
    check(map.roads() > 50, "1: roads between the settlements");
    int ruins = 0;
    int burned = 0;
    int buildings = 0;
    std::map<int, int> tiers;
    std::set<int> cultures;
    std::set<int> styles;
    for (const Settlement& s : map.settlements()) {
        ruins += s.fell >= 0 ? 1 : 0;
        burned += s.burned ? 1 : 0;
        buildings += static_cast<int>(s.buildings.size());
        ++tiers[s.tier];
        cultures.insert(s.culture);
        styles.insert(s.style);
    }
    std::cout << "  " << map.settlements().size() << " settlements (" << tiers[0] << " hamlets, " << tiers[1] << " villages, " << tiers[2] << " towns, " << tiers[3]
              << " cities), " << ruins << " in ruins, " << burned << " burned; " << buildings << " buildings, " << cultures.size() << " cultures, " << styles.size() << " styles; "
              << map.chronicle().size() << " events\n";
    check(map.settlements().size() >= 20, "1: people live here");
    check(tiers[1] + tiers[2] + tiers[3] >= 5, "1: some of it grew beyond hamlets");
    check(ruins >= 2 && burned >= 1, "1: history left ruins");
    check(cultures.size() >= 3 && styles.size() >= 3, "1: peoples build differently");
    check(map.chronicle().size() >= map.settlements().size(), "1: history is written down");
}

void check_purity(const TerrainConfig& t, const WorldMap& map)
{
    // A map made anew from the same seed, asked in another order.
    std::string text;
    {
        std::ifstream in("assets/world/styles.world", std::ios::binary);
        std::ostringstream all;
        all << in.rdbuf();
        text = all.str();
    }
    const WorldMap again(t.seed, t.sea_level, t.nz, t.chunk, text);
    bool same = again.settlements().size() == map.settlements().size() && again.rivers() == map.rivers() && again.roads() == map.roads();
    for (int k = 0; k < 400 && same; ++k) {
        const int x = -30000 + k * 149;
        const int y = 20000 - k * 97;
        const Column a = map.column(x, y);
        const Column b = again.column(x, y);
        same = a.top == b.top && a.water == b.water && a.surface == b.surface && a.turf == b.turf;
    }
    check(same, "2: the same seed makes the same map");
    int differ = 0;
    const WorldMap other(t.seed + 1, t.sea_level, t.nz, t.chunk, text);
    for (int k = 0; k < 100; ++k) {
        differ += other.column(-30000 + k * 500, 100).top != map.column(-30000 + k * 500, 100).top ? 1 : 0;
    }
    check(differ > 50, "2: another seed, another map");
    // A chunk comes out the same however it is reached.
    ChunkData first;
    ChunkData second;
    int hx = 0;
    int hy = 0;
    map.home(hx, hy);
    const int cx = hx / t.chunk;
    const int cy = hy / t.chunk;
    const int cz = map.column(hx, hy).top / t.chunk;
    generate_chunk(t, cx, cy, cz, first, nullptr);
    ChunkData elsewhere;
    generate_chunk(t, cx + 40, cy - 7, cz, elsewhere, nullptr);
    generate_chunk(t, cx, cy, cz, second, nullptr);
    bool chunk_same = first.fields.size() == second.fields.size();
    for (std::size_t i = 0; chunk_same && i < first.fields.size(); ++i) {
        chunk_same = first.fields[i].second == second.fields[i].second;
    }
    check(chunk_same, "2: a chunk is the same, whatever was generated in between");
    // tile() and column() agree: a seam between them would be a seam in the world.
    bool agree = true;
    const std::shared_ptr<const ColumnTile> tile = map.tile(cx, cy);
    for (int y = 0; y < t.chunk && agree; ++y) {
        for (int x = 0; x < t.chunk && agree; ++x) {
            const Column c = map.column(cx * t.chunk + x, cy * t.chunk + y);
            const Column& d = tile->columns[static_cast<std::size_t>(y * t.chunk + x)];
            agree = c.top == d.top && c.water == d.water;
        }
    }
    check(agree, "2: a column reads the same alone and in its chunk");
    // A river's chunk holds its water up to the top cell the column says.
    int rx = 0;
    int ry = 0;
    if (map.river_point(4.0, 40.0, rx, ry)) {
        const Column c = map.column(rx, ry);
        const int ccx = static_cast<int>(std::floor(static_cast<double>(rx) / t.chunk));
        const int ccy = static_cast<int>(std::floor(static_cast<double>(ry) / t.chunk));
        ChunkData water;
        generate_chunk(t, ccx, ccy, c.water / t.chunk, water, nullptr);
        const std::size_t i = (static_cast<std::size_t>(c.water % t.chunk) * t.chunk + static_cast<std::size_t>(ry - ccy * t.chunk)) * t.chunk + static_cast<std::size_t>(rx - ccx * t.chunk);
        check((*water.find("water"))[i] == 1.0f, "2: the top of the water is in its chunk");
    }
}

void check_water(const TerrainConfig& t, const WorldMap& map)
{
    // Every water column, around rivers of all sizes: nothing lower beside it, or something standing as high.
    int checked = 0;
    int leaks = 0;
    int tile_differs = 0;
    for (const double width : {3.0, 6.0, 12.0}) {
        int rx = 0;
        int ry = 0;
        if (!map.river_point(width, 40.0, rx, ry)) {
            continue;
        }
        for (int y = ry - 60; y < ry + 60; ++y) {
            for (int x = rx - 60; x < rx + 60; ++x) {
                const Column c = map.column(x, y);
                const int ccx = static_cast<int>(std::floor(static_cast<double>(x) / t.chunk));
                const int ccy = static_cast<int>(std::floor(static_cast<double>(y) / t.chunk));
                const Column& in_tile = map.tile(ccx, ccy)->columns[static_cast<std::size_t>((y - ccy * t.chunk) * t.chunk + x - ccx * t.chunk)];
                tile_differs += in_tile.water != c.water || in_tile.top != c.top ? 1 : 0;
                if (c.water < 0) {
                    continue;
                }
                ++checked;
                for (const auto [dx, dy] : {std::pair{1, 0}, std::pair{-1, 0}, std::pair{0, 1}, std::pair{0, -1}}) {
                    const Column n = map.column(x + dx, y + dy);
                    if (n.top < c.water && n.water < c.water) {
                        ++leaks;
                    }
                }
            }
        }
    }
    std::cout << "  water columns looked at: " << checked << ", with a way out: " << leaks << "\n";
    check(checked > 200, "3: there are rivers to look at");
    check(leaks == 0, "3: every level is held in");
    check(tile_differs == 0, "3: and a chunk says the same as the column");

    // And the law agrees: a stretch of river under the law for half a minute stays as it is.
    std::optional<Playground> world = make_world(5);
    int rx = 0;
    int ry = 0;
    check(map.river_point(6.0, 40.0, rx, ry), "3: a river above the sea");
    const Column here = map.column(rx, ry);
    go_to(*world, rx / t.chunk, ry / t.chunk, here.water / t.chunk);
    for (int tick = 0; tick < 20; ++tick) {
        world->step(Control{}); // you land; the law starts from what you left
    }
    const double before = world->total("water");
    std::vector<float> start = world->fields().values(world->fields().find("water"));
    for (int tick = 0; tick < 60 * 30; ++tick) {
        world->step(Control{});
    }
    const std::vector<float>& end = world->fields().values(world->fields().find("water"));
    double moved = 0.0;
    for (std::size_t i = 0; i < end.size(); ++i) {
        moved += std::abs(static_cast<double>(end[i]) - start[i]);
    }
    std::cout << "  river under the law, 30 s: water " << before << " -> " << world->total("water") << ", moved " << moved << "\n";
    check(before > 50.0, "3: the window holds a river");
    check(moved < before * 0.02 + 2.0, "3: under the law the river stays in its bed");
}

// The walkable reach of a building: a body two cells wide and two tall, a step of one cell up, any drop down.
void check_buildings(const TerrainConfig& t, const WorldMap& map)
{
    int tested = 0;
    int entered = 0;
    int all_rooms = 0;
    int all_floors = 0;
    std::string first_failure;
    const std::vector<Style>& styles = map.styles();
    std::map<std::int64_t, std::shared_ptr<ChunkData>> chunks;
    const auto solid = [&](int x, int y, int z) -> bool {
        const int cx = static_cast<int>(std::floor(static_cast<double>(x) / t.chunk));
        const int cy = static_cast<int>(std::floor(static_cast<double>(y) / t.chunk));
        const int cz = z / t.chunk;
        std::shared_ptr<ChunkData>& data = chunks[chunk_key(cx, cy, cz)];
        if (!data) {
            data = std::make_shared<ChunkData>();
            generate_chunk(t, cx, cy, cz, *data, nullptr);
        }
        const std::size_t i = (static_cast<std::size_t>(z - cz * t.chunk) * t.chunk + static_cast<std::size_t>(y - cy * t.chunk)) * t.chunk + static_cast<std::size_t>(x - cx * t.chunk);
        return (*data->find("solid"))[i] + (*data->find("wood"))[i] + (*data->find("leaf"))[i] + (*data->find("rubble"))[i] >= 0.5f;
    };
    constexpr int kToTest = 500;
    std::size_t every = std::max<std::size_t>(1, map.settlements().size() / 60);
    std::map<int, int> kinds_walked;
    for (std::size_t si = 0; si < map.settlements().size(); si += every) {
        const Settlement& s = map.settlements()[si];
        if (s.fell >= 0 || tested >= kToTest) {
            continue;
        }
        chunks.clear();
        for (const Building& b : s.buildings) {
            if (!b.has_rooms() || b.ruin > 0.0f || tested >= kToTest) {
                continue;
            }
            ++kinds_walked[static_cast<int>(b.kind)];
            ++tested;
            const Style& st = styles[static_cast<std::size_t>(b.style)];
            // Start in the street, in front of the door.
            const int out = st.raise + 3;
            int x = 0;
            int y = 0;
            switch (b.facing) {
            case 0: x = b.x1 + out; y = b.door; break;
            case 1: x = b.x0 - out - 1; y = b.door; break;
            case 2: x = b.door; y = b.y1 + out; break;
            default: x = b.door; y = b.y0 - out - 1; break;
            }
            int z = b.ground + 1;
            const auto free_at = [&](int ax, int ay, int az) {
                for (int dz = 0; dz < 2; ++dz) {
                    for (int dy = 0; dy < 2; ++dy) {
                        for (int dx = 0; dx < 2; ++dx) {
                            // (A door opens: that is what it is for.)
                            if (solid(ax + dx, ay + dy, az + dz) && !building_cell(b, st, ax + dx, ay + dy, az + dz).door) {
                                return false;
                            }
                        }
                    }
                }
                return true;
            };
            const auto standing = [&](int ax, int ay, int az) {
                return solid(ax, ay, az - 1) || solid(ax + 1, ay, az - 1) || solid(ax, ay + 1, az - 1) || solid(ax + 1, ay + 1, az - 1);
            };
            std::set<std::tuple<int, int, int>> seen;
            std::deque<std::tuple<int, int, int>> open;
            if (!free_at(x, y, z)) {
                ++z;
            }
            seen.insert({x, y, z});
            open.push_back({x, y, z});
            // How far it may wander: the building and a little around it.
            const int bx0 = b.x0 - st.raise - 6;
            const int bx1 = b.x1 + st.raise + 6;
            const int by0 = b.y0 - st.raise - 6;
            const int by1 = b.y1 + st.raise + 6;
            std::set<std::pair<int, int>> floors_reached;
            std::set<int> rooms_reached;
            while (!open.empty()) {
                const auto [ax, ay, az] = open.front();
                open.pop_front();
                if (ax >= b.x0 && ax + 1 <= b.x1 && ay >= b.y0 && ay + 1 <= b.y1 && az >= b.ground + 1 + st.raise) {
                    const int level = (az - (b.ground + 2 + st.raise)) / st.storey;
                    floors_reached.insert({level, 0});
                    if (level == 0) {
                        const int room = (b.part[0] >= 0 && ax > b.part[0] ? 1 : 0) + (b.part[1] >= 0 && ax > b.part[1] ? 1 : 0);
                        rooms_reached.insert(room);
                    }
                }
                for (const auto [dx, dy] : {std::pair{1, 0}, std::pair{-1, 0}, std::pair{0, 1}, std::pair{0, -1}}) {
                    const int nx = ax + dx;
                    const int ny = ay + dy;
                    if (nx < bx0 || nx > bx1 || ny < by0 || ny > by1) {
                        continue;
                    }
                    if (seen.size() > 6000) {
                        continue;
                    }
                    for (int dz = 1; dz >= -2; --dz) {
                        const int nz = az + dz;
                        if (free_at(nx, ny, nz) && standing(nx, ny, nz) && (dz <= 0 || free_at(ax, ay, az + 1))) {
                            if (seen.insert({nx, ny, nz}).second) {
                                open.push_back({nx, ny, nz});
                            }
                            break;
                        }
                    }
                }
            }
            const int rooms = 1 + (b.part[0] >= 0 ? 1 : 0) + (b.part[1] >= 0 ? 1 : 0);
            const bool in = !rooms_reached.empty();
            entered += in ? 1 : 0;
            all_rooms += static_cast<int>(rooms_reached.size()) == rooms ? 1 : 0;
            all_floors += static_cast<int>(floors_reached.size()) == b.floors ? 1 : 0;
            if (first_failure.empty() && (!in || static_cast<int>(rooms_reached.size()) != rooms || static_cast<int>(floors_reached.size()) != b.floors)) {
                first_failure = st.name + " (" + std::to_string(b.x0) + "," + std::to_string(b.y0) + ")-(" + std::to_string(b.x1) + "," + std::to_string(b.y1) + ") facing " +
                    std::to_string(b.facing) + " door " + std::to_string(b.door) + " floors " + std::to_string(b.floors) + " rooms " + std::to_string(rooms_reached.size()) + "/" +
                    std::to_string(rooms) + " floors reached " + std::to_string(floors_reached.size());
            }
        }
    }
    std::cout << "  " << tested << " buildings walked: entered " << entered << ", every room " << all_rooms << ", every floor " << all_floors << "\n";
    if (!first_failure.empty()) {
        std::cout << "  first failure: " << first_failure << "\n";
    }
    std::cout << "  walked: " << kinds_walked[0] << " houses, " << kinds_walked[1] << " halls, " << kinds_walked[2] << " square towers, " << kinds_walked[3] << " round towers\n";
    check(tested >= 100, "4: there are buildings to walk through");
    check(entered == tested, "4: every door opens on the street");
    check(all_rooms == tested, "4: every room can be reached");
    check(all_floors == tested, "4: every floor can be reached");

    // The file is read with suspicion.
    std::vector<std::string> errors;
    parse_styles("style a\n  biomes grassland\n  storey 3\n", errors);
    check(!errors.empty(), "4: a storey too low for a door is refused");
    errors.clear();
    parse_styles("style a\n  biomes marsh\n", errors);
    check(!errors.empty(), "4: an unknown biome is refused");
    errors.clear();
    parse_styles("style a\n  biomes grassland\n  wall stone marble\n", errors);
    check(!errors.empty(), "4: an unknown material is refused");
    errors.clear();
    parse_styles("style a\n  biomes grassland\n  floors 2 3\n  depth 12 12\n  storey 9\n", errors);
    check(!errors.empty(), "4: floors with no room for a stair are refused");
    check(map.errors().empty(), "4: the shipped styles are all well formed");
    // Every style in the world, the ones worked out too, is the same text a person could write.
    int bad_text = 0;
    for (const Style& style : map.styles()) {
        std::vector<std::string> parsed_errors;
        const std::vector<Style> back = parse_styles(style_text(style), parsed_errors);
        bad_text += !parsed_errors.empty() || back.size() != 1 || style_text(back[0]) != style_text(style) ? 1 : 0;
        if (back.size() == 1) {
            const Style& b = back[0];
            bad_text += b.jetty != style.jetty || b.wing != style.wing || b.yard != style.yard || b.chimney != style.chimney || b.round_towers != style.round_towers || b.layout != style.layout ||
                    b.temple != style.temple || b.roof != style.roof || b.pitch != style.pitch || b.raise != style.raise || b.piles != style.piles || b.ring_material != style.ring_material ||
                    (style.ring_material >= 0 && b.ring_round != style.ring_round) || b.upper_material != style.upper_material || b.trim_material != style.trim_material
                ? 1
                : 0;
        }
    }
    std::cout << "  " << map.styles().size() << " styles in the world, " << bad_text << " not well formed\n";
    check(bad_text == 0, "4: every style in the world reads back as the text it is");
}

// Nothing is buried: no hill stands over a wall, no ground over a floor. That is what levelling is for.
void check_not_buried(const TerrainConfig& t, const WorldMap& map)
{
    int buildings = 0;
    int buried = 0;
    int walls = 0;
    int walls_buried = 0;
    std::string first;
    for (const Settlement& s : map.settlements()) {
        for (const Building& b : s.buildings) {
            if (b.kind == Building::Kind::Ring) {
                if (b.ruin > 0.0f) {
                    continue; // what is left of it is not a wall
                }
                // The wall all round, every few cells.
                const int r = (b.x1 - b.x0) / 2;
                for (int k = 0; k < 64; ++k) {
                    const double a = 6.283185307 * k / 64.0;
                    const int x = b.part[0] == 1 ? s.x + static_cast<int>((r - 1) * std::cos(a)) : (k < 32 ? b.x0 + (b.x1 - b.x0) * (k % 16) / 16 : (k < 48 ? b.x0 : b.x1));
                    const int y = b.part[0] == 1 ? s.y + static_cast<int>((r - 1) * std::sin(a)) : (k < 16 ? b.y0 : (k < 32 ? b.y1 : b.y0 + (b.y1 - b.y0) * (k % 16) / 16));
                    if (building_cell(b, map.styles()[static_cast<std::size_t>(b.style)], x, y, b.ground + 3).kind == BuildingCell::Outside) {
                        continue; // a gate
                    }
                    const Column c = map.column(x, y);
                    if (c.water >= 0) {
                        continue;
                    }
                    // Right above the ground that is there: the wall.
                    ++walls;
                    const int cx = static_cast<int>(std::floor(static_cast<double>(x) / t.chunk));
                    const int cy = static_cast<int>(std::floor(static_cast<double>(y) / t.chunk));
                    const int z = c.top + 2;
                    ChunkData data;
                    generate_chunk(t, cx, cy, z / t.chunk, data, nullptr);
                    const std::size_t i = (static_cast<std::size_t>(z % t.chunk) * t.chunk + static_cast<std::size_t>(y - cy * t.chunk)) * t.chunk + static_cast<std::size_t>(x - cx * t.chunk);
                    if ((*data.find("solid"))[i] < 0.5f && (*data.find("wood"))[i] < 0.5f) {
                        ++walls_buried;
                    }
                }
                continue;
            }
            if (!b.has_rooms() && b.kind != Building::Kind::Stepped) {
                continue;
            }
            ++buildings;
            bool under = false;
            for (int y = b.y0; y <= b.y1 && !under; y += 2) {
                for (int x = b.x0; x <= b.x1 && !under; x += 2) {
                    const Column c = map.column(x, y);
                    under = c.top > b.ground || c.top < b.ground - 1 || c.water >= 0;
                }
            }
            buried += under ? 1 : 0;
            if (under && first.empty()) {
                first = s.name + " (" + std::to_string(b.x0) + ", " + std::to_string(b.y0) + ")";
            }
        }
    }
    std::cout << "  " << buildings << " buildings, " << buried << " not on level ground" << (first.empty() ? "" : ": first " + first) << "; " << walls << " points of town wall, " << walls_buried << " buried\n";
    check(buildings > 1000, "9: a world's worth of buildings");
    check(buried == 0, "9: every building stands on its own level ground, and none is under a hill");
    check(walls_buried == 0, "9: no ground stands over a town wall");
}

// No two towns alike, and for a reason: the land, then what happened.
void check_towns(const TerrainConfig& t, const WorldMap& map)
{
    (void)t;
    std::set<std::string> full;
    std::set<std::tuple<int, int, int, int, int, int>> looks;
    std::map<Biome, std::set<std::tuple<int, int, int, int>>> looks_by_land;
    std::map<Biome, int> towns_by_land;
    int towns = 0;
    int bad_adaption = 0;
    int cold_peoples = 0;
    int wet_peoples = 0;
    int dry_peoples = 0;
    std::string first_bad;
    std::set<int> layouts;
    std::set<int> roofs;
    std::set<int> temples;
    const auto style_of = [&](int id) -> const Style& { return map.styles()[static_cast<std::size_t>(id)]; };
    for (const Settlement& s : map.settlements()) {
        const Style& st = style_of(s.style);
        std::string text = style_text(st);
        text = text.substr(text.find('\n'));
        full.insert(text);
        if (s.tier < 1) {
            continue;
        }
        ++towns;
        looks.insert({st.wall_material, static_cast<int>(st.roof), st.roof_material, static_cast<int>(st.layout), st.raise > 1 ? 1 : 0, st.jetty ? 1 : 0});
        const Biome land = map.biome_at(s.x, s.y);
        looks_by_land[land].insert({st.wall_material, static_cast<int>(st.roof), st.roof_material, static_cast<int>(st.layout)});
        ++towns_by_land[land];
        layouts.insert(static_cast<int>(st.layout));
        roofs.insert(static_cast<int>(st.roof));
        temples.insert(static_cast<int>(st.temple));
    }
    // The way of a people fits the land it began in.
    for (const Culture& c : map.cultures()) {
        const Settlement& home = map.settlements()[static_cast<std::size_t>(c.home)];
        const Style& st = style_of(c.style);
        const double temperature = map.temperature_at(home.x, home.y);
        const Biome land = st.biomes.empty() ? Biome::Grassland : st.biomes[0];
        bool fits = true;
        const double at_making = temperature;
        cold_peoples += at_making < 5.5 ? 1 : 0;
        wet_peoples += land == Biome::Swamp || land == Biome::Jungle ? 1 : 0;
        dry_peoples += land == Biome::Desert ? 1 : 0;
        fits = fits && !(at_making < 5.5 && (st.roof == Style::Roof::Flat || st.roof == Style::Roof::Dome)); // snow lies on a flat roof
        fits = fits && !(land == Biome::Swamp && st.raise < 2);                                                   // the ground is wet
        fits = fits && !(land == Biome::Desert && st.wall != Style::Stuff::Stone);                                // nothing grows to build with
        fits = fits && !(land == Biome::Desert && st.roof_stuff == Style::Stuff::Leaf);
        fits = fits && st.roof_material != st.wall_material; // or a house is one lump
        if (!fits) {
            ++bad_adaption;
            if (first_bad.empty()) {
                first_bad = c.name + ": " + style_words(st) + " in " + biome_name(land);
            }
        }
    }
    std::cout << "  " << map.settlements().size() << " settlements, " << full.size() << " different ways of building; " << towns << " towns look " << looks.size() << " ways; " << layouts.size()
              << " layouts, " << roofs.size() << " roofs, " << temples.size() << " temples; " << map.cultures().size() << " peoples\n";
    if (!first_bad.empty()) {
        std::cout << "  out of place: " << first_bad << "\n";
    }
    // A city is a city: it has the houses for it, however its streets happen to run.
    std::map<int, int> houses_by_tier;
    std::map<int, int> places_by_tier;
    for (const Settlement& s : map.settlements()) {
        if (s.fell >= 0) {
            continue;
        }
        ++places_by_tier[s.tier];
        houses_by_tier[s.tier] += static_cast<int>(std::count_if(s.buildings.begin(), s.buildings.end(), [](const Building& b) { return b.kind == Building::Kind::House; }));
    }
    std::cout << "  houses on average: hamlet " << houses_by_tier[0] / std::max(places_by_tier[0], 1) << ", village " << houses_by_tier[1] / std::max(places_by_tier[1], 1) << ", town "
              << houses_by_tier[2] / std::max(places_by_tier[2], 1) << ", city " << houses_by_tier[3] / std::max(places_by_tier[3], 1) << "\n";
    check(houses_by_tier[1] >= 12 * places_by_tier[1], "10: a village is a dozen houses or more");
    check(houses_by_tier[2] >= 35 * places_by_tier[2], "10: a town is dozens");
    check(houses_by_tier[3] >= 70 * places_by_tier[3], "10: a city is a hundred or so");
    int temples_standing = 0;
    int could_have = 0;
    for (const Settlement& s : map.settlements()) {
        if (s.fell < 0 && s.tier >= 1) {
            ++could_have;
            temples_standing += std::any_of(s.buildings.begin(), s.buildings.end(), [](const Building& b) { return b.kind == Building::Kind::Hall || b.kind == Building::Kind::Stepped; }) ? 1 : 0;
        }
    }
    std::cout << "  " << temples_standing << " of " << could_have << " towns have a house of what they believe\n";
    check(temples_standing * 10 >= could_have * 6, "10: most towns have a temple or a hall");
    check(full.size() * 100 >= map.settlements().size() * 90, "10: nearly every settlement has its own way of building");
    check(looks.size() * 100 >= static_cast<std::size_t>(towns) * 25, "10: and you can see it: walls, roofs and plans differ from town to town");
    check(layouts.size() == 5 && roofs.size() >= 3 && temples.size() >= 3, "10: every plan, most roofs, most temples are in use");
    for (const auto& [land, count] : towns_by_land) {
        if (count >= 15) {
            check(looks_by_land[land].size() * 3 >= static_cast<std::size_t>(count), std::string("10: towns of one land still differ: ") + biome_name(land));
        }
    }
    std::cout << "  peoples from the cold " << cold_peoples << ", from wet land " << wet_peoples << ", from the desert " << dry_peoples << "\n";
    check(cold_peoples >= 2 && wet_peoples >= 2 && dry_peoples >= 2, "10: peoples from every kind of land, or the next claim says nothing");
    check(bad_adaption == 0, "10: a people builds for the land it began in");

    // What happened to a town is standing in it.
    int rebuilt = 0;
    int flooded = 0;
    int plagued = 0;
    int victors = 0;
    int taken = 0;
    int bad = 0;
    std::string first_wrong;
    const auto wrong = [&](const Settlement& s, const char* why) {
        ++bad;
        if (first_wrong.empty()) {
            first_wrong = s.name + ": " + why;
        }
    };
    const auto told = [&](const Settlement& s, const char* word) {
        return std::any_of(map.chronicle().begin(), map.chronicle().end(), [&](const Chronicle& e) { return e.settlement == s.id && e.text.find(word) != std::string::npos; });
    };
    for (const Settlement& s : map.settlements()) {
        if (s.fell >= 0) {
            continue;
        }
        const Style& st = style_of(s.style);
        int charred = 0;
        int empty = 0;
        int houses = 0;
        int monuments = 0;
        int graves = 0;
        std::set<int> builders;
        for (const Building& b : s.buildings) {
            charred += b.kind == Building::Kind::House && b.burned ? 1 : 0;
            empty += b.kind == Building::Kind::House && !b.burned && b.ruin > 0.0f ? 1 : 0;
            monuments += b.kind == Building::Kind::Monument ? 1 : 0;
            graves += b.kind == Building::Kind::Graves ? 1 : 0;
            if (b.kind == Building::Kind::House) {
                ++houses;
                builders.insert(b.style);
            }
        }
        if (s.burned_and_rebuilt > 0) {
            ++rebuilt;
            if (st.wall != Style::Stuff::Stone || st.roof_stuff != Style::Stuff::Stone) {
                wrong(s, "burned, and built in what burns again");
            }
            if (!told(s, "fire")) {
                wrong(s, "its fire is not told");
            }
        }
        if (s.floods > 0) {
            ++flooded;
            if (st.raise < 1) {
                wrong(s, "flooded, and still on the ground");
            }
            if (!told(s, "river rose")) {
                wrong(s, "its flood is not told");
            }
        }
        if (s.plagues > 0) {
            ++plagued;
            if (!told(s, "Plague")) {
                wrong(s, "its plague is not told");
            }
        }
        if (s.wars_won > 0) {
            ++victors;
            if (monuments == 0) {
                wrong(s, "won a war, and no stone says so");
            }
        }
        if (s.conquerors >= 0) {
            ++taken;
        }
        (void)charred;
        (void)empty;
        (void)graves;
    }
    // Over the world (a small place may have no lot in the quarter that fell), the scars are there.
    int scarred_burnt = 0;
    int scarred_plague = 0;
    int graves_seen = 0;
    int mixed = 0;
    for (const Settlement& s : map.settlements()) {
        if (s.fell >= 0) {
            continue;
        }
        std::set<int> builders;
        bool burnt = false;
        bool abandoned = false;
        bool stones = false;
        for (const Building& b : s.buildings) {
            burnt = burnt || (b.kind == Building::Kind::House && b.burned);
            abandoned = abandoned || (b.kind == Building::Kind::House && !b.burned && b.ruin > 0.0f);
            stones = stones || b.kind == Building::Kind::Graves;
            if (b.kind == Building::Kind::House) {
                builders.insert(b.style);
            }
        }
        scarred_burnt += s.burned_and_rebuilt > 0 && burnt ? 1 : 0;
        scarred_plague += s.plagues > 0 && abandoned ? 1 : 0;
        graves_seen += s.plagues > 0 && stones ? 1 : 0;
        mixed += s.conquerors >= 0 && builders.size() >= 2 ? 1 : 0;
    }
    std::cout << "  history standing: " << rebuilt << " burned and rebuilt (" << scarred_burnt << " with the burnt quarter left), " << flooded << " flooded, " << plagued << " plagued (" << scarred_plague
              << " with an empty quarter, " << graves_seen << " with graves), " << victors << " with a war to remember, " << taken << " taken (" << mixed << " with a conqueror's quarter)\n";
    if (!first_wrong.empty()) {
        std::cout << "  first wrong: " << first_wrong << "\n";
    }
    check(bad == 0, "10: what happened to a town is standing in it, and told");
    check(rebuilt >= 10 && flooded >= 3 && plagued >= 10 && victors >= 5 && taken >= 5, "10: history happened, of every kind");
    check(scarred_burnt * 3 >= rebuilt, "10: burnt quarters are left standing");
    check(scarred_plague * 3 >= plagued && graves_seen > 0, "10: empty quarters and graves after a plague");
    check(mixed * 3 >= taken, "10: a conqueror builds in its own way");
}

// In the law: what is built stands, and what is timber burns.
void check_buildings_in_law(const TerrainConfig& t, const WorldMap& map)
{
    const Settlement* town = nullptr;
    for (const Settlement& s : map.settlements()) {
        const Style& st = map.styles()[static_cast<std::size_t>(s.style)];
        if (s.fell < 0 && s.tier >= 1 && st.wall == Style::Stuff::Wood && (town == nullptr || std::hypot(s.x, s.y) < std::hypot(town->x, town->y))) {
            town = &s;
        }
    }
    check(town != nullptr, "4: a timber town");
    if (town == nullptr) {
        return;
    }
    std::optional<Playground> world = make_world(5);
    go_to(*world, town->x / t.chunk, town->y / t.chunk, (town->ground + 4) / t.chunk);
    const double wood = world->total("wood");
    const double stone = world->total("solid");
    for (int tick = 0; tick < 60 * 10; ++tick) {
        world->step(Control{});
    }
    check(wood > 100.0, "4: the window holds timber walls");
    check(std::abs(world->total("wood") - wood) < 0.5, "4: under the law a house stands as it was");
    check(std::abs(world->total("solid") - stone) < 0.5, "4: and its stone too");
    // The house nearest the middle of the town, so it is in the window.
    const Building* house = nullptr;
    for (const Building& b : town->buildings) {
        if (b.kind == Building::Kind::House && (house == nullptr || std::hypot(b.x0 - town->x, b.y0 - town->y) < std::hypot(house->x0 - town->x, house->y0 - town->y))) {
            house = &b;
        }
    }
    check(house != nullptr, "4: a house in it");
    const Pour* fire = nullptr;
    for (const Pour& s : world->pours()) {
        fire = s.name == "kindle" ? &s : fire;
    }
    const Vec3 o = world->origin();
    Vec3 wall{(house->x0 + 0.5) * t.cell - o.x, ((house->y0 + house->y1) * 0.5) * t.cell - o.y, (house->ground + 4) * t.cell - o.z};
    // A wall has timbers and what is between them: the ember goes on the nearest timber.
    {
        const FieldSet& f = world->fields();
        const GridShape& g = f.shape();
        const std::vector<float>& timber = f.values(f.find("wood"));
        double best = 1e30;
        const Vec3 aimed = wall;
        for (int z = 0; z < g.nz; ++z) {
            for (int y = 0; y < g.ny; ++y) {
                for (int x = 0; x < g.nx; ++x) {
                    if (timber[g.index(x, y, z)] > 0.5f) {
                        const Vec3 p{(x + 0.5) * g.cell, (y + 0.5) * g.cell, (z + 0.5) * g.cell};
                        if (length(p - aimed) < best) {
                            best = length(p - aimed);
                            wall = p;
                        }
                    }
                }
            }
        }
    }
    for (int k = 0; k < 3; ++k) {
        world->pour(*fire, wall, Vec3{1.0, 0.0, 0.0});
        for (int tick = 0; tick < 60; ++tick) {
            world->step(Control{});
        }
    }
    for (int tick = 0; tick < 60 * 6; ++tick) {
        world->step(Control{});
    }
    std::cout << "  a house set on fire: wood " << wood << " -> " << world->total("wood") << "\n";
    check(world->total("wood") < wood - 10.0, "4: a timber house burns, by the law alone");
}

void check_far_layers(const TerrainConfig& t, const WorldMap& map)
{
    // Heights: far columns stand about where the near ones do.
    for (const int factor : {8, 32, 128}) {
        double off = 0.0;
        int n = 0;
        for (int k = 0; k < 600; ++k) {
            const int x = -40000 + k * 137;
            const int y = 30000 - k * 89;
            const Column near = map.column(x, y);
            const FarColumn far = map.far_column(x, y, factor);
            off += std::abs(near.top - far.top);
            ++n;
        }
        std::cout << "  factor " << factor << ": ground off by " << off / n * t.cell << " m on average\n";
        check(off / n * t.cell < 3.0 + factor * t.cell * 0.2, "5: far ground stands where near ground does");
    }
    // Rivers: a thin one is gone from the sky, a big one stays.
    int thin_x = 0;
    int thin_y = 0;
    int big_x = 0;
    int big_y = 0;
    check(map.river_point(3.0, 4.0, thin_x, thin_y), "5: a thin river");
    const bool has_big = map.river_point(20.0, 40.0, big_x, big_y);
    check(has_big, "5: a big river");
    int thin_near = 0;
    int thin_far = 0;
    for (int dy = -40; dy <= 40; dy += 2) {
        for (int dx = -40; dx <= 40; dx += 2) {
            thin_near += map.column(thin_x + dx, thin_y + dy).water > t.sea_level ? 1 : 0;
            thin_far += map.far_column(thin_x + dx, thin_y + dy, 512).water > t.sea_level ? 1 : 0;
        }
    }
    check(thin_near > 0, "5: a thin river is there from near");
    int big_far = 0;
    for (int dy = -300; dy <= 300; dy += 32) {
        for (int dx = -300; dx <= 300; dx += 32) {
            big_far += map.far_column(big_x + dx, big_y + dy, 128).water > t.sea_level ? 1 : 0;
        }
    }
    std::cout << "  river cells: thin near " << thin_near << ", thin at factor 512 " << thin_far << ", big at factor 128 " << big_far << "\n";
    check(big_far > 0, "5: a big river still shows from high up");
    check(thin_far < thin_near / 4 + 1, "5: a thin one does not");
    // Buildings: roofs, then a patch.
    const Settlement* town = nullptr;
    for (const Settlement& s : map.settlements()) {
        if (s.fell < 0 && s.tier >= 1) {
            town = &s;
            break;
        }
    }
    check(town != nullptr, "5: a town");
    if (town != nullptr) {
        int roofs_16 = 0;
        int roofs_128 = 0;
        const int r = town->radius;
        for (int dy = -r; dy <= r; dy += 8) {
            for (int dx = -r; dx <= r; dx += 8) {
                roofs_16 += map.far_column(town->x + dx, town->y + dy, 16).roof_top > 0 ? 1 : 0;
                roofs_128 += map.far_column(town->x + dx, town->y + dy, 128).roof_top > 0 ? 1 : 0;
            }
        }
        check(roofs_16 > 0, "5: from far, houses are roofs");
        check(roofs_128 > roofs_16, "5: from the sky, a town is a patch of roofs");
    }
    // Forests: as canopy where no voxel can hold a tree.
    int canopy = 0;
    int forest = 0;
    for (int k = 0; k < 4000; ++k) {
        const int x = -50000 + (k * 773) % 100000;
        const int y = -50000 + (k * 379) % 100000;
        if (map.biome_at(x, y) == Biome::Forest) {
            ++forest;
            canopy += map.far_column(x, y, 64).crown_top > 0 ? 1 : 0;
        }
    }
    std::cout << "  forest samples " << forest << ", canopy at factor 64: " << canopy << "\n";
    check(forest > 10 && canopy > forest / 8, "5: a forest from far is canopy");
    // Coarse is cheap.
    const auto cost = [&](int factor) {
        const auto start = std::chrono::steady_clock::now();
        int sum = 0;
        for (int k = 0; k < 20000; ++k) {
            sum += map.far_column(-40000 + k * 3, 1000 + k, factor).top;
        }
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 20000.0;
        return sum == 0 ? 0.0 : us;
    };
    const double fine = cost(2);
    const double coarse = cost(256);
    std::cout << "  a far column costs " << fine << " us at factor 2, " << coarse << " us at factor 256\n";
    check(coarse < fine * 0.85, "5: the farther, the cheaper");
}

void check_bounds(const TerrainConfig& t, const WorldMap& map)
{
    int outside = 0;
    int tried = 0;
    for (const int chunks : {1, 4, 32, 256}) {
        for (int k = 0; k < 24; ++k) {
            const int cx = -3000 + k * 251;
            const int cy = 2000 - k * 173;
            int low = 0;
            int high = 0;
            map.height_bounds(cx * t.chunk, cy * t.chunk, (cx + chunks) * t.chunk, (cy + chunks) * t.chunk, low, high);
            for (int s = 0; s < 8; ++s) {
                const std::shared_ptr<const ColumnTile> tile = map.tile(cx + (s * 37) % chunks, cy + (s * 53) % chunks);
                ++tried;
                outside += tile->min_top < low || tile->feature_top > high ? 1 : 0;
            }
        }
    }
    // And where it matters most: the highest ground that can be found, and the deepest.
    for (int k = 0; k < 40000; ++k) {
        const int x = -WorldMap::kHalfCells + (k * 7919) % (2 * WorldMap::kHalfCells);
        const int y = -WorldMap::kHalfCells + (k * 104729) % (2 * WorldMap::kHalfCells);
        if (map.column(x, y, 64).top < t.sea_level + 300) {
            continue;
        }
        const int cx = static_cast<int>(std::floor(static_cast<double>(x) / t.chunk));
        const int cy = static_cast<int>(std::floor(static_cast<double>(y) / t.chunk));
        int low = 0;
        int high = 0;
        map.height_bounds(cx * t.chunk, cy * t.chunk, (cx + 1) * t.chunk, (cy + 1) * t.chunk, low, high);
        const std::shared_ptr<const ColumnTile> tile = map.tile(cx, cy);
        ++tried;
        outside += tile->min_top < low || tile->feature_top > high ? 1 : 0;
    }
    for (const Settlement& s : map.settlements()) {
        const int cx = s.x / t.chunk;
        const int cy = s.y / t.chunk;
        int low = 0;
        int high = 0;
        map.height_bounds(cx * t.chunk, cy * t.chunk, (cx + 1) * t.chunk, (cy + 1) * t.chunk, low, high);
        const std::shared_ptr<const ColumnTile> tile = map.tile(cx, cy);
        ++tried;
        outside += tile->min_top < low || tile->feature_top > high ? 1 : 0;
    }
    std::cout << "  height bounds: " << outside << " of " << tried << " chunks outside\n";
    check(outside == 0, "6: all matter lies within the bounds");
}

void check_vertical_window(const TerrainConfig& t)
{
    std::optional<Playground> world = make_world();
    for (int tick = 0; tick < 30; ++tick) {
        world->step(Control{});
    }
    const int start_cz = world->window_cz();
    const Vec3 home = world->player().position + world->origin();
    // Dropped from high up: the window goes with you, and you come down on the same ground.
    world->player().position = world->player().position + Vec3{0.0, 0.0, 45.0};
    double worst = 0.0;
    int highest = start_cz;
    for (int tick = 0; tick < 60 * 6; ++tick) {
        const Vec3 before = world->player().position + world->origin();
        world->step(Control{});
        const Vec3 after = world->player().position + world->origin();
        worst = std::max(worst, length(after - before));
        highest = std::max(highest, world->window_cz());
    }
    const Vec3 landed = world->player().position + world->origin();
    std::cout << "  dropped: window layer " << start_cz << " -> " << highest << " -> " << world->window_cz() << ", worst tick " << worst << " m, landed " << landed.z - home.z << " m from where it left\n";
    check(highest > start_cz, "7: the window went up with you");
    check(std::abs(world->window_cz() - start_cz) <= 1, "7: and came back down");
    check(worst < 1.0, "7: your world position never jumps when it moves (at 25 m/s a tick is 0.4 m)");
    check(std::abs(landed.z - home.z) < 1.0 && world->player().grounded, "7: you land on the ground you left it above");
    check(world->violations().empty(), "7: no contract broke");

    // What is written while the window moves up or down lands where it was written: poured on every tick of a fall
    // with the law on its own thread, it ends where the same pours end with the law on the caller's thread. Water
    // flows downhill after it lands, so "where you landed" is not the reference; the synchronous world is. A pour
    // taken up in the old lattice's cells would be a window layer (8 m) off.
    const auto pour_while_falling = [](bool async, int& layers_moved) {
        std::optional<Playground> w = make_world(3, async);
        for (int tick = 0; tick < 12; ++tick) {
            w->step(Control{});
        }
        const Pour* water = nullptr;
        for (const Pour& s : w->pours()) {
            water = s.field == "mana_water" || s.name == "water" ? &s : water;
        }
        const std::string field = water->field;
        w->player().position = w->player().position + Vec3{0.0, 0.0, 30.0};
        layers_moved = 0;
        int last_layer = w->window_cz();
        for (int tick = 0; tick < 60 * 3; ++tick) {
            w->pour(*water, w->player().position + Vec3{4.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0});  // beside you: a pour under you would lift you
            w->step(Control{});
            layers_moved += w->window_cz() != last_layer ? 1 : 0;
            last_layer = w->window_cz();
        }
        w->step(Control{});
        // The world height of the lowest wet cell and of the wet mass's centre.
        const FieldSet& f = w->fields();
        const GridShape& g = f.shape();
        const std::vector<float>& v = f.values(f.find(field));
        double lowest = 1e30, mass = 0.0, mz = 0.0;
        for (int z = 0; z < g.nz; ++z) {
            for (int y = 0; y < g.ny; ++y) {
                for (int x = 0; x < g.nx; ++x) {
                    const float m = v[g.index(x, y, z)];
                    if (m > 0.05f) {
                        const double wz = (z + 0.5) * g.cell + w->origin().z;
                        lowest = std::min(lowest, wz);
                        mass += m;
                        mz += m * wz;
                    }
                }
            }
        }
        return std::pair<double, double>{lowest, mass > 0.0 ? mz / mass : 0.0};
    };
    int moved_async = 0;
    int moved_sync = 0;
    const auto [low_async, centre_async] = pour_while_falling(true, moved_async);
    const auto [low_sync, centre_sync] = pour_while_falling(false, moved_sync);
    std::cout << "  poured while falling: " << moved_async << " layer moves; lowest water " << low_async - low_sync
              << " m and its centre " << centre_async - centre_sync << " m from the synchronous world\n";
    check(moved_async > 0, "7: the window moved while it was being written to");
    check(std::abs(low_async - low_sync) < 2.0 && std::abs(centre_async - centre_sync) < 2.0, "7: and what was poured lands where it does without the thread");
}

void check_far_life(const TerrainConfig& t, const WorldMap& map)
{
    // A fire left burning goes on burning: as much as one you stay by, about, and unlike a frozen world.
    const double chunk_m = t.chunk * t.cell;
    struct Fire {
        double before = 0.0;
        double after = 0.0;
        std::uint64_t turns = 0;
        std::size_t busy_start = 0;
        std::size_t busy_most = 0;
    };
    const auto burn = [&](bool leave, bool far_life) {
        std::vector<std::string> errors;
        std::vector<LawSource> laws;
        std::string spells;
        std::string path;
        Playground::read_default_law(laws, spells, path, errors);
        Playground::Config c;
        c.open_world = true;
        c.window_chunks = 3;
        c.field_every = 4;
        c.far_life = far_life;
        std::optional<Playground> w = Playground::create(laws, spells, errors, c);
        w->build_open_world();
        // A stack of timber of our own beside you, and fire at its foot.
        const GridShape g = w->fields().shape();
        const Vec3 you = w->player().position;
        const int ax = static_cast<int>(you.x / g.cell) + 6;
        const int ay = static_cast<int>(you.y / g.cell);
        const int az = static_cast<int>(you.z / g.cell);
        const Vec3 o = w->origin() + Vec3{(ax + 0.5) * g.cell, (ay + 0.5) * g.cell, 0.0};
        const int ccx = static_cast<int>(std::floor(o.x / chunk_m));
        const int ccy = static_cast<int>(std::floor(o.y / chunk_m));
        const int lowest = w->window_cz();
        {
            FieldSet& f = w->solver().fields();
            const int wood = f.find("wood");
            for (int z = az; z < az + 12; ++z) {
                for (int y = ay - 1; y <= ay + 1; ++y) {
                    for (int x = ax - 1; x <= ax + 1; ++x) {
                        f.values(wood)[g.index(x, y, z)] = 1.0f;
                    }
                }
            }
            w->solver().refresh_derived();
        }
        const Pour* fire = nullptr;
        for (const Pour& sp : w->pours()) {
            fire = sp.name == "kindle" ? &sp : fire;
        }
        for (int k = 0; k < 2; ++k) {
            w->pour(*fire, Vec3{(ax + 0.5) * g.cell, (ay + 0.5) * g.cell, (az + 2.5) * g.cell}, Vec3{1.0, 0.0, 0.0});
            for (int tick = 0; tick < 30; ++tick) {
                w->step(Control{});
            }
        }
        const auto wood_there = [&]() {
            double sum = 0.0;
            for (int cz = lowest; cz < lowest + 3; ++cz) {
                for (int cy = ccy - 1; cy <= ccy + 1; ++cy) {
                    for (int cx = ccx - 1; cx <= ccx + 1; ++cx) {
                        ChunkData data;
                        if (w->in_window(cx, cy, cz)) {
                            const FieldSet& f = w->fields();
                            const GridShape& lg = f.shape();
                            for (int z = 0; z < t.chunk; ++z) {
                                for (int y = 0; y < t.chunk; ++y) {
                                    for (int x = 0; x < t.chunk; ++x) {
                                        sum += f.values(f.find("wood"))[lg.index((cx - w->window_cx()) * t.chunk + x, (cy - w->window_cy()) * t.chunk + y, (cz - w->window_cz()) * t.chunk + z)];
                                    }
                                }
                            }
                            continue;
                        }
                        w->read_chunk(cx, cy, cz, data);
                        if (const std::vector<float>* v = data.find("wood")) {
                            for (const float value : *v) {
                                sum += value;
                            }
                        }
                    }
                }
            }
            return sum;
        };
        Fire out;
        if (leave) {
            go_to(*w, ccx + 40, ccy, (terrain_top(t, (ccx + 40) * t.chunk, ccy * t.chunk) + 1) / t.chunk);
            out.busy_start = w->far_busy();
        }
        out.before = wood_there();
        out.busy_most = out.busy_start;
        for (int tick = 0; tick < 60 * 40; ++tick) {
            w->step(Control{});
            out.busy_most = std::max(out.busy_most, w->far_busy());
        }
        out.after = wood_there();
        out.turns = w->far_runs();
        check(w->violations().empty(), "8: no contract broke in a fire far away");
        return out;
    };
    const Fire stayed = burn(false, true);
    const Fire left = burn(true, true);
    const Fire frozen = burn(true, false);
    std::cout << "  a fire, 40 s on: stayed by " << stayed.before << " -> " << stayed.after << "; left behind " << left.before << " -> " << left.after << " (" << left.turns
              << " turns, busy " << left.busy_start << ", most " << left.busy_most << "); frozen " << frozen.before << " -> " << frozen.after << "\n";
    check(left.turns > 10, "8: busy chunks get their turns");
    check(left.busy_start > 0, "8: a chunk left burning is still going");
    check(std::abs(frozen.after - frozen.before) < 0.01, "8: with far life off the world stands still (so the next claim means something)");
    check(stayed.before - stayed.after > 5.0, "8: a fire burns where you stand");
    check(left.before - left.after > 0.75 * (stayed.before - stayed.after), "8: and goes on burning with nobody there, about as much");
    check(left.before - left.after < 1.25 * (stayed.before - stayed.after), "8: and no more than it would");
    check(left.busy_most >= 2, "8: a fire keeps the chunks beside it awake");

    // Creatures walk on.
    std::optional<Playground> herd = make_world();
    const Vec3 at = herd->player().position;
    for (int k = 0; k < 4; ++k) {
        herd->spawn("grazer", at + Vec3{1.5 + k, 2.0, 1.0});
    }
    for (int tick = 0; tick < 60; ++tick) {
        herd->step(Control{});
    }
    std::vector<std::uint32_t> herd_ids;
    for (const Creature& c : herd->creatures()) {
        herd_ids.push_back(c.id);
    }
    const int home_cx = herd->window_cx() + 1;
    const int home_cy = herd->window_cy() + 1;
    const int home_cz = herd->window_cz() + 1;
    (void)home_cz;
    go_to(*herd, home_cx + 30, home_cy, (terrain_top(t, (home_cx + 30) * t.chunk, home_cy * t.chunk) + 1) / t.chunk);
    int stayed_behind = 0;
    for (const std::uint32_t id : herd_ids) {
        stayed_behind += std::any_of(herd->far_bodies().begin(), herd->far_bodies().end(), [&](const FarBody& b) { return b.id == id; }) ? 1 : 0;
    }
    check(!herd_ids.empty() && stayed_behind == static_cast<int>(herd_ids.size()), "8: the herd stays behind, as far bodies");
    std::map<std::uint32_t, std::pair<double, double>> start;
    for (const FarBody& b : herd->far_bodies()) {
        start[b.id] = {b.x, b.y};
    }
    for (int tick = 0; tick < 60 * 30; ++tick) {
        herd->step(Control{});
    }
    double walked = 0.0;
    bool wet = false;
    for (const FarBody& b : herd->far_bodies()) {
        const auto it = start.find(b.id);
        if (it != start.end()) {
            walked = std::max(walked, std::hypot(b.x - it->second.first, b.y - it->second.second));
        }
        wet = wet || map.column(static_cast<int>(std::floor(b.x / t.cell)), static_cast<int>(std::floor(b.y / t.cell))).water >= 0;
    }
    std::cout << "  far bodies: " << herd->far_bodies().size() << ", the farthest walked " << walked << " m in 30 s\n";
    check(walked > 2.0, "8: they walk on while nobody watches");
    check(!wet, "8: and not into the water");
    const std::uint32_t one = herd->far_bodies().empty() ? 0 : herd->far_bodies().front().id;
    const double back_x = herd->far_bodies().empty() ? 0.0 : herd->far_bodies().front().x;
    const double back_y = herd->far_bodies().empty() ? 0.0 : herd->far_bodies().front().y;
    const double back_z = herd->far_bodies().empty() ? 0.0 : herd->far_bodies().front().z;
    go_to(*herd, static_cast<int>(std::floor(back_x / chunk_m)), static_cast<int>(std::floor(back_y / chunk_m)), static_cast<int>(std::floor(back_z / chunk_m)));
    bool returned = false;
    for (const Creature& c : herd->creatures()) {
        returned = returned || c.id == one;
    }
    check(returned, "8: and are bodies again when you come");

    // The land's law.
    const double grazers = herd->map_total("grazers");
    check(grazers > 100.0, "8: the land holds herds");
    double most = 0.0;
    double least = 1e30;
    for (int tick = 0; tick < 60 * 180; tick += 4) {
        herd->step(Control{});
        const double g_now = herd->map_total("grazers");
        most = std::max(most, g_now);
        least = std::min(least, g_now);
    }
    check(most > least, "8: the land's law runs");
    std::cout << "  the land's law: grazers " << grazers << ", then between " << least << " and " << most << ", hunters " << herd->map_total("hunters") << "\n";
    check(least > grazers * 0.2 && most < grazers * 5.0, "8: herds on the land stay in bounds");
    check(herd->far_errors().empty(), "8: the land's law compiled");
    check(herd->violations().empty(), "8: no contract broke, near or far");

    // Determinism, with all of it going on, in step and on threads.
    auto play = [&](bool async, bool slow) {
        std::optional<Playground> w = make_world(3, async);
        const Pour* burn = nullptr;
        for (const Pour& s : w->pours()) {
            burn = s.name == "kindle" ? &s : burn;
        }
        w->pour(*burn, w->player().position + Vec3{3.0, 0.0, 0.5}, Vec3{1.0, 0.0, 0.0});
        for (int tick = 0; tick < 700; ++tick) {
            w->step(Control{.move_x = 1.0, .move_y = 0.2, .jump = tick % 40 == 0});
            if (slow && tick % 9 == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            }
        }
        return std::pair{w->digest(), w->far_runs()};
    };
    const auto a = play(false, false);
    const auto b = play(false, false);
    check(a.first == b.first, "8: the same inputs, the same world, far life included");
    const auto c = play(true, false);
    const auto d = play(true, true);
    std::cout << "  far turns in a walk: " << a.second << " in line, " << c.second << " on threads\n";
    check(c.first == d.first, "8: and on threads, however busy the machine is");
}

} // namespace

// With arguments, only the sections named (a mutation batch need not wait for the rest).
int main(int argc, char** argv)
{
    const auto wanted = [&](const char* name) {
        if (argc < 2) {
            return true;
        }
        for (int a = 1; a < argc; ++a) {
            if (std::string(argv[a]) == name) {
                return true;
            }
        }
        return false;
    };
    const TerrainConfig t;
    const auto start = std::chrono::steady_clock::now();
    const std::shared_ptr<const WorldMap> map = WorldMap::of(t);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cout << "  the map of " << WorldMap::kNodes * WorldMap::kNodeCells * t.cell / 1000.0 << " km takes " << ms << " ms to make\n";
    for (const std::string& e : map->errors()) {
        std::cerr << "  " << e << "\n";
    }
    check(ms < 20000.0, "1: a world is made in seconds");
    if (wanted("layers")) {
        check_layers(*map, t);
    }
    if (wanted("purity")) {
        check_purity(t, *map);
    }
    if (wanted("bounds")) {
        check_bounds(t, *map);
    }
    if (wanted("far")) {
        check_far_layers(t, *map);
    }
    if (wanted("buildings")) {
        check_buildings(t, *map);
    }
    if (wanted("buried")) {
        check_not_buried(t, *map);
    }
    if (wanted("towns")) {
        check_towns(t, *map);
    }
    if (wanted("water")) {
        check_water(t, *map);
    }
    if (wanted("law")) {
        check_buildings_in_law(t, *map);
    }
    if (wanted("window")) {
        check_vertical_window(t);
    }
    if (wanted("life")) {
        check_far_life(t, *map);
    }
    return fieldengine::gate::report("voxel_map");
}
