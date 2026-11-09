#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// THE WORLD AS LAYERS. Every layer is a pure function of the seed and of the layers above it, and each is known
// at every scale, so the same map serves a voxel at your feet and a pixel of the whole continent:
//
//   continent   plates, ranges, coasts                     nodes, 128 m apart
//   climate     wind, rain, temperature, biomes            nodes
//   water       drainage, rivers, lakes                    nodes, then polylines
//   history     cultures, foundings, wars, ruin            a list of what happened
//   plan        settlements, streets, roads                segments and lots
//   building    walls, rooms, stairs, roofs                a function of a cell, by style (assets/world/styles.world)
//   voxels      what a chunk holds                         Terrain.cpp, from all of the above
//
// A column can be asked for at any coarseness (`factor`, in cells): noise octaves finer than that are skipped,
// a river too small to show is left out and a big one is kept as wide as a voxel, a tree becomes canopy, a house
// becomes a roof and a town a patch of roofs. That is the level of detail of the layers themselves.
//
// The map is finite: continents and island chains, 262 km across, in a sea that goes on beyond it. So drainage and
// history can be worked out whole, once, when the world is made, and every chunk after that only reads the answers.
// Pure: same seed, same styles, same bits, on any thread.
//
// No two places are built alike. A people's way of building comes from the land it began in (what grows and what can
// be quarried there, how it rains, how cold it gets); each of its towns takes that and changes it by what happened
// to that town. All of it is the same text a person or a model could write (Style, style_text, parse_styles).
namespace fieldengine::field {

struct TerrainConfig;

enum class Biome : std::uint8_t { Ocean, Beach, Desert, Savanna, Grassland, Forest, Jungle, Taiga, Tundra, Snow, Swamp, Mountain };
const char* biome_name(Biome biome);

// HOW A PEOPLE BUILDS, AS TEXT: what a model can write and the engine can check. See assets/world/styles.world.
struct Style {
    enum class Roof : std::uint8_t { Gable, Hip, Flat, Dome, Spire };
    enum class Stuff : std::uint8_t { Stone, Wood, Leaf }; // stone stands, wood and thatch burn, by the law
    enum class Layout : std::uint8_t { Grid, Organic, Ring, Ribbon, Clusters };
    enum class Temple : std::uint8_t { Hall, Dome, Stepped, Spire };
    enum class Door : std::uint8_t { None, Plank, Double, Slide, Lift }; // none: a doorway, and that is that
    std::string name;
    std::vector<Biome> biomes;
    Stuff wall = Stuff::Stone;
    int wall_material = 0;
    Stuff upper = Stuff::Stone; // what the floors above the first are made of
    int upper_material = -1;    // -1: the same as the wall
    int trim_material = -1;     // corner posts; -1: none
    Roof roof = Roof::Gable;
    Stuff roof_stuff = Stuff::Wood;
    int roof_material = 0;
    int pitch = 1;        // cells up per cell in
    int eaves = 1;        // cells the roof stands out over the wall
    int floors_min = 1;
    int floors_max = 2;
    int storey = 6;       // cells, floor included
    int width_min = 12;
    int width_max = 18;
    int depth_min = 12;
    int depth_max = 18;
    int raise = 0;        // cells the floor stands above the ground
    bool piles = false;   // on posts, open underneath
    int base_material = 0;
    int window_every = 4; // 0: none
    int ring_material = -1; // a town wall, for the towns that have one
    bool ring_round = false;
    bool jetty = false;   // upper floors stand out over the street
    bool wing = false;    // low houses are L-shaped
    bool yard = false;    // houses have a walled yard behind
    bool chimney = false;
    bool round_towers = false;
    Layout layout = Layout::Grid;
    Temple temple = Temple::Hall;
    Door door = Door::Plank;
    int door_material = 13;  // timber
    bool door_stone = false;
    bool door_set = false;   // it said so; otherwise it comes from its name
};
// Errors name the line. A style that could not hold a person or a stair is refused here, not found by walking.
std::vector<Style> parse_styles(const std::string& text, std::vector<std::string>& errors);
// A style as the text parse_styles reads.
std::string style_text(const Style& style);
const char* door_name(Style::Door door);
int material_id(const std::string& name); // -1 if there is none
// In a few words, for a person.
std::string style_words(const Style& style);

struct Building {
    enum class Kind : std::uint8_t { House, Hall, Tower, Round, Well, Ring, Monument, Graves, Yard, Stepped };
    Kind kind = Kind::House;
    int x0 = 0; // footprint, world cells, inclusive
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;
    int ground = 0; // the top cell of the ground it stands on
    int floors = 1;
    int style = 0;
    int facing = 0; // the side the door is in: 0 +x, 1 -x, 2 +y, 3 -y
    int door = 0;   // the door's first cell along that side
    int part[2] = {-1, -1}; // inner walls, at these x
    int gap[2] = {0, 0};    // and the door in each, at this y
    int roof = -1;          // another roof than the style's (a Style::Roof)
    bool wing = false;      // L-shaped: the corner away from the door is cut off
    std::uint64_t seed = 0;
    float ruin = 0.0f;   // 0 whole, 1 hardly anything left
    bool burned = false;
    [[nodiscard]] int lowest() const { return ground - 2; }
    // The bounding box in the world, roof and steps included.
    void bounds(const std::vector<Style>& styles, int& bx0, int& by0, int& bx1, int& by1, int& top) const;
    // Can it be walked into? (Monuments, walls and ruins are not houses.)
    [[nodiscard]] bool has_rooms() const { return kind == Kind::House || kind == Kind::Hall || kind == Kind::Tower || kind == Kind::Round; }
};

// What a building makes of one cell.
struct BuildingCell {
    enum Kind : std::uint8_t { Outside, Air, Stone, Wood, Leaf, Rubble };
    Kind kind = Outside;
    int material = 0;
    bool door = false; // the leaf of a door: what a door does with it is the player's business
};
BuildingCell building_cell(const Building& building, const Style& style, int x, int y, int z);

struct Settlement {
    int id = 0;
    std::string name;
    int x = 0; // world cells, its centre
    int y = 0;
    int ground = 0;  // the top cell of its levelled ground
    int radius = 40; // cells
    int tier = 0;    // 0 hamlet, 1 village, 2 town, 3 city
    int culture = 0;
    int style = 0;        // its own: the people's way, changed by what happened here
    int conquerors = -1;  // the style of those who took it: they built a quarter of their own
    int founded = 0;  // the age
    int fell = -1;    // the age it was left in ruins, -1 if it stands
    bool burned = false;
    int people = 0;
    // What has happened to it, which is why it looks as it does.
    int burned_and_rebuilt = 0;
    int floods = 0;
    int plagues = 0;
    int wars_won = 0;
    bool lone = false; // a house or two, a family's own: no streets, no road comes here
    struct Street {
        int x0, y0, x1, y1;
        int half;        // half its width
        int circle = 0;  // >0: a ring of this radius around (x0, y0)
    };
    std::vector<Street> streets;
    std::vector<Building> buildings;
};

struct Culture {
    std::string name;
    int style = 0;
    int home = -1; // settlement it began from
};

// What the history layer remembers: what happened, to whom, in which age.
struct Chronicle {
    int age = 0;
    int settlement = -1;
    std::string text;
};

// One column of the world, ready to be voxels.
struct Column {
    int top = 0;          // the highest ground cell
    int water = -1;       // the top water cell, -1 if the column is dry
    std::uint8_t surface = 4;
    bool turf = false;
    bool clear = false;   // a street, a road or a lot: nothing grows here
    Biome biome = Biome::Grassland;
    float bend = 0.0f;    // the strata's slow wave
};

// WHAT GROWS, AS TEXT: assets/world/trees.world. The numbers a tree's growth is worked out from.
struct TreeSpecies {
    enum class Form : std::uint8_t { Spreading, Conifer, Palm, Column, Bush, Snag };
    std::string name;
    std::vector<Biome> biomes;
    Form form = Form::Spreading;
    double height_min = 10.0; // metres
    double height_max = 20.0;
    double slender = 20.0;    // d = (H / slender)^(3/2): how close to buckling under itself it lives
    int limbs = 3;            // at each fork
    int levels = 2;           // of forks
    double spread = 40.0;     // degrees off the line of what it forks from
    double rise = 0.1;        // turned toward the light, each step
    double droop = 0.1;       // bent by its own weight: times (l / r)^2
    double leaf = 2.5;        // cells: how far leaves stand around a branch tip
    double flat = 1.0;        // of a crown: how high against how wide
    int bark = 13;
    int colour = 20;
    double share = 1.0;
};
// What grows on the ground.
struct FloraKind {
    std::string name;
    std::vector<Biome> biomes;
    int height = 1;
    int colour = 22;
    double share = 1.0;
};
struct LandGrowth {
    double big = 0.0;   // of the stands of big trees, how many have one
    double small = 0.0;
    double flora = 0.0; // of the cells
};
struct TreeVoxel {
    std::int8_t dx = 0; // from the foot of the tree
    std::int8_t dy = 0;
    std::uint16_t dz = 0;
    std::uint8_t material = 0;
    std::uint8_t wood = 0;
};

struct Tree {
    enum Kind : std::uint8_t { Oak, Pine, Cactus, Shrub };
    Kind kind = Oak;
    int species = 0;
    int x = 0;
    int y = 0;
    int ground = 0;
    int trunk = 0;       // cells: its height
    double crown = 0.0;  // cells: leaves stand this far over the top of what grew
    double spread = 0.0; // cells: how far its crown reaches out, about
    std::shared_ptr<const std::vector<TreeVoxel>> shape; // what grew (tile() fills it; far away it is not needed)
    [[nodiscard]] int top() const { return ground + trunk + static_cast<int>(crown) + 2; }
};
// Worked out from the species and a seed alone. Sorted, no two voxels in one place, wood over leaves.
std::vector<TreeVoxel> grow_tree(const TreeSpecies& species, std::uint64_t seed, int height_cells, double cell_metres);

// One chunk's columns and everything standing on them, kept while chunks of it are asked for.
struct ColumnTile {
    std::array<Column, 256> columns;
    int min_top = 0;
    int max_top = 0;     // ground and water
    int feature_top = 0; // and whatever stands on them
    std::vector<Tree> trees;                 // those that reach into this chunk
    std::vector<const Building*> buildings;  // those that reach into this chunk
};

// A column from far away.
struct FarColumn {
    int top = 0;
    int water = -1;
    int surface = 4;
    bool turf = false;
    int crown_top = 0;  // leaves over it, 0 if none
    int roof_top = 0;   // a roof over it, 0 if none
    int roof_material = 0;
};

class WorldMap {
public:
    static constexpr int kNodes = 2048;
    static constexpr int kNodeCells = 256;
    static constexpr int kHalfCells = kNodes * kNodeCells / 2;

    // The one map of this configuration: made on first use, shared after that.
    static std::shared_ptr<const WorldMap> of(const TerrainConfig& config);
    // What was wrong with the styles, if anything (the world then has the plainest one).
    [[nodiscard]] const std::vector<std::string>& errors() const { return errors_; }

    [[nodiscard]] Column column(int wx, int wy, int factor = 1) const;
    [[nodiscard]] FarColumn far_column(int wx, int wy, int factor) const;
    [[nodiscard]] std::shared_ptr<const ColumnTile> tile(int cx, int cy) const;
    // Every cell of the box [x0, x1) * [y0, y1) lies between these (a little conservative, and quick at any size).
    void height_bounds(int x0, int y0, int x1, int y1, int& lo, int& hi) const;

    [[nodiscard]] Biome biome_at(int wx, int wy, int factor = 1) const;
    [[nodiscard]] double temperature_at(int wx, int wy) const; // Celsius
    [[nodiscard]] double rain_at(int wx, int wy) const;        // 0..1
    [[nodiscard]] const Settlement* settlement_at(int wx, int wy) const; // whose ground this is
    [[nodiscard]] const Settlement* nearest_settlement(int wx, int wy) const;
    [[nodiscard]] const Building* building_over(const Settlement& s, int wx, int wy) const;
    [[nodiscard]] double capacity_at_node(int nx, int ny) const; // how much a node can feed, 0..1

    [[nodiscard]] const std::vector<Settlement>& settlements() const { return settlements_; }
    [[nodiscard]] const std::vector<Culture>& cultures() const { return cultures_; }
    [[nodiscard]] const std::vector<Chronicle>& chronicle() const { return chronicle_; }
    [[nodiscard]] const std::vector<Style>& styles() const { return styles_; }
    [[nodiscard]] const std::vector<TreeSpecies>& species() const { return species_; }
    [[nodiscard]] const std::vector<FloraKind>& flora_kinds() const { return flora_; }
    [[nodiscard]] const LandGrowth& growth_of(Biome biome) const { return growth_[static_cast<std::size_t>(biome)]; }
    // What grows on the ground at this column, if anything.
    [[nodiscard]] const FloraKind* flora_at(int wx, int wy, const Column& column) const;
    [[nodiscard]] int rivers() const { return static_cast<int>(segments_.size()); }
    [[nodiscard]] int roads() const { return static_cast<int>(roads_.size()); }
    // Dry turf a little outside the nearest living village to the middle of the map.
    void home(int& wx, int& wy) const;
    // Where a river runs: a point of the channel of a river this wide, in cells (false if there is none).
    [[nodiscard]] bool river_point(double min_width, double max_width, int& wx, int& wy) const;
    // Does every reach of every river run down to the next?
    [[nodiscard]] bool rivers_run_down() const;

    WorldMap(std::uint64_t seed, int sea_level, int height, int chunk, const std::string& styles);

private:
    struct Base {
        double ground = 0.0;
        int water = -1;
        bool clear = false;
        bool street = false;
        bool bed = false;
    };
    struct Segment { // a stretch of river or road
        float ax, ay, bx, by;
        float level_a, level_b;
        float width;
    };
    struct Index { // per node, what lies near it
        std::vector<std::uint32_t> first;
        std::vector<std::uint32_t> items;
        [[nodiscard]] std::size_t begin(int node) const { return first[static_cast<std::size_t>(node)]; }
        [[nodiscard]] std::size_t end(int node) const { return first[static_cast<std::size_t>(node) + 1]; }
    };

    void make_continent();
    void make_climate();
    void make_water();
    void classify_biomes();
    void make_history();
    void make_roads();
    void plan_settlement(Settlement& s);
    void make_local_styles();
    [[nodiscard]] Style architect(std::uint64_t seed, Biome land, double temperature, double rain, const std::string& name) const;
    void make_bounds();
    Index index_boxes(const std::vector<std::array<int, 4>>& boxes) const;

    [[nodiscard]] Base base(int wx, int wy, int factor) const;
    [[nodiscard]] Column finish(const Base& b, int wx, int wy, int factor = 1) const;
    [[nodiscard]] Column contained(int wx, int wy) const;
    [[nodiscard]] bool tree_at(int gx, int gy, Tree& out, int factor) const;
    [[nodiscard]] std::shared_ptr<const std::vector<TreeVoxel>> grown(const Tree& tree, int gx, int gy) const;
    void read_growth(const std::string& text);
    [[nodiscard]] double smooth_at(const std::vector<float>& values, double wx, double wy) const;
    [[nodiscard]] int node_of(int wx, int wy) const; // -1 beyond the map

    std::uint64_t seed_;
    int sea_;
    int height_;
    int chunk_;
    std::vector<std::string> errors_;
    std::vector<Style> styles_;
    std::vector<TreeSpecies> species_;
    std::vector<FloraKind> flora_;
    std::array<LandGrowth, 12> growth_{};
    std::array<std::vector<int>, 12> big_of_;   // which species grow big, and small, on each land
    std::array<std::vector<int>, 12> small_of_;
    std::array<std::vector<int>, 12> flora_of_;
    mutable std::mutex grown_mutex_;
    mutable std::unordered_map<std::int64_t, std::shared_ptr<const std::vector<TreeVoxel>>> grown_;

    std::vector<float> elevation_; // cells
    std::vector<float> filled_;    // with every pit filled to its spill
    std::vector<float> rough_;     // how much small relief a node has, cells
    std::vector<float> temperature_;
    std::vector<float> rain_;
    std::vector<float> flow_;      // rain gathered from upstream
    std::vector<float> level_;     // the water surface of a river or lake here
    std::vector<int> down_;        // where the water goes, -1 in the sea
    std::vector<int> order_;       // sea first, then upstream
    std::vector<char> lake_;
    std::vector<Biome> biome_;
    std::vector<float> site_;      // how good a place to live, for the history layer

    std::vector<Segment> segments_;
    Index river_near_;
    std::vector<Segment> roads_;
    Index road_near_;
    std::vector<Settlement> settlements_;
    Index settlement_near_;
    std::vector<Culture> cultures_;
    std::vector<Chronicle> chronicle_;

    // min / max of the relief over 1, 2, 4... nodes, for height_bounds.
    std::vector<std::vector<float>> low_;
    std::vector<std::vector<float>> high_;

    mutable std::mutex tiles_mutex_;
    mutable std::unordered_map<std::int64_t, std::shared_ptr<const ColumnTile>> tiles_;
};

} // namespace fieldengine::field
