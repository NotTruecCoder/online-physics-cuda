#pragma once

#include "field/Items.hpp"
#include "field/Looks.hpp"
#include "field/Melee.hpp"
#include "field/Body.hpp"
#include "field/Law.hpp"
#include "field/Ray.hpp"
#include "field/Society.hpp"
#include "field/Solver.hpp"
#include "field/Terrain.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

// THE VOXEL WORLD. Nothing but fields on a lattice, the law that couples them (assets/law/*.law), and
// bodies that live in it. What a tool pours is a shape of mana, water or stone poured into voxels; what happens next — stone
// melting, breaking, falling, setting, water freezing, turf burning — is the law's business, not this
// file's. Bodies read the same fields: they stand on `occ` (through surface_distance), burn by `harm`,
// are thrown by `push_*`. Creatures are bodies whose wants and appetites are law too.
//
// Two shapes of world: the fixed proving ground (one lattice, for gates and experiments), and the open world — a
// generated continent (WorldMap.hpp) of which a box of chunks around you is simulated. The box follows you, up and
// down too; a chunk leaving it keeps exactly what the law did to it (only what differs from the generator is
// stored). Beyond the box life goes on, more simply, at three grains (FarLife.cpp):
//   a chunk still busy (hot, burning, molten, full of mana) runs the same law on its own, on the CPU, as time allows;
//   a creature walks over the map's columns, once a second;
//   the land as a whole is one more law (assets/law/map.law) on the map's nodes: how much life each can feed.
//
// Deterministic: bodies tick at a fixed 60 Hz, the fields every `field_every` body ticks, the window
// moves on your position alone; no clocks, no RNG, one evaluation order.
namespace fieldengine::field {

struct MeshVertex;
class WorldModel;

// A SHAPE OF SOMETHING POURED INTO THE FIELDS: mana, water, stone, heat, whatever field the law has, in a ball, a wall, a beam,
// a ring, a pillar or a cut. What it does after that is the law's (assets/law): a ball of fire mana is not a fireball until the
// law makes turf catch from it. The named ones are text (assets/law/pours.law).
struct Pour {
    enum class Shape { Ball, Wall, Beam, Ring, Pillar, Cut };
    std::string name;
    std::string about;       // a line for the tooltip
    std::string field;       // the field it pours into
    double amount = 1.0;
    std::string also;        // and a second field, in the same shape
    double also_amount = 0.0;
    Shape shape = Shape::Ball;
    double radius = 1.0;
    double width = 4.0;      // of a wall
    double height = 2.5;     // of a wall, a ring, a pillar, a cut
    double thickness = 1.0;  // of a wall, a ring, a cut
    double range = 20.0;     // metres: a beam's length, a cut's
    bool flat = false;       // a beam, level and whole (a causeway), not thickest along its line
    float colour[3] = {1.0f, 1.0f, 1.0f};
};

// Parses pours.law; errors name the line.
std::vector<Pour> parse_pours(const std::string& text, std::vector<std::string>& errors);

struct PlayBody {
    std::string name;
    Vec3 position{}; // metres, lattice coordinates (world = origin() + position)
    Vec3 velocity{};
    double radius = 0.4;
    double hp = 100.0;
    double hp_max = 100.0;
    bool grounded = false;
    bool controlled = false; // walks (gives way to bodies that stand), not standing still
    bool parked = false;     // outside the simulated window: frozen, not drawn
    double surge = 0.0;      // seconds of a dash left: it holds its velocity, nothing else moves it
    double stagger = 0.0;    // seconds since it was thrown: its feet do not catch it yet
    Swing swing;             // a blow under way
    double guard = -1.0;     // seconds its guard has been up; below 0, it is down
    double guard_yaw = 0.0;  // where the guard faces
    double armour = 0.0;     // of a blow gets through 100 / (100 + this)
    double strength = 1.0;   // what its own blows are multiplied by
    double quick = 1.0;      // how fast its blows go: what it wears and how strong it is
    std::uint32_t id = 0;    // of a creature; 0, not one
    std::uint32_t foe = 0;   // whom its blows are for: a creature's id; 0, you
    std::uint32_t struck_by = 0; // who hit it last, if that was not you
};

// A body the law steers: its def in the law, its own state, where it ambles.
struct Creature {
    int def = -1;
    std::uint32_t id = 0;
    PlayBody body;
    std::vector<double> locals; // CreatureDef::locals
    double heading = 0.0;       // radians
    double brood = 0.0;         // adds up to the next newborn
    Vec3 want{};                // the direction it chose last tick (unit, horizontal; for drawing)
    Vec3 planned{};             // what its world model chose (a mind with a plan line), |planned| <= 1
    bool planning = false;
    double fight_wait = 0.0;    // seconds before it swings again
    // Someone of a town (Society.hpp); town -1, a beast. Its mind is the law's lines; these are what it does about them.
    enum class Doing : std::uint8_t { Working, Market, Home, Patrol, Talking, Looking, Fleeing, Fighting };
    int town = -1;
    Person who;
    Doing doing = Doing::Working;
    bool hostile = false;
    bool sees_you = false;
    bool began_it = false;    // it came for you before you touched it
    double face = 0.0;        // radians
    double opinion = 0.0;     // of you, as it was last worked out
    double last_hp = 0.0;
    double deed_wait = 0.0;   // seconds before another blow counts as another story
    double bark_wait = 0.0;   // seconds before it says anything again
    double look_wait = 0.0;   // seconds it stops to look at you
    Vec3 goal{};
    int turning = 1;          // which way it goes round what is in its way
    Vec3 last_place{};        // where it was at the last look
    int pressed = 0;          // looks in a row it wanted to go somewhere and did not
    double detour_left = 0.0; // seconds it goes its own way to get out of a corner
    double detour_angle = 0.0;
    int walled = 0;           // ticks in a row inside something solid
    Looks looks;
    Inventory bag;            // what it has, and what it is wearing
    std::string arm;          // the weapon it fights as: what is in its hand
    // A fight that is not with you (Chat.cpp, Melee.cpp).
    std::uint32_t foe = 0;    // a creature it has gone for; 0, nobody but perhaps you
    enum class Plan : std::uint8_t { Close, Circle, Back, Strike, Guard };
    Plan plan = Plan::Close;
    int plan_move = -1;
    double plan_left = 0.0;   // seconds before it thinks again
    double guard_wait = 0.0;  // seconds before it can get its guard up again: nobody hides behind one for ever
    double riled_left = 0.0;  // of a beast that only fights when hurt
    double prey_hold = 0.0;   // seconds it stays on the one it was set on
    double flee_left = 0.0;   // seconds it runs from `flee_from`, whoever is there
    Vec3 flee_from{};
    int caravan = -1;         // of Society's: it is on the road
    std::uint64_t stolen_from = 0; // pid + 1 of whom it robbed
    int stolen_item = -1;
    bool brawling = false;    // with its fists, and not to the death
    std::string arm_before;
};

// A line of the chat: what you said, what was said to you, what happened here, what the world tells you.
struct ChatLine {
    enum class Kind : std::uint8_t { System, You, Person, Local };
    Kind kind = Kind::System;
    std::string who;
    std::string text;
    std::uint64_t tick = 0;
};

// A piece of what stood, in one piece: a rigid body made of voxels (Falling.cpp).
struct FallingCell {
    float x = 0.0f; // metres, about the centre of mass, as it was when it stood
    float y = 0.0f;
    float z = 0.0f;
    std::uint32_t abgr = 0xffffffffu;
    std::uint8_t faces = 63;
    bool leaf = false;
    short material = 0;
    // What it is again when it comes to rest: 0 wood (or a leaf, by `leaf`), 1 stone, 2 ice.
    std::uint8_t matter = 0;
};

// What a thrown ball is made of (Playground::throw_ball).
enum class BallKind : std::uint8_t { Stone, Wood, Ice };

struct FallingPiece {
    std::vector<FallingCell> cells;
    std::vector<int> feel; // the cells that feel for the ground
    Vec3 at{};             // world metres of the centre of mass
    Vec3 velocity{};
    Vec3 spin{};           // radians a second, world
    Vec3 tip{};            // what the blow that cut it goes on turning it by, radians a second a second, while it finds its feet
    double tip_left = 0.0;
    double turned[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    double inverse[3][3] = {}; // of its moment of inertia, at rest
    double mass = 1.0;
    double age = 0.0;
    double quiet = 0.0;
    double hit_wait = 0.0;
    bool landed = false;
    double bounce = 0.08;  // restitution of what is not a leaf
    double crush = 4.0;    // HP a body loses per m/s of a hit
    // Which cells of its own lattice it fills: what bodies run into.
    int grid[3] = {0, 0, 0};
    Vec3 grid_from{};
    std::vector<char> filled;
};

// Something said over someone's head.
struct Bark {
    std::uint32_t id = 0;
    std::string text;
    double left = 3.0;
};

// What the body the window follows is asked to do this tick (the sandbox's observer asks nothing).
struct Control {
    double move_x = 0.0; // world axes, metres
    double move_y = 0.0;
    bool jump = false;
    Vec3 aim{};          // lattice point being aimed at (metres)
};

// What just happened, for whoever draws it. Nothing in the simulation reads these back.
struct PlayEvent {
    enum class Kind { Impact, Jump, Landing, Swing, MeleeHit, Parried, Blocked, Clang, Shock };
    Kind kind = Kind::Impact;
    int pour = -1;     // Impact: which named pour
    Vec3 at{};         // world metres
    Vec3 to{};         // world metres: where it was aimed, or the other end
    double power = 0.0; // Impact: its radius. Landing: the speed it came down at.
    int hand = 0;       // MeleeHit: 1 for the last blow of a string
    bool yours = true;  // MeleeHit: it has to do with you
};

// One field step's result: the state, which fields it changed, and the window it is of.
struct FieldStep {
    const FieldSet& fields;
    const std::vector<char>& changed;
    std::uint64_t step = 0;
    int window_cx = 0;
    int window_cy = 0;
    int window_cz = 0;
};

// A creature beyond the window: where it is and what it was.
struct FarBody {
    std::string kind;
    double x = 0.0; // world metres
    double y = 0.0;
    double z = 0.0;
    double heading = 0.0;
    double hp = 0.0;
    double brood = 0.0;
    std::uint32_t id = 0;
    std::vector<std::pair<std::string, double>> locals; // by name, so a changed law keeps what it can
};

class Playground {
public:
    struct Config {
        GridShape shape{.nx = 48, .ny = 48, .nz = 20, .cell = 0.5};
        int field_every = 3;      // field step every N body ticks (60 Hz / 3 = 20 Hz)
        double walk_speed = 7.0;  // m/s
        double jump_speed = 10.5; // m/s
        double gravity = 20.0;    // more than the Earth's: you go up fast and come down fast
        double fall_limit = 45.0; // m/s
        double air_control = 28.0; // m/s^2: you can add to your speed in the air, and steering never takes it away
        // The open world: `shape` becomes window_chunks x window_chunks x window_layers chunks of the terrain.
        bool open_world = false;
        TerrainConfig terrain{};
        int window_chunks = 3;
        int window_layers = 3;
        bool far_life = true;     // what goes on beyond the window; off, the rest of the world is frozen
        // THE LAW ON ITS OWN THREAD. A field step starts at one fixed tick and is taken up at the next:
        // bodies meanwhile read the last finished state, and whatever they write into the fields waits
        // for that same tick. When it is taken up never depends on how long it took (a slow step makes
        // the frame wait, not the world differ), so this is as deterministic as stepping in line, one
        // step of latency behind it. A frame never pays for the solver, so the window can be larger.
        bool async_fields = false;
        // Not in step with the frame: a step not finished when it is due is left to finish, the frame goes on with what is
        // there, and the law catches up by up to `field_most_owed` seconds (after that, its time runs slower than yours).
        // No longer deterministic, so for playing and not for a gate.
        bool field_never_waits = false;
        double field_most_owed = 0.25;
        int law_checks_every = 1; // the contracts, on the GPU, are looked at in one step of so many
        // The law as compute shaders on the graphics card (GpuSolver.hpp); on the CPU if there is none.
        bool gpu_fields = false;
        double parry_window = 0.18;   // seconds: a guard raised this late turns the blow back on who struck
        double parry_stagger = 0.7;
        double guard_keeps = 0.2;     // of a blow, through a guard
        int max_creatures = 64;   // alive in the window at once: a law that breeds cannot swamp the frame
        // The people of the towns (Society.hpp). Off, there are none, and what the world does is as it was.
        bool people = false;
        int people_most = 60;     // bodies of theirs in the window at once; the rest are numbers
        double out_share = 0.08;  // of a town's people, how many are out of doors at a time (bodies in the window, proxies beyond)
        // What comes away comes away in one piece, as a rigid body. Off, the law's `held` lines bring it down as rubble.
        bool falling = false;
        // Things happen among the people around you on their own, every `incident_every` seconds (people.law).
        bool local_life = false;
        // NOBODY PLAYS. The body the window follows is only a point of interest: it does not move by itself, nothing sees it,
        // hits it or smells it (no scent), and it cannot be hurt. Whoever sets its position moves the window.
        bool observer = false;
    };

    static constexpr double kTick = 1.0 / 60.0;

    // Compiles the laws and the named pours; nullopt (with errors) if any of it is broken.
    static std::optional<Playground> create(const std::vector<LawSource>& laws, const std::string& pours_text,
        std::vector<std::string>& errors, Config config);
    static std::optional<Playground> create(const std::vector<LawSource>& laws, const std::string& pours_text,
        std::vector<std::string>& errors)
    {
        return create(laws, pours_text, errors, Config{});
    }
    // Reads assets/law/{world,magic,life,people}.law and pours.law from the first root that has them (life and people
    // are optional). The paths it read are kept (law_paths()) so the inspector can write edits back.
    static bool read_default_law(std::vector<LawSource>& laws, std::string& pours_text, std::string& pours_path,
        std::vector<std::string>& errors);
    // The proving ground (open_world = false) or the open world, built and ready.
    static std::optional<Playground> load_default(std::vector<std::string>& errors, bool open_world = false);

    // The proving ground: a floor, a long wall, a pillar, steps, a basin of water; you and a target.
    void build_arena();
    // A round pit with a wall and two pillars, you in it, and nobody else yet.
    void build_ring();
    // Someone comes into it, at the wall, spoiling for it. False if the law has no such kind.
    bool ring_spawn(const std::string& kind);
    int deaths() const { return deaths_; }
    // The open world around world-cell (0, 0): the window loaded, you on dry ground, a target nearby.
    void build_open_world();

    // THE LAW, CHANGED WHILE THE WORLD RUNS. Compiles `laws` + `pours_text`; if anything is broken the
    // running law stays and `errors` says why. Otherwise every field keeps its values by name (new
    // fields start at their init, removed ones are dropped), params take the new text's values,
    // creatures keep their bodies and the states the new law still has.
    bool reload_law(const std::vector<LawSource>& laws, const std::string& pours_text, std::vector<std::string>& errors);
    const std::vector<LawSource>& law_sources() const { return sources_; }
    const std::string& pours_text() const { return pours_text_; }

    void step(const Control& control);

    // What the cursor points at: the first surface along the ray (metres), if any.
    std::optional<Vec3> pick(Vec3 origin, Vec3 direction) const;
    // BEYOND THE WINDOW the law is not running, but the world is there. The top of what stands at a world column (ground, roof
    // or crown, as the map makes it), in world metres: what a piece that leaves the window lands on. What the law changed out
    // there is not looked at (that is in the store; when the window comes back it is what is there). Proving ground: none.
    [[nodiscard]] double ground_at(double wx, double wy) const;
    // The same without what stands on it: the top of the ground alone.
    [[nodiscard]] double terrain_at(double wx, double wy) const;
    // The first surface along a ray in world metres: the window's own (the law's), and past it the map's.
    [[nodiscard]] std::optional<Vec3> pick_world(Vec3 from, Vec3 direction, double max_length = 4000.0) const;

    // Mutable access waits for a running field step and assumes the fields are about to be edited.
    Solver& solver();
    const Solver& solver() const { return solver_; }
    const Law& law() const { return solver_.law(); }
    // What bodies, rays and the picture read.
    const FieldSet& fields() const { return config_.async_fields ? front_ : solver_.fields(); }
    // A param change, taken up at the next field step.
    void set_param(const std::string& name, double value);
    // Told of the state every field step leaves (and once now), on the thread that ran the step: what the
    // picture needs from it is worked out there, off the frame. Nothing it does reaches the simulation.
    using FieldListener = std::function<void(const FieldStep& step)>;
    void set_field_listener(FieldListener listener);
    Surface surface() const { return Surface{.sdf = sdf_, .solid = occ_}; }
    const std::vector<Pour>& pours() const { return pours_; }

    PlayBody& player() { return bodies_[0]; }
    const PlayBody& player() const { return bodies_[0]; }
    PlayBody& target() { return bodies_[1]; }
    const PlayBody& target() const { return bodies_[1]; }
    const std::vector<PlayBody>& bodies() const { return bodies_; }
    const std::vector<Creature>& creatures() const { return creatures_; }
    // Blows (Melee.cpp): the moves people and beasts fight with come from the book given here (assets/world/moves.world).
    void set_melee(MeleeBook book);
    // A blow begins in any body: the move (by its place in the book) and the way it faces. It lands as the move says, on the ticks
    // that follow (Melee.cpp), whoever's body it is.
    void swing(PlayBody& body, int move, Vec3 way);
    [[nodiscard]] const MeleeBook& melee() const { return melee_; }
    [[nodiscard]] double phase_of(const PlayBody& body) const;
    int pour_slot(const std::string& name) const; // -1 if pours.law has no such pour
    // Since the last clear_events(): at most the last few hundred.
    const std::vector<PlayEvent>& events() const { return events_; }
    void clear_events() { events_.clear(); }
    std::uint64_t tick() const { return tick_; }
    const Config& config() const { return config_; }
    // Wall-clock cost of the last field step, for the overlay. Nothing in the simulation reads it.
    double field_ms() const { return field_ms_; }
    // The world models of the kinds that plan (model 1), one per kind; empty where a kind seeks.
    [[nodiscard]] const std::vector<std::shared_ptr<WorldModel>>& minds() const { return minds_; }
    // What the last field step cost, term by term (a copy: the step runs on its own thread).
    [[nodiscard]] StepProfile field_profile() const
    {
        const std::lock_guard<std::mutex> lock(*profile_mutex_);
        return field_profile_;
    }
    // Learned schemes of the law (`learned d ...`) on or off; they run under guard either way.
    void set_learned(bool on) { solver().set_learned(on); }
    [[nodiscard]] bool learned() const { return solver_.learned(); }
    const std::vector<std::string>& log() const { return log_; }
    // Every distinct contract the law broke since the start (first occurrence of each term+field).
    const std::vector<Violation>& violations() const { return violations_; }

    // Pours into the fields at `where` (lattice metres; the foot of a wall or a pillar), heading `dir`. A beam goes from
    // `where` along `dir`, as long as `dir` is.
    void pour(const Pour& pour, Vec3 where, Vec3 dir);
    // A named pour, and an Impact event for whoever draws it.
    void pour_at(int slot, Vec3 where, Vec3 dir);
    // Puts a creature of the law's kind `kind` at a lattice point; false if the law has no such kind.
    bool spawn(const std::string& kind, Vec3 where);

    // Bodies + creatures + every field + the window, bit for bit.
    std::uint64_t digest() const;

    // Sum of a field over the lattice (for gates and the HUD).
    double total(const std::string& field) const;

    // ---- the people beyond the live window (People.cpp) ------------------------------------------------
    // Inside the window everyone out of doors is a body the law moves. Beyond it they are where their day has them, walking
    // between work, the market and home at a walking pace, with no body: what is drawn of a town further off. Who is out of
    // doors today is a hash of who they are and the day, the same for both, so a person crossing into the window becomes a
    // body where they were and a body leaving it goes on as the same person.
    struct PersonProxy {
        std::uint64_t pid = 0;
        int town = -1;
        int index = 0;
        Vec3 at{};          // world metres, feet
        double yaw = 0.0;   // where it is going
        double speed = 0.0; // m/s
        Creature::Doing doing = Creature::Doing::Working;
    };
    // The people out of doors within `radius` of (cx, cy), world metres, the nearest towns first, at most `most` of them.
    void people_beyond(double cx, double cy, double radius, std::size_t most, std::vector<PersonProxy>& out) const;
    // Where its day has someone at `seconds` of world time: false while indoors.
    bool day_place(const Person& who, int town, double seconds, Vec3& at, double& yaw, double& speed, Creature::Doing& doing) const;
    [[nodiscard]] bool out_today(int town, int index, double seconds) const;
    [[nodiscard]] double world_seconds() const { return static_cast<double>(tick_) * kTick + day_shift_; }
    // How someone looks and what they wear and hold, from who they are and the weather of their town.
    void outfit(const Person& who, int town, Looks& looks, Inventory& worn, std::string& arm) const;

    // ---- the people (People.cpp) ---------------------------------------------------------------------
    Society* society() { return society_.get(); }
    const Society* society() const { return society_.get(); }
    const std::vector<Bark>& barks() const { return barks_; }  // what is being said over people's heads
    const Creature* creature_with(std::uint32_t id) const;
    const char* doing_words(const Creature& c) const;

    // ---- what happens among the people (Chat.cpp): a log of it, and what can be made to happen ----------------------------
    const std::vector<ChatLine>& chat() const { return chat_; }
    void chat_line(ChatLine::Kind kind, const std::string& who, const std::string& text);
    bool make_incident(const std::string& kind); // something happens here, now: a brawl, a beast; "" for what this place is like
    double day_phase() const;                    // 0 to 1 of the day
    // ---- what people wear and hold (Goods.cpp), what falls (Falling.cpp) ----------------------------------------------------
    const ItemBook& items() const { return items_; }
    const std::vector<FallingPiece>& pieces() const { return pieces_; }
    void mesh_pieces(std::vector<MeshVertex>& out) const;
    // A ball of voxels, thrown as one rigid body: it flies under gravity, hits the same surface bodies stand on, bounces,
    // rolls, knocks over whoever it hits, and where it comes to rest it is voxels again (stone, wood or ice), in the
    // fields, for the law. `at` and `velocity` in world metres; `material` is a stone's palette index. False if the air
    // is already full of pieces.
    bool throw_ball(Vec3 at, Vec3 velocity, double radius, BallKind kind, int material = 0);
    static constexpr std::size_t kMostPieces = 48;
    void note_force(Vec3 at_world, Vec3 impulse); // for what comes away because of it

    // ---- the open world ------------------------------------------------------------------------------
    bool open_world() const { return config_.open_world; }
    // World metres of lattice point (0, 0, 0).
    Vec3 origin() const;
    int window_cx() const { return origin_cx_; } // chunk coordinates of the window's first chunk
    int window_cy() const { return origin_cy_; }
    int window_cz() const { return origin_cz_; }
    int window_chunks() const { return config_.open_world ? config_.window_chunks : 0; }
    int window_layers() const { return config_.open_world ? config_.window_layers : 0; }
    bool in_window(int cx, int cy, int cz) const;
    bool column_in_window(int cx, int cy) const;
    // The layers of the map this world stands on (null for the proving ground).
    const WorldMap* map() const { return map_.get(); }
    // Bumped every time the window moves.
    std::uint64_t window_version() const { return window_version_; }
    const ChunkStore& store() const { return store_; }
    // A chunk outside the window as it now is: what is stored over what the generator makes. Fields
    // the generator and the store know nothing about are absent (they hold their law's init).
    void read_chunk(int cx, int cy, int cz, ChunkData& out) const;
    // Moves the window so chunk (cx, cy, cz) is its centre (the step does this on its own as you walk). Without
    // a layer, the one holding the ground there.
    void center_window_on(int cx, int cy, int cz);
    void center_window_on(int cx, int cy);

    // ---- beyond the window ---------------------------------------------------------------------------
    const std::vector<FarBody>& far_bodies() const { return far_bodies_; }
    std::size_t far_busy() const { return far_busy_.size(); }      // chunks still running on their own
    std::uint64_t far_runs() const { return far_runs_; }           // how many turns they have had
    double far_lag() const;                                        // seconds the busiest of them is behind
    double far_ms() const { return far_ms_; }                      // wall-clock cost of the last turn, for the overlay
    double map_total(const std::string& field) const;              // the land's law, summed over the map
    double map_at(const std::string& field, int nx, int ny) const; // one node of it
    const std::vector<std::string>& far_errors() const { return far_errors_; }

private:
    Playground(Solver solver, std::vector<Pour> pours, Config config);

    struct FieldWorker;
    struct FarJob;
    struct CellEdit {
        int field = -1;
        std::size_t cell = 0;
        float delta = 0.0f;
    };

    void bind_law();
    void add_to_cell(int field, std::size_t cell, float delta);
    void settle();     // waits for a running field step and takes it up
    void sync_front(); // the picture = the solver's state
    void note_violations();
    void move_body(PlayBody& body, double dt, const Vec3* want, bool jump);
    void think(Creature& creature, double dt);
    void apply_flows(const CreatureDef& def, std::vector<double>& locals, const PlayBody& body, double dt);
    void separate_bodies();
    void swing_turn(PlayBody& you);
    void start_swing(PlayBody& body, int slot, Vec3 way);
    bool fight_turn(Creature& c, const CreatureDef& def, Vec3& want);
    void land_blow(PlayBody& from, PlayBody& to, const MoveDef& move);
    void move_player(const Control& control);
    void pour_shape(int id, double amount, const Pour& pour, Vec3 where, Vec3 dir);
    void event(PlayEvent::Kind kind, int pour, Vec3 at, Vec3 to, double power);
    double sample(int id, Vec3 p) const;
    void note(std::string line);
    struct WindowMove {
        struct Leaving {
            int cx = 0;
            int cy = 0;
            int cz = 0;
            StoredChunk chunk;
        };
        struct Arriving {
            int cx = 0;
            int cy = 0;
            int cz = 0;
            std::vector<CreatureSpawn> spawns;
        };
        int x = 0; // the new first chunk
        int y = 0;
        int z = 0;
        int old_x = 0;
        int old_y = 0;
        int old_z = 0;
        std::vector<Leaving> leaving;
        std::vector<Arriving> arriving;
        std::vector<char> touched; // fields that are not all zero, before or after: the rest did not change
    };
    std::optional<WindowMove> wanted_move() const;
    void move_lattice(WindowMove& move);
    void finish_move(WindowMove& move);
    StoredChunk capture_chunk(const FieldSet& fields, int lx0, int ly0, int lz0, int cx, int cy, int cz) const;
    void take_creatures(int cx, int cy, int cz);
    void fill_chunk(FieldSet& f, int lx0, int ly0, int lz0, int cx, int cy, int cz, std::vector<CreatureSpawn>& spawns) const;
    void bring_creatures(int cx, int cy, int cz, const std::vector<CreatureSpawn>& spawns);
    void bring_far_bodies();
    void load_chunk(int cx, int cy, int cz);
    // FarLife.cpp
    void start_far_life();
    void far_tick();
    void finish_far_job();
    void walk_far_bodies();
    void note_busy(int cx, int cy, int cz, const StoredChunk& chunk);
    bool still_busy(const StoredChunk& chunk) const;
    void run_far_job(FarJob& job, Solver& solver);
    void run_far_turn();
    double feed_chance(const std::string& kind, double wx, double wy) const;
    // Goods.cpp
    void start_goods();
    // Falling.cpp
    void scan_falling();
    void lift_piece(const std::vector<std::size_t>& trunk, const std::vector<std::size_t>& crown);
    bool out_of_pieces(PlayBody& body) const; // pushed out of what is falling: true if it stands on it
    void pieces_turn();
    void set_piece_down(const FallingPiece& piece);
    // People.cpp
    void dress_person(Creature& c, const Settlement& s);
    void start_people();
    void people_turn();
    void bring_people();
    Vec3 door_of(const Person& who, const Settlement& s) const;
    bool bring_person(int town, int index);
    void caravans_near();
    bool bring_caravan(int i);
    void person_mind(Creature& c);
    void person_turn(Creature& c, const CreatureDef& def, Vec3& want, bool& jump, Vec3 pull, double strength);
    void hurt_by_you(Creature& c, double hurt);
    void person_died(const Creature& c);
    void beast_died(const Creature& c);
    // Chat.cpp
    void local_life_turn();
    void set_on(Creature& c, const Creature& foe);
    void end_brawl(Creature& c);
    void fight_over(Creature& c);
    PlayBody* foe_of(Creature& c, const CreatureDef& def);
    Creature* creature_mut(std::uint32_t id);
    void chatter();
    void bark(const Creature& c, const std::string& text, bool to_chat = true);
    Vec3 goal_of(const Creature& c, Creature::Doing& doing) const;
    // Where someone's day has them going at `seconds`, world metres, standing on its town's ground.
    Vec3 goal_at(const Person& who, int town, double seconds, Creature::Doing& doing) const;
    bool blocked_at(Vec3 p) const;
    std::vector<std::uint64_t> witnesses_of(const Creature& c, bool blames_you = true);
    std::vector<PlayBody*> all_bodies();
    Creature make_creature(int def, Vec3 where);

    Solver solver_;
    FieldSet front_;  // async: the finished state everyone reads
    FieldSet spare_;  // async: where the worker leaves the next one
    std::vector<CellEdit> edits_; // async: what was written since the running step started
    std::optional<WindowMove> moving_; // async: the move running instead of a step
    double owed_dt_ = 0.0;             // async: law time the moves have put off
    std::uint64_t move_due_ = 0;       // async: the tick the running move is taken up at
    std::vector<std::pair<std::string, double>> pending_params_;
    FieldListener listener_;
    // Async: the spare must take what changed in the last two steps (front and spare each saw one of them).
    std::vector<char> changed_now_;
    std::vector<char> changed_before_;
    bool stepping_ = false;
    double field_ms_ = 0.0;
    std::vector<std::shared_ptr<WorldModel>> minds_; // per creature kind; null for kinds that seek
    void plan_minds();
    std::shared_ptr<std::mutex> profile_mutex_ = std::make_shared<std::mutex>(); // a pointer: the playground moves
    StepProfile field_profile_;
    bool front_stale_ = false;
    std::vector<Pour> pours_;
    std::vector<LawSource> sources_;
    std::string pours_text_;
    Config config_;
    int occ_ = -1;
    int sdf_ = -1;
    int harm_ = -1;
    int push_[3] = {-1, -1, -1};
    int player_def_ = -1;
    std::vector<double> player_locals_;
    std::vector<PlayBody> bodies_;
    std::vector<Creature> creatures_;
    std::vector<Creature> born_;
    std::uint32_t next_creature_id_ = 1;
    MeleeBook melee_;
    std::vector<PlayEvent> events_;
    double coyote_ = 0.0;       // seconds since the body stepped off an edge: the ground still counts for a jump
    double jump_buffer_ = 0.0;  // seconds since it was asked to jump: it happens when it lands
    std::uint64_t tick_ = 0;
    std::vector<std::string> log_;
    std::vector<Violation> violations_;

    std::shared_ptr<const WorldMap> map_;
    ChunkStore store_;
    int origin_cx_ = 0;
    int origin_cy_ = 0;
    int origin_cz_ = 0;
    Vec3 home_{}; // world metres where you start (and wake)
    std::uint64_t window_version_ = 0;

    // Beyond the window.
    std::vector<FarBody> far_bodies_;
    std::map<std::int64_t, double> far_busy_; // chunk -> the world time it has been run up to
    std::vector<std::shared_ptr<FarJob>> far_jobs_; // the turn running, taken up at its fixed tick
    std::uint64_t far_due_ = 0;
    std::uint64_t far_next_turn_ = 0;
    std::uint64_t far_next_walk_ = 0;
    std::uint64_t far_next_map_ = 0;
    std::uint64_t far_runs_ = 0;
    std::vector<std::unique_ptr<Solver>> far_solvers_; // each one chunk and a ring around it
    double far_ms_ = 0.0;
    std::unique_ptr<Solver> map_solver_;      // the land's law, one cell per node
    std::vector<float> map_grazers_;          // what it said last, for the frame's readers
    std::vector<float> map_hunters_;
    std::vector<std::string> far_errors_;

    // What people wear and hold (assets/world/items.world).
    ItemBook items_;
    // What is coming down.
    std::vector<FallingPiece> pieces_;
    std::vector<char> tree_was_;
    std::vector<std::size_t> gone_waiting_; // what went while too much was already in the air
    std::set<std::size_t> set_down_;        // cells just filled by what came to rest
    std::uint64_t set_down_tick_ = 0;
    // A chunk outside the window, as it now is, kept as what differs from the generator.
    void store_state(int cx, int cy, int cz, const ChunkData& state);
    struct BeyondCell {
        int wx = 0; // world cells
        int wy = 0;
        int wz = 0;
        std::uint8_t matter = 0; // as FallingCell::matter, with 3 for a leaf
        short material = 0;
    };
    void set_down_beyond(const std::vector<BeyondCell>& cells);
    std::uint64_t tree_was_version_ = 0;
    Vec3 last_force_{};
    Vec3 last_force_at_{};
    std::uint64_t last_force_tick_ = 0;

    // The people.
    std::shared_ptr<Society> society_;
    std::vector<Bark> barks_;
    std::vector<ChatLine> chat_;
    int deaths_ = 0;
    double incident_left_ = 40.0;
    double day_shift_ = 0.0;  // seconds of the day that were skipped

    // Last, so they are destroyed first: their threads may be running on the members above.
    std::shared_ptr<FieldWorker> far_worker_;
    std::shared_ptr<FieldWorker> worker_;
};

} // namespace fieldengine::field
