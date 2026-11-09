#pragma once

#include "field/Grid.hpp"
#include "field/nn/Net.hpp"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// A LAW IS TEXT. The developer writes fields and the equations that couple them; the engine compiles
// them and integrates them. Magic is not a system bolted onto physics — it is more terms in the same
// law, so every body, item and ray already accounts for it, because "accounting for it" is simply
// reading the fields it writes.
//
//   # fields carry their contracts
//   field T     init 290  range 0 6000  unit K
//   field fuel  init 0    range 0 1
//   field solid init 0    range 0 1     static
//
//   param k_heat = 0.8
//
//   term conduction                      # a term: equations evaluated on the same pre-state
//     d T = diffuse(k_heat, T)
//     conserves T                        # contract: the term moves heat, never makes it
//     max_dt = 0.15 * h * h / k_heat     # contract: the stability limit the author vouches for
//
//   derive sdf = distance(solid)         # recomputed after every step; read by rays and bodies
//   derive n   = 1 - 2 * phi / c2        # the optical index light bends along
//
// Stability and invariants are declared by the author, and the engine's job is to honour and CHECK
// them: `max_dt` sets the substeps, `conserves` and `range` are verified after every term and broken
// ones are reported by name, never silently clamped (unless the field says `clamp`).
namespace fieldengine::field {

// One source file of law. Several are compiled together (base physics + magic + a mod), and an error
// names the source and the line.
struct LawSource {
    std::string name;
    std::string text;
};

struct FieldDecl {
    std::string name;
    double init = 0.0;
    double lo = -1e30;
    double hi = 1e30;
    bool clamp = false;   // out-of-range values are clamped instead of reported
    bool is_static = false; // no term may write it (geometry edits come from outside the law)
    bool derived = false; // recomputed by a `derive`, never integrated
    std::string unit;
};

// Expression tree, evaluated over the whole lattice at once (one op = one pass over the cells).
struct Node;

struct Equation {
    int target = -1; // field id
    std::shared_ptr<Node> rhs;
};

struct Term {
    std::string name;
    std::string where; // "source:line"
    std::vector<Equation> equations;
    // Groups of field ids whose COMBINED total this term must not change: `conserves T` is one group
    // of one; `conserves solid + rubble + lava` is one group of three (melting moves matter between
    // them, never makes it). Several groups on one line (`conserves a b`) are checked separately.
    std::vector<std::vector<int>> conserves;
    std::shared_ptr<Node> max_dt;    // scalar; null = unlimited
    // A LEARNED SCHEME FOR THE SAME TERM, RUN UNDER GUARD. `learned d T = ldiffuse(net, k, T)` and `learned max_dt = ...`:
    // when learned schemes are on, these run instead of the equations above, with their own (larger) step. If the step
    // breaks a contract of the term (range, finite, conserves), it is undone and the classical equations run instead,
    // with their substeps: what is learned can make the world faster, never wrong past what the law promises.
    std::vector<Equation> learned;
    std::shared_ptr<Node> learned_max_dt;
    std::vector<int> reads;          // derived from the expressions, never declared by hand
    std::vector<int> writes;
};

// `derive X = e` sets X; `derive X += e` adds a contribution, so another source (magic, a mod) can
// bend light or add a gravity source without editing the base law. All derives of one field run
// together — the `=` first, then the `+=` in source order — at the position the field first appears.
struct Derive {
    int target = -1;
    bool accumulate = false;
    std::string where;
    std::shared_ptr<Node> rhs;
    std::vector<int> reads;
};

// A CREATURE IS A BODY THAT READS THE FIELDS WHERE IT STANDS AND WRITES THEM BACK. Its state is a few
// numbers with their own equations (`d hunger = ...`), it walks up the gradient of one expression over
// the fields around it (`seek = ...`), and it takes from or gives to the cell it stands in (`eat grass
// = ...`, `emit scent = ...`). Nothing about grazing, fear or herding is in the engine.
//
//   creature grazer
//     radius 0.35  speed 2.2  hp 40  wander 0.4
//     state hunger init 0.3 range 0 1
//     d hunger = hunger_rate - eat_gain * eaten
//     seek = hunger * grass - shy * scent
//     eat grass = graze * hunger * grass
//     hurt = starve * step(hunger - 0.99)
//     breed = litter * step(0.2 - hunger)
//     colour 0.6 0.5 0.3
//
// `creature player` is the body you steer: its seek is your keyboard, the rest of its lines hold.
inline constexpr int kLocalHp = 0;      // builtin locals every creature can read
inline constexpr int kLocalEaten = 1;   // what its `eat` lines took last tick, per second
inline constexpr int kLocalAge = 2;     // seconds alive
inline constexpr int kFirstState = 3;   // its own `state` lines follow

struct CreatureFlow {
    int field = -1;
    bool eat = false; // eat: taken from the cell (never more than is there); emit: added to it
    std::shared_ptr<Node> rate; // per second
};

// event NAME STATE above|below X weight W [sets STATE V]: a state crossing a line is something that happened.
struct CreatureEvent {
    std::string name;
    int state = -1;
    bool above = true;
    double line = 0.0;
    double weight = 0.5; // how much it matters to whoever hears of it, 0..1
    int sets = -1;       // and what it does to the state of things
    double to = 0.0;
};

struct CreatureDef {
    std::string name;
    std::string where;
    std::string skin;    // assets/world/skins/NAME.model; none, it is drawn by its colour
    std::vector<CreatureEvent> events;
    double radius = 0.35;
    double speed = 2.0;
    double hp = 40.0;
    double wander = 0.4; // fraction of its speed it ambles at where the seek field is flat
    double sense = 0.5;  // metres: how far either side it compares its seek expression
    std::string fights;  // the weapon whose moves it uses; empty, it does not
    int skill = 0;       // and how many of them it knows
    bool riled = false;  // it only fights once it is hurt
    std::vector<std::string> prey; // what it goes for, by name, `player` for you; empty, only you
    double heed = 0.05;  // a pull of this gradient moves it at half speed: dx/dt = v grad s / (|grad s| + heed)
    std::vector<std::string> locals; // hp, eaten, age, then its states
    std::vector<double> init;        // per local
    std::vector<double> lo;
    std::vector<double> hi;
    std::vector<std::shared_ptr<Node>> rates; // per local: d LOCAL = expr (null = constant)
    std::shared_ptr<Node> seek;
    std::shared_ptr<Node> hurt;               // HP lost per second, on top of what the fields do
    std::shared_ptr<Node> breed;              // newborns per second
    float colour[3] = {0.63f, 0.49f, 0.31f};
    std::vector<CreatureFlow> flows;
    std::vector<int> reads;                   // fields its expressions read
    // A MIND THAT PLANS (world model): `plan = cost over its states` replaces `seek`. It observes `observe` fields in a
    // ring around it, learns online how what it sees and its states move with what it does (a neural ODE in CUDA), and
    // picks its way by sampling `samples` futures `horizon` steps of `plan_dt` long (MPPI). Null: it seeks.
    std::shared_ptr<Node> plan;
    std::vector<int> observe;
    int horizon = 12;
    int samples = 128;
    double plan_dt = 0.25;
};

// WHAT A FIELD LOOKS LIKE IN THE OPEN AIR. The picture draws matter by itself; everything else — flame,
// smoke, mana — is invisible until the law says how to show it, so a new element needs no renderer code:
//
//   show T         colour 1 0.55 0.12  above 480  full 1300  alpha 0.75
//   show mana_fire colour 1 0.45 0.10  above 0.15 full 6     alpha 0.6
//
// A cell not full of matter, with the field above `above`, holds a translucent cube, growing to a whole
// voxel at `alpha` by `full`.
struct FieldVisual {
    int field = -1;
    float colour[3] = {1.0f, 1.0f, 1.0f};
    double above = 0.0;
    double full = 1.0;
    double alpha = 0.6;
};

struct Law {
    std::vector<FieldDecl> fields; // ids = positions
    std::vector<std::string> param_names;
    std::vector<double> param_values;
    std::vector<Term> terms;
    std::vector<Derive> derives;
    std::vector<CreatureDef> creatures;
    std::vector<FieldVisual> visuals;
    std::vector<NetDecl> nets;

    [[nodiscard]] int field_id(std::string_view name) const;
    [[nodiscard]] int net_id(std::string_view name) const;
    [[nodiscard]] int param_id(std::string_view name) const;
    [[nodiscard]] int creature_id(std::string_view name) const;
};

struct CompileResult {
    std::optional<Law> law;
    std::vector<std::string> errors;
};

CompileResult compile_law(const std::vector<LawSource>& sources);

// Reads a law file from disk; the source name is the path (so errors point at it).
std::optional<LawSource> read_law_file(const std::string& path);

// A deep copy: expression nodes own their buffers (and Poisson its warm start), so two solvers built
// from one compiled law must not share them.
Law clone_law(const Law& law);

// ---- evaluation (used by the solver; exposed for tests) ------------------------------------------

struct EvalContext {
    const FieldSet* fields = nullptr;
    const std::vector<double>* params = nullptr;
    // Optional: per field, true when every cell is exactly 0. Such a field evaluates as the scalar 0,
    // and zero propagates through products, so a term about mana nobody has cast costs nothing.
    const std::vector<char>* zero = nullptr;
    // Optional: one gather of one term (or one derive). Every equation of a gather reads the same
    // pre-state, so a subexpression the term writes twice — `step(T - melt_point)` in four equations
    // of melting — is computed once per epoch and shared. 0 = no sharing.
    std::uint64_t epoch = 0;
    // The law's networks (mlp, flux, ldiffuse) and the step they are taken for: flux limits itself so no cell goes past
    // its field's range in one step of `dt`. 0 = no step (a derive): no limit.
    const std::vector<NetDecl>* nets = nullptr;
    double dt = 0.0;
};

// Result of evaluating a node: either a scalar or a pointer to `cells` floats.
struct Value {
    const float* array = nullptr;
    double scalar = 0.0;
    [[nodiscard]] bool is_scalar() const { return array == nullptr; }
    [[nodiscard]] double at(std::size_t i) const { return array ? static_cast<double>(array[i]) : scalar; }
};

Value evaluate(Node& node, const EvalContext& context);
// Evaluates an expression that must be scalar (max_dt); returns nullopt if it reads a field.
std::optional<double> evaluate_scalar(Node& node, const EvalContext& context);

// A creature's expressions, at one point: a field reads its trilinear sample there (metres, lattice
// coordinates), a local reads the creature's own state.
struct PointContext {
    const FieldSet* fields = nullptr;
    const std::vector<double>* params = nullptr;
    const std::vector<double>* locals = nullptr;
    const std::vector<NetDecl>* nets = nullptr;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};
double evaluate_at(const Node& node, const PointContext& context);

// ---- the law as mathematics ----------------------------------------------------------------------
// The same expression trees the solver runs, printed as the equations they are: diffuse(k, T) is
// ∇·(k∇T), step(x) is the Heaviside H(x), pos(x) is [x]₊. What the designer reads in the inspector is
// what is integrated, not a description of it.
std::string to_math(const Node& node, const Law& law, const std::vector<std::string>* locals = nullptr);

struct LawEquation {
    enum class Kind { Rate, Derived, Creature };
    Kind kind = Kind::Rate;
    std::string lhs;                 // "∂T/∂t", "occ", "grazer: d hunger/dt"
    std::vector<std::string> parts;  // summands, each already printed
    std::vector<std::string> source; // per part: the term (or line) it comes from
};
// Every integrated field's full right-hand side (all terms that write it, summed), then every derived
// field, then every creature's equations.
std::vector<LawEquation> law_equations(const Law& law);

} // namespace fieldengine::field
