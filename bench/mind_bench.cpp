// bench_mind: does the learned world model plan as well as planning with the true dynamics, and better than the
// hand-written rule?
//
// The task. life.law's forager in a 64 x 64 m world of grass patches and danger (the `claw` field), some of them
// overlapping, so eating and staying safe pull against each other. Its own lines give the dynamics (with the law's
// params): eaten = graze * hunger * grass, d hunger = hunger_rate - eat_gain * eaten, d fear = fear_rise * claw -
// fear_calm * fear, and it moves at `speed` along its action (|a| <= 1). Its cost is its plan line,
// hunger + 2 fear - 0.5 eaten. 64 foragers; an episode is 60 s (240 decisions of 0.25 s), after which each starts again
// at a random place with random hunger. The layout of the world is drawn from a seed.
//
// Policies, all deciding every 0.25 s:
//   still        does nothing;
//   random       a random direction at full speed;
//   seek         the hand-written rule the other creatures use: climb the gradient of
//                (0.2 + hunger) grass - (0.3 + 4 fear) claw, read 2.5 m either side;
//   mppi_oracle  MPPI with the TRUE dynamics (the same samples, horizon, noise and softmin as the learned planner):
//                what a perfect model would give this planner;
//   mppi_learned the engine's WorldModel: it observes its states and the fields on a ring around it, learns
//                ds/dt = f(s, a) online from its own experience and plans through it (CUDA when there is a card).
//
// Then the model alone, on a world it has never seen (another seed), under random actions: one-step prediction of the
// change of each feature (R^2), and the cost 12 steps ahead open loop (RMSE), against persistence (nothing changes).
//
//   bench_mind [--seeds 5] [--episodes 30] [--out FILE]
#include "field/Law.hpp"
#include "field/Solver.hpp"
#include "field/mind/WorldModel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace fieldengine::field;

namespace {

constexpr int kAgents = 64;
constexpr int kSteps = 240; // an episode: 60 s of 0.25 s decisions
constexpr int kWorld = 64;

std::uint64_t mix(std::uint64_t h)
{
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 33;
    return h;
}

double uniform(std::uint64_t key)
{
    return static_cast<double>(mix(key) >> 11) * (1.0 / 9007199254740992.0);
}

double gauss(std::uint64_t key)
{
    const double u1 = (static_cast<double>(mix(key) >> 40) + 0.5) / 16777216.0;
    const double u2 = (static_cast<double>(mix(key ^ 0x9e3779b97f4a7c15ull) >> 40) + 0.5) / 16777216.0;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

struct Params {
    double hunger_rate, eat_gain, graze, fear_rise, fear_calm, speed;
};

struct Agent {
    double x = 0, y = 0, hunger = 0.5, fear = 0.0, eaten = 0.0, age = 0.0;
};

// The world: a solver's fields (so the model observes them exactly as in the engine), grass and claw drawn from a seed.
struct World {
    std::unique_ptr<Solver> solver;
    int grass = -1;
    int claw = -1;
    const FieldSet& fields() const { return std::as_const(*solver).fields(); }
    double g(double x, double y) const { return fields().sample(grass, x, y, 0.5); }
    double c(double x, double y) const { return fields().sample(claw, x, y, 0.5); }
};

World make_world(const Law& law, std::uint64_t seed)
{
    World w;
    w.solver = std::make_unique<Solver>(law, GridShape{kWorld, kWorld, 1, 1.0});
    w.grass = w.solver->field("grass");
    w.claw = w.solver->field("claw");
    FieldSet& f = w.solver->fields();
    std::vector<float>& grass = f.values(w.grass);
    std::vector<float>& claw = f.values(w.claw);
    std::fill(grass.begin(), grass.end(), 0.0f);
    std::fill(claw.begin(), claw.end(), 0.0f);
    struct Blob { double x, y, r, a; };
    std::vector<Blob> food;
    std::vector<Blob> danger;
    for (int i = 0; i < 6; ++i) {
        food.push_back({6 + 52 * uniform(seed * 131 + i * 7 + 1), 6 + 52 * uniform(seed * 131 + i * 7 + 2), 4 + 4 * uniform(seed * 131 + i * 7 + 3), 1.0});
    }
    for (int i = 0; i < 5; ++i) {
        // Three sit on a patch of grass (eating there frightens), two anywhere.
        const Blob& near = food[static_cast<std::size_t>(i % food.size())];
        const bool on = i < 3;
        const double x = on ? near.x + 3 * gauss(seed * 977 + i * 5 + 1) : 6 + 52 * uniform(seed * 977 + i * 5 + 2);
        const double y = on ? near.y + 3 * gauss(seed * 977 + i * 5 + 3) : 6 + 52 * uniform(seed * 977 + i * 5 + 4);
        danger.push_back({x, y, 3 + 3 * uniform(seed * 977 + i * 5 + 5), 0.35});
    }
    for (int yy = 0; yy < kWorld; ++yy) {
        for (int xx = 0; xx < kWorld; ++xx) {
            const double px = xx + 0.5;
            const double py = yy + 0.5;
            double gv = 0.0;
            double cv = 0.0;
            for (const Blob& b : food) {
                gv += b.a * std::exp(-((px - b.x) * (px - b.x) + (py - b.y) * (py - b.y)) / (2 * b.r * b.r));
            }
            for (const Blob& b : danger) {
                cv += b.a * std::exp(-((px - b.x) * (px - b.x) + (py - b.y) * (py - b.y)) / (2 * b.r * b.r));
            }
            grass[static_cast<std::size_t>(yy * kWorld + xx)] = static_cast<float>(std::min(gv, 1.0));
            claw[static_cast<std::size_t>(yy * kWorld + xx)] = static_cast<float>(cv);
        }
    }
    return w;
}

// One decision's worth of the true dynamics (Euler, as the engine steps a creature's states).
void advance(const World& w, const Params& p, Agent& s, double ax, double ay, double dt)
{
    const double eaten = p.graze * s.hunger * w.g(s.x, s.y);
    const double claw = w.c(s.x, s.y);
    s.eaten = eaten;
    s.hunger = std::clamp(s.hunger + dt * (p.hunger_rate - p.eat_gain * eaten), 0.0, 1.0);
    s.fear = std::clamp(s.fear + dt * (p.fear_rise * claw - p.fear_calm * s.fear), 0.0, 1.0);
    s.x = std::clamp(s.x + p.speed * ax * dt, 0.5, kWorld - 0.5);
    s.y = std::clamp(s.y + p.speed * ay * dt, 0.5, kWorld - 0.5);
    s.age += dt;
}

double cost_of(const Agent& s)
{
    return s.hunger + 2.0 * s.fear - 0.5 * s.eaten;
}

Agent spawn(std::uint64_t seed, int episode, int id)
{
    const std::uint64_t k = seed * 1000003ull + static_cast<std::uint64_t>(episode) * 4099ull + static_cast<std::uint64_t>(id);
    Agent a;
    a.x = 2 + 60 * uniform(k * 3 + 1);
    a.y = 2 + 60 * uniform(k * 3 + 2);
    a.hunger = 0.3 + 0.6 * uniform(k * 3 + 3);
    return a;
}

// MPPI with the true dynamics: the planner of WorldModel (K samples, horizon H, sigma, softmin with a temperature a
// quarter of the spread, warm start shifted), a perfect model in its place.
struct Oracle {
    int K = 256;
    int H = 12;
    double sigma = 0.6;
    std::vector<std::vector<double>> mean;
    void act(const World& w, const Params& p, const Agent& s, int id, std::uint64_t tick, double dt, double& ax, double& ay)
    {
        std::vector<double>& m = mean[static_cast<std::size_t>(id)];
        if (m.empty()) {
            m.assign(static_cast<std::size_t>(H) * 2, 0.0);
        }
        std::vector<double> cost(static_cast<std::size_t>(K));
        std::vector<double> taken(static_cast<std::size_t>(K) * H * 2);
        for (int k = 0; k < K; ++k) {
            Agent f = s;
            double J = 0.0;
            for (int t = 0; t < H; ++t) {
                double a[2];
                for (int d = 0; d < 2; ++d) {
                    const double e = k == 0 ? 0.0 : gauss((static_cast<std::uint64_t>(id) * 0x100000001b3ull) ^ (tick * 0x9e3779b97f4a7c15ull) ^
                                                           (static_cast<std::uint64_t>(k) << 20) ^ (static_cast<std::uint64_t>(t) << 8) ^ static_cast<std::uint64_t>(d));
                    a[d] = std::clamp(m[static_cast<std::size_t>(t * 2 + d)] + sigma * e, -1.0, 1.0);
                    taken[(static_cast<std::size_t>(k) * H + t) * 2 + d] = a[d];
                }
                advance(w, p, f, a[0], a[1], dt);
                J += cost_of(f) + 0.01 * (a[0] * a[0] + a[1] * a[1]);
            }
            cost[static_cast<std::size_t>(k)] = J;
        }
        double lo = 1e30;
        double sum = 0.0;
        for (double c : cost) {
            lo = std::min(lo, c);
            sum += c;
        }
        const double lam = std::max(1e-4, 0.25 * (sum / K - lo));
        for (int q = 0; q < H * 2; ++q) {
            double wsum = 0.0;
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                const double e = std::exp(-(cost[static_cast<std::size_t>(k)] - lo) / lam);
                wsum += e;
                acc += e * taken[static_cast<std::size_t>(k) * H * 2 + static_cast<std::size_t>(q)];
            }
            m[static_cast<std::size_t>(q)] = acc / wsum;
        }
        ax = m[0];
        ay = m[1];
        const double norm = std::sqrt(ax * ax + ay * ay);
        if (norm > 1.0) {
            ax /= norm;
            ay /= norm;
        }
        m.erase(m.begin(), m.begin() + 2);
        m.push_back(0.0);
        m.push_back(0.0);
    }
};

std::vector<double> locals_of(const CreatureDef& def, const Agent& s)
{
    std::vector<double> out;
    for (const std::string& name : def.locals) {
        if (name == "hp") out.push_back(40.0);
        else if (name == "eaten") out.push_back(s.eaten);
        else if (name == "age") out.push_back(s.age);
        else if (name == "hunger") out.push_back(s.hunger);
        else if (name == "fear") out.push_back(s.fear);
        else out.push_back(0.0);
    }
    return out;
}

struct RunResult {
    std::vector<double> per_episode; // mean cost a decision, over all agents
    double plan_ms = 0.0;
    double learn_ms = 0.0;
};

RunResult run(const std::string& policy, const Law& law, const CreatureDef& def, int forager, const Params& p, const World& w,
    std::uint64_t seed, int episodes, std::unique_ptr<WorldModel>* keep_model)
{
    const double dt = def.plan_dt;
    RunResult r;
    std::unique_ptr<WorldModel> mind;
    if (policy == "mppi_learned") {
        mind = std::make_unique<WorldModel>(law, forager);
    }
    Oracle oracle;
    oracle.H = def.horizon;
    oracle.K = def.samples;
    oracle.mean.assign(kAgents + 1, {});
    const int F = mind ? mind->features() : 0;
    std::vector<float> obs;
    std::vector<std::uint32_t> ids;
    std::vector<float> actions;
    double plan_ms = 0.0;
    double learn_ms = 0.0;
    int plans = 0;
    int learns = 0;
    std::uint64_t tick = 0;
    for (int e = 0; e < episodes; ++e) {
        std::vector<Agent> agents(kAgents);
        for (int i = 0; i < kAgents; ++i) {
            agents[static_cast<std::size_t>(i)] = spawn(seed, e, i);
            if (mind) {
                mind->forget(static_cast<std::uint32_t>(i + 1));
            }
            oracle.mean[static_cast<std::size_t>(i + 1)].clear();
        }
        double total = 0.0;
        for (int step = 0; step < kSteps; ++step, ++tick) {
            std::vector<double> ax(kAgents, 0.0);
            std::vector<double> ay(kAgents, 0.0);
            if (mind) {
                obs.assign(static_cast<std::size_t>(kAgents) * F, 0.0f);
                ids.clear();
                for (int i = 0; i < kAgents; ++i) {
                    const Agent& s = agents[static_cast<std::size_t>(i)];
                    float* o = &obs[static_cast<std::size_t>(i) * F];
                    mind->observe(w.fields(), s.x, s.y, 0.5, locals_of(def, s), o);
                    mind->remember(static_cast<std::uint32_t>(i + 1), o, dt);
                    ids.push_back(static_cast<std::uint32_t>(i + 1));
                }
                mind->plan(obs, ids, tick, actions);
                plan_ms += mind->plan_ms();
                ++plans;
                for (int i = 0; i < kAgents; ++i) {
                    ax[static_cast<std::size_t>(i)] = actions[2 * static_cast<std::size_t>(i)];
                    ay[static_cast<std::size_t>(i)] = actions[2 * static_cast<std::size_t>(i) + 1];
                }
                if (tick % 4 == 3) {
                    mind->learn(6);
                    if (mind->updates() > 0) {
                        learn_ms += mind->learn_ms();
                        ++learns;
                    }
                }
            } else {
                const auto t0 = std::chrono::steady_clock::now();
                // Agents are independent (each has its own plan, the world is only read): one thread an agent.
#pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < kAgents; ++i) {
                    const Agent& s = agents[static_cast<std::size_t>(i)];
                    double& x = ax[static_cast<std::size_t>(i)];
                    double& y = ay[static_cast<std::size_t>(i)];
                    if (policy == "random") {
                        const double th = 6.283185307179586 * uniform(seed * 7919 + tick * 131 + static_cast<std::uint64_t>(i));
                        x = std::cos(th);
                        y = std::sin(th);
                    } else if (policy == "seek") {
                        const double d = def.sense;
                        auto v = [&](double px, double py) { return (0.2 + s.hunger) * w.g(px, py) - (0.3 + 4 * s.fear) * w.c(px, py); };
                        const double gx = (v(s.x + d, s.y) - v(s.x - d, s.y)) / (2 * d);
                        const double gy = (v(s.x, s.y + d) - v(s.x, s.y - d)) / (2 * d);
                        const double norm = std::sqrt(gx * gx + gy * gy) + 0.05;
                        x = gx / norm;
                        y = gy / norm;
                    } else if (policy == "mppi_oracle") {
                        oracle.act(w, p, s, i + 1, tick, dt, x, y);
                    }
                }
                plan_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                ++plans;
            }
            for (int i = 0; i < kAgents; ++i) {
                Agent& s = agents[static_cast<std::size_t>(i)];
                advance(w, p, s, ax[static_cast<std::size_t>(i)], ay[static_cast<std::size_t>(i)], dt);
                total += cost_of(s);
            }
        }
        r.per_episode.push_back(total / (kAgents * kSteps));
    }
    r.plan_ms = plans ? plan_ms / plans : 0.0;
    r.learn_ms = learns ? learn_ms / learns : 0.0;
    if (keep_model) {
        *keep_model = std::move(mind);
    }
    return r;
}

// The model on a world it has not seen, under random actions.
struct ModelEval {
    double r2_mean = 0.0;   // one-step, mean over features that change
    double r2_hunger = 0.0;
    double r2_fear = 0.0;
    double cost_rmse_model = 0.0;     // cost 12 steps ahead, open loop
    double cost_rmse_persist = 0.0;
};

ModelEval evaluate_model(WorldModel& mind, const CreatureDef& def, const Params& p, const World& w, std::uint64_t seed)
{
    const int F = mind.features();
    const double dt = def.plan_dt;
    const int H = def.horizon;
    int hunger_i = -1;
    int fear_i = -1;
    for (std::size_t i = 0; i < def.locals.size(); ++i) {
        if (def.locals[i] == "hunger") hunger_i = static_cast<int>(i);
        if (def.locals[i] == "fear") fear_i = static_cast<int>(i);
    }
    auto cost_feat = [&](const float* s) { return mind.cost(s); };
    std::vector<double> sse(static_cast<std::size_t>(F), 0.0);
    std::vector<double> sum(static_cast<std::size_t>(F), 0.0);
    std::vector<double> sq(static_cast<std::size_t>(F), 0.0);
    std::size_t n = 0;
    double se_model = 0.0;
    double se_persist = 0.0;
    std::size_t rollouts = 0;
    for (int e = 0; e < 4; ++e) {
        for (int i = 0; i < kAgents; ++i) {
            Agent s = spawn(seed, 500 + e, i);
            std::vector<float> o(static_cast<std::size_t>(F));
            std::vector<float> o2(static_cast<std::size_t>(F));
            std::vector<float> ds(static_cast<std::size_t>(F));
            for (int step = 0; step < 60; ++step) {
                const std::uint64_t key = seed * 31337 + static_cast<std::uint64_t>(e * 100000 + i * 1000 + step);
                // An open-loop rollout of H steps from here, every 10 steps.
                if (step % 10 == 0) {
                    Agent truth = s;
                    mind.observe(w.fields(), s.x, s.y, 0.5, locals_of(def, s), o.data());
                    std::vector<float> pred = o;
                    const float c0 = cost_feat(o.data());
                    for (int t = 0; t < H; ++t) {
                        const double th = 6.283185307179586 * uniform(key * 17 + static_cast<std::uint64_t>(t));
                        const float a[2] = {static_cast<float>(std::cos(th)), static_cast<float>(std::sin(th))};
                        mind.predict(pred.data(), a, ds.data());
                        for (int f = 0; f < F; ++f) {
                            pred[static_cast<std::size_t>(f)] += static_cast<float>(dt) * ds[static_cast<std::size_t>(f)];
                        }
                        advance(w, p, truth, a[0], a[1], dt);
                    }
                    mind.observe(w.fields(), truth.x, truth.y, 0.5, locals_of(def, truth), o2.data());
                    const double c_true = cost_feat(o2.data());
                    se_model += (cost_feat(pred.data()) - c_true) * (cost_feat(pred.data()) - c_true);
                    se_persist += (c0 - c_true) * (c0 - c_true);
                    ++rollouts;
                }
                // One step.
                mind.observe(w.fields(), s.x, s.y, 0.5, locals_of(def, s), o.data());
                const double th = 6.283185307179586 * uniform(key);
                const float a[2] = {static_cast<float>(std::cos(th)), static_cast<float>(std::sin(th))};
                advance(w, p, s, a[0], a[1], dt);
                mind.observe(w.fields(), s.x, s.y, 0.5, locals_of(def, s), o2.data());
                mind.predict(o.data(), a, ds.data());
                for (int f = 0; f < F; ++f) {
                    const double truth = (o2[static_cast<std::size_t>(f)] - o[static_cast<std::size_t>(f)]) / dt;
                    const double err = ds[static_cast<std::size_t>(f)] - truth;
                    sse[static_cast<std::size_t>(f)] += err * err;
                    sum[static_cast<std::size_t>(f)] += truth;
                    sq[static_cast<std::size_t>(f)] += truth * truth;
                }
                ++n;
            }
        }
    }
    ModelEval out;
    double r2_total = 0.0;
    int counted = 0;
    for (int f = 0; f < F; ++f) {
        const double mean = sum[static_cast<std::size_t>(f)] / n;
        const double var = sq[static_cast<std::size_t>(f)] / n - mean * mean;
        if (var < 1e-10) {
            continue; // a feature that never changes (hp, a constant field)
        }
        const double r2 = 1.0 - sse[static_cast<std::size_t>(f)] / n / var;
        r2_total += r2;
        ++counted;
        if (f == hunger_i) out.r2_hunger = r2;
        if (f == fear_i) out.r2_fear = r2;
    }
    out.r2_mean = counted ? r2_total / counted : 0.0;
    out.cost_rmse_model = std::sqrt(se_model / std::max<std::size_t>(rollouts, 1));
    out.cost_rmse_persist = std::sqrt(se_persist / std::max<std::size_t>(rollouts, 1));
    return out;
}

} // namespace

int main(int argc, char** argv)
{
    int seeds = 5;
    int episodes = 30;
    std::optional<std::string> out_path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--seeds" && i + 1 < argc) seeds = std::atoi(argv[++i]);
        else if (a == "--episodes" && i + 1 < argc) episodes = std::atoi(argv[++i]);
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
    }
    std::vector<LawSource> sources;
    for (const char* path : {"assets/law/world.law", "assets/law/magic.law", "assets/law/life.law"}) {
        const std::optional<LawSource> s = read_law_file(path);
        if (!s) {
            std::cerr << "run from the build directory (" << path << ")\n";
            return 1;
        }
        sources.push_back(*s);
    }
    CompileResult compiled = compile_law(sources);
    if (!compiled.law) {
        std::cerr << "the laws do not compile\n";
        return 1;
    }
    const Law& law = *compiled.law;
    const int forager = law.creature_id("forager");
    const CreatureDef& def = law.creatures[static_cast<std::size_t>(forager)];
    auto param = [&](const char* name) { return law.param_values[static_cast<std::size_t>(law.param_id(name))]; };
    const Params p{param("hunger_rate"), param("eat_gain"), param("graze"), param("fear_rise"), param("fear_calm"), def.speed};

    const std::vector<std::string> policies = {"still", "random", "seek", "mppi_oracle", "mppi_learned"};
    std::ostringstream json;
    json << std::setprecision(6) << "{\"agents\":" << kAgents << ",\"episode_steps\":" << kSteps << ",\"episodes\":" << episodes
         << ",\"horizon\":" << def.horizon << ",\"samples\":" << def.samples << ",\"seeds\":[";
    for (int seed = 1; seed <= seeds; ++seed) {
        const World w = make_world(law, static_cast<std::uint64_t>(seed));
        json << (seed > 1 ? "," : "") << "{\"seed\":" << seed << ",\"policies\":{";
        std::unique_ptr<WorldModel> model;
        for (std::size_t pi = 0; pi < policies.size(); ++pi) {
            const std::string& policy = policies[pi];
            const auto t0 = std::chrono::steady_clock::now();
            const RunResult r = run(policy, law, def, forager, p, w, static_cast<std::uint64_t>(seed), episodes, policy == "mppi_learned" ? &model : nullptr);
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            double last = 0.0;
            const int tail = std::min<int>(10, static_cast<int>(r.per_episode.size()));
            for (int e = static_cast<int>(r.per_episode.size()) - tail; e < static_cast<int>(r.per_episode.size()); ++e) {
                last += r.per_episode[static_cast<std::size_t>(e)];
            }
            last /= std::max(tail, 1);
            std::cerr << "seed " << seed << " " << std::left << std::setw(13) << policy << std::right << " cost/decision, last " << tail
                      << " episodes: " << last << "  (plan " << r.plan_ms << " ms for 64, " << secs << " s)\n";
            json << (pi ? "," : "") << "\"" << policy << "\":{\"last10\":" << last << ",\"plan_ms\":" << r.plan_ms << ",\"learn_ms\":" << r.learn_ms
                 << ",\"per_episode\":[";
            for (std::size_t e = 0; e < r.per_episode.size(); ++e) {
                json << (e ? "," : "") << r.per_episode[e];
            }
            json << "]}";
        }
        json << "}";
        if (model) {
            const World unseen = make_world(law, static_cast<std::uint64_t>(seed) + 1000);
            const ModelEval m = evaluate_model(*model, def, p, unseen, static_cast<std::uint64_t>(seed) + 1000);
            std::cerr << "seed " << seed << " model on an unseen world: one-step R2 " << m.r2_mean << " (hunger " << m.r2_hunger << ", fear "
                      << m.r2_fear << "); cost " << def.horizon << " steps ahead RMSE " << m.cost_rmse_model << " vs persistence "
                      << m.cost_rmse_persist << "; backend " << model->backend() << "\n";
            // The same, on the world it learned in (other starts, the same random actions): separates "does not carry
            // over to a new layout" from "does not predict what random actions do at all".
            const ModelEval s = evaluate_model(*model, def, p, w, static_cast<std::uint64_t>(seed) + 2000);
            std::cerr << "seed " << seed << " model on its own world: one-step R2 " << s.r2_mean << " (hunger " << s.r2_hunger << ", fear "
                      << s.r2_fear << "); cost RMSE " << s.cost_rmse_model << " vs persistence " << s.cost_rmse_persist << "\n";
            json << ",\"model\":{\"backend\":\"" << model->backend() << "\",\"r2_mean\":" << m.r2_mean << ",\"r2_hunger\":" << m.r2_hunger
                 << ",\"r2_fear\":" << m.r2_fear << ",\"cost_rmse_model\":" << m.cost_rmse_model << ",\"cost_rmse_persist\":" << m.cost_rmse_persist
                 << ",\"seen_r2_mean\":" << s.r2_mean << ",\"seen_r2_hunger\":" << s.r2_hunger << ",\"seen_r2_fear\":" << s.r2_fear
                 << ",\"seen_cost_rmse_model\":" << s.cost_rmse_model << ",\"seen_cost_rmse_persist\":" << s.cost_rmse_persist
                 << ",\"updates\":" << model->updates() << "}";
        }
        json << "}";
    }
    json << "]}\n";
    if (out_path) {
        std::ofstream(*out_path) << json.str();
    }
    std::cout << json.str();
    return 0;
}
