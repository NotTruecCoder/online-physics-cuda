#pragma once

#include "field/Grid.hpp"
#include "field/Law.hpp"
#include "field/nn/Net.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

// A WORLD MODEL FOR A KIND OF CREATURE (model 1). A creature of the law whose block has `plan = cost` and `observe F...`
// does not walk up a seek gradient: it learns, while it lives, how what it sees and its own states change with the way
// it goes, and it plans its way through what it learned.
//
//   observation  s = its locals (hp, eaten, age, its states), then each observed field at its cell and at eight points
//                    on a ring `3 sense` metres out: what is around it;
//   dynamics     ds/dt = f(s, a), a small MLP (a neural ODE on the observation), a = where it goes, |a| <= 1;
//   learning     online, from what actually happened every `dt` of its life: (s, a) -> (s' - s) / dt, the whole kind
//                pooled, in CUDA: fused forward and backward per row, weight gradients one thread a weight summing rows
//                in order (no atomics: a run repeats to the bit), Adam;
//   planning     MPPI: `samples` futures of `horizon` steps each, every creature at once (a block a creature, a thread
//                a future), the dynamics integrated in registers with the weights in shared memory, the cost of each
//                future the plan line of the law evaluated on the predicted states; the noise is a hash of
//                (creature, tick, future, step), so the same world plans the same way.
//
// With no CUDA it runs the same algorithm on the CPU (fewer futures), which is also what the gate compares against.
namespace fieldengine::field {

class WorldModel {
public:
    WorldModel(const Law& law, int creature);
    ~WorldModel();
    WorldModel(const WorldModel&) = delete;
    WorldModel& operator=(const WorldModel&) = delete;

    static constexpr int kRing = 8;

    [[nodiscard]] int features() const { return features_; }
    [[nodiscard]] int creature() const { return def_; }
    void observe(const FieldSet& fields, double x, double y, double z, const std::vector<double>& locals, float* out) const;

    // Pairs this observation with the one `id` made last, under the action it took then: one transition.
    void remember(std::uint32_t id, const float* obs, double dt);
    // Plans for every creature of the kind at once (obs: n x features). Returns n actions (ax, ay), |a| <= 1, and keeps
    // each one's plan to start from next time.
    void plan(const std::vector<float>& obs, const std::vector<std::uint32_t>& ids, std::uint64_t tick, std::vector<float>& actions);
    // A few gradient steps on what it has seen. Nothing until it has seen enough to know its scales.
    void learn(int iterations);
    void forget(std::uint32_t id);

    [[nodiscard]] const char* backend() const { return gpu_ ? "CUDA" : "CPU"; }
    [[nodiscard]] double loss() const { return loss_; }
    [[nodiscard]] std::size_t transitions() const { return count_; }
    [[nodiscard]] double plan_ms() const { return plan_ms_; }
    [[nodiscard]] double learn_ms() const { return learn_ms_; }
    [[nodiscard]] int updates() const { return updates_; }
    [[nodiscard]] const NetDecl& net() const;
    [[nodiscard]] const std::string& why_cpu() const { return why_cpu_; }

    // The plan line as CUDA (on the predicted state s) and its host twin, for the gate.
    [[nodiscard]] std::string cost_code() const;
    [[nodiscard]] float cost(const float* s) const;
    // One prediction through the model (host), for the gate.
    void predict(const float* s, const float* a, float* ds_dt) const;

private:
    struct Gpu;
    void normalise();
    void rollout_cpu(const float* obs, std::uint32_t id, std::uint64_t tick, std::vector<float>& mean);

    const Law& law_;
    int def_ = -1;
    int locals_ = 0;
    int features_ = 0;
    int horizon_ = 12;
    int samples_ = 128;
    double dt_ = 0.25;
    double radius_ = 6.0;
    std::vector<int> observed_;
    NetDecl net_;
    std::vector<float> scale_;  // what one unit of the net's output is, per feature
    bool scaled_ = false;
    // Replay: rows of [s, a] and their target (s' - s) / dt / scale.
    static constexpr std::size_t kCapacity = 16384;
    std::vector<float> x_;
    std::vector<float> t_;
    std::size_t count_ = 0;
    std::size_t head_ = 0;
    struct Last {
        std::vector<float> obs;
        float action[2] = {0.0f, 0.0f};
        bool acted = false;
    };
    std::map<std::uint32_t, Last> last_;
    std::map<std::uint32_t, std::vector<float>> mean_; // each creature's plan (horizon x 2), shifted on
    std::unique_ptr<Gpu> gpu_;
    std::string why_cpu_;
    double loss_ = 0.0;
    double plan_ms_ = 0.0;
    double learn_ms_ = 0.0;
    int updates_ = 0;
    std::uint64_t learn_round_ = 0;
};

} // namespace fieldengine::field
