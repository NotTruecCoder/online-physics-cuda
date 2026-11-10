#include "world/Playground.hpp"

#include "world/VoxelMesh.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

// WHAT COMES AWAY IN ONE PIECE. A trunk with nothing under it is not a heap of rubble, it is a rigid body: it leaves the fields,
// falls and turns by Newton, with the mass and the moment of inertia of its own voxels, a force applied at where it was struck,
// gravity about wherever it touches, and collisions with the same surface bodies stand on. So where it falls is worked out, not
// chosen: struck low, what is above the blow goes the other way, and with nothing else its own lean takes it. When it comes to
// rest it is voxels again.
namespace fieldengine::field {

namespace {

using M3 = std::array<std::array<double, 3>, 3>;

constexpr int kLeafReach = 8;   // cells of leaf a leaf can hang by, from wood
constexpr int kCrownReach = 16; // and how far a tree's crown is looked for

Vec3 apply(const double m[3][3], Vec3 v)
{
    return Vec3{m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z, m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z, m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
}

Vec3 apply_transposed(const double m[3][3], Vec3 v)
{
    return Vec3{m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z, m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z, m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z};
}

Vec3 horizontal_of(Vec3 v)
{
    return Vec3{v.x, v.y, 0.0};
}

Vec3 cross(Vec3 a, Vec3 b)
{
    return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

void invert(const double a[3][3], double out[3][3])
{
    const double c00 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
    const double c01 = a[1][2] * a[2][0] - a[1][0] * a[2][2];
    const double c02 = a[1][0] * a[2][1] - a[1][1] * a[2][0];
    const double det = a[0][0] * c00 + a[0][1] * c01 + a[0][2] * c02;
    const double k = 1.0 / (std::abs(det) > 1e-12 ? det : 1e-12);
    out[0][0] = c00 * k;
    out[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * k;
    out[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * k;
    out[1][0] = c01 * k;
    out[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * k;
    out[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * k;
    out[2][0] = c02 * k;
    out[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * k;
    out[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * k;
}

} // namespace

void Playground::note_force(Vec3 at_world, Vec3 impulse)
{
    last_force_at_ = at_world;
    last_force_ = impulse;
    last_force_tick_ = tick_;
}

// What has been taken away, and what stood on it. Wood only leans on wood or on what does not move; leaves go with the wood they grow on.
void Playground::scan_falling()
{
    if (!config_.falling || tick_ % 15u != 3u) {
        return;
    }
    const FieldSet& f = fields();
    const GridShape& g = f.shape();
    const int wood = solver_.field("wood");
    const int leaf = solver_.field("leaf");
    const int solid = solver_.field("solid");
    const int ice = solver_.field("ice");
    const int built = solver_.field("built");
    if (wood < 0 || leaf < 0 || solid < 0) {
        return;
    }
    const std::vector<float>& w = f.values(wood);
    const std::vector<float>& l = f.values(leaf);
    const std::vector<float>& st = f.values(solid);
    const std::vector<float>* b = built >= 0 ? &f.values(built) : nullptr;
    const std::vector<float>* ic = ice >= 0 ? &f.values(ice) : nullptr;
    const std::size_t cells = w.size();
    const bool fresh = tree_was_.size() != cells || tree_was_version_ != window_version_;
    tree_was_.resize(cells, 0);
    tree_was_version_ = window_version_;
    const auto is_wood = [&](std::size_t i) { return w[i] > 0.5f && (b == nullptr || (*b)[i] < 0.5f); };
    const auto is_leaf = [&](std::size_t i) { return l[i] > 0.5f && (b == nullptr || (*b)[i] < 0.5f); };
    std::vector<std::size_t> gone;
    if (fresh) {
        gone_waiting_.clear();
    }
    gone.swap(gone_waiting_);
    for (std::size_t i = 0; i < cells; ++i) {
        const char now = is_wood(i) || is_leaf(i) ? 1 : 0;
        if (!fresh && tree_was_[i] != 0 && now == 0) {
            gone.push_back(i);
        }
        tree_was_[i] = now;
    }
    // What went is not forgotten because the air is full: it is looked at when there is room.
    if (pieces_.size() >= 12) {
        gone_waiting_.swap(gone);
        return;
    }
    if (gone.empty()) {
        return;
    }
    const int nxy = g.nx * g.ny;
    const auto neighbour = [&](std::size_t i, int k, std::size_t& out) {
        static const int dx[6] = {1, -1, 0, 0, 0, 0};
        static const int dy[6] = {0, 0, 1, -1, 0, 0};
        static const int dz[6] = {0, 0, 0, 0, 1, -1};
        const int z = static_cast<int>(i / static_cast<std::size_t>(nxy));
        const int rest = static_cast<int>(i % static_cast<std::size_t>(nxy));
        const int y = rest / g.nx;
        const int x = rest % g.nx;
        const int nx = x + dx[k];
        const int ny = y + dy[k];
        const int nz = z + dz[k];
        if (nx < 0 || ny < 0 || nz < 0 || nx >= g.nx || ny >= g.ny || nz >= g.nz) {
            return false;
        }
        out = g.index(nx, ny, nz);
        return true;
    };
    std::set<std::size_t> seen;
    for (const std::size_t from : gone) {
        for (int side = 0; side < 6; ++side) {
            std::size_t start = 0;
            if (!neighbour(from, side, start) || seen.count(start) != 0) {
                continue;
            }
            // Leaves hang on wood, through other leaves, only so far: touching the next tree's crown holds nothing up.
            if (!is_wood(start) && is_leaf(start)) {
                std::vector<std::size_t> loose{start};
                seen.insert(start);
                bool too_many = false;
                for (std::size_t q = 0; q < loose.size() && !too_many; ++q) {
                    too_many = loose.size() > 40000; // a whole forest's canopy can be one mass of leaves
                    for (int k = 0; k < 6; ++k) {
                        std::size_t n = 0;
                        if (neighbour(loose[q], k, n) && is_leaf(n) && seen.insert(n).second) {
                            loose.push_back(n);
                        }
                    }
                }
                if (too_many) {
                    continue;
                }
                // From what holds them, inward.
                std::map<std::size_t, int> reached;
                std::vector<std::size_t> front;
                for (const std::size_t i : loose) {
                    for (int k = 0; k < 6; ++k) {
                        std::size_t n = 0;
                        // Past the edge of the lattice is out of sight: it may hold.
                        const bool edge = !neighbour(i, k, n);
                        if (edge || w[n] > 0.5f || (k == 5 && st[n] > 0.5f)) {
                            reached[i] = 0;
                            front.push_back(i);
                            break;
                        }
                    }
                }
                for (std::size_t q = 0; q < front.size(); ++q) {
                    const int depth = reached[front[q]];
                    if (depth >= kLeafReach) {
                        continue;
                    }
                    for (int k = 0; k < 6; ++k) {
                        std::size_t n = 0;
                        if (neighbour(front[q], k, n) && is_leaf(n) && reached.count(n) == 0) {
                            reached[n] = depth + 1;
                            front.push_back(n);
                        }
                    }
                }
                std::vector<std::size_t> falling;
                for (const std::size_t i : loose) {
                    if (reached.count(i) == 0) {
                        falling.push_back(i);
                    }
                }
                if (!falling.empty()) {
                    lift_piece({}, falling);
                }
                continue;
            }
            if (!is_wood(start)) {
                continue;
            }
            // The wood that hangs together with this.
            // Only so far: crowns that touch are not joined, and what is holding up a tree is under it, not two trees away.
            std::vector<std::size_t> trunk{start};
            seen.insert(start);
            bool held = false;
            const int start_rest = static_cast<int>(start % static_cast<std::size_t>(nxy));
            const int sx = start_rest % g.nx;
            const int sy = start_rest / g.nx;
            for (std::size_t q = 0; q < trunk.size() && !held; ++q) {
                for (int k = 0; k < 6; ++k) {
                    std::size_t n = 0;
                    if (!neighbour(trunk[q], k, n)) {
                        // Past the edge of the lattice it may stand on anything; but nothing stands on what is over it.
                        if (k != 4) {
                            held = true;
                            break;
                        }
                        continue;
                    }
                    const int rest = static_cast<int>(n % static_cast<std::size_t>(nxy));
                    if (std::abs(rest % g.nx - sx) > 14 || std::abs(rest / g.nx - sy) > 14) {
                        continue;
                    }
                    if (st[n] > 0.5f || (ic != nullptr && (*ic)[n] > 0.5f) || (b != nullptr && (*b)[n] > 0.5f && (w[n] > 0.5f || st[n] > 0.5f))) {
                        held = true;
                        break;
                    }
                    if (is_wood(n) && seen.insert(n).second) {
                        trunk.push_back(n);
                    }
                }
            }
            if (held) {
                continue;
            }
            // Its leaves: all those nearer to this wood than to any other, and whatever hung on them and on nothing else.
            std::set<std::size_t> own(trunk.begin(), trunk.end());
            int x0 = g.nx, x1 = 0, y0 = g.ny, y1 = 0, z0 = g.nz, z1 = 0;
            for (const std::size_t t : trunk) {
                const int z = static_cast<int>(t / static_cast<std::size_t>(nxy));
                const int rest = static_cast<int>(t % static_cast<std::size_t>(nxy));
                x0 = std::min(x0, rest % g.nx);
                x1 = std::max(x1, rest % g.nx);
                y0 = std::min(y0, rest / g.nx);
                y1 = std::max(y1, rest / g.nx);
                z0 = std::min(z0, z);
                z1 = std::max(z1, z);
            }
            x0 = std::max(x0 - kCrownReach, 0);
            y0 = std::max(y0 - kCrownReach, 0);
            z0 = std::max(z0 - kCrownReach, 0);
            x1 = std::min(x1 + kCrownReach, g.nx - 1);
            y1 = std::min(y1 + kCrownReach, g.ny - 1);
            z1 = std::min(z1 + kCrownReach, g.nz - 1);
            // Two fronts, one from the other trees and one from this: each leaf goes with the first to reach it.
            std::map<std::size_t, bool> whose;
            std::vector<std::pair<std::size_t, int>> front;
            for (int z = z0; z <= z1; ++z) {
                for (int y = y0; y <= y1; ++y) {
                    for (int x = x0; x <= x1; ++x) {
                        const std::size_t i = g.index(x, y, z);
                        if (is_wood(i) && own.count(i) == 0) {
                            front.push_back({i, 0});
                        }
                    }
                }
            }
            const std::size_t others = front.size();
            for (const std::size_t t : trunk) {
                front.push_back({t, 0});
            }
            // (Level by level: the queue is in order of depth, with the other trees first on each.)
            for (std::size_t q = 0; q < front.size(); ++q) {
                const auto [cell, depth] = front[q];
                if (depth >= kCrownReach) {
                    continue;
                }
                const auto seen_as = whose.find(cell);
                const bool mine = seen_as != whose.end() ? seen_as->second : q >= others;
                for (int k = 0; k < 6; ++k) {
                    std::size_t n = 0;
                    if (neighbour(cell, k, n) && is_leaf(n) && whose.count(n) == 0) {
                        whose[n] = mine;
                        front.push_back({n, depth + 1});
                    }
                }
            }
            std::vector<std::size_t> crown;
            for (const auto& [cell, mine] : whose) {
                if (mine) {
                    crown.push_back(cell);
                }
            }
            // Further out than any wood reaches, joined only through its own crown.
            for (std::size_t q = 0; q < crown.size(); ++q) {
                for (int k = 0; k < 6; ++k) {
                    std::size_t n = 0;
                    if (neighbour(crown[q], k, n) && is_leaf(n) && whose.count(n) == 0) {
                        whose[n] = true;
                        crown.push_back(n);
                    }
                }
            }
            lift_piece(trunk, crown);
        }
    }
}

// Out of the fields and into the air.
void Playground::lift_piece(const std::vector<std::size_t>& trunk, const std::vector<std::size_t>& crown)
{
    const FieldSet& f = fields();
    const GridShape& g = f.shape();
    const int wood = solver_.field("wood");
    const int leaf = solver_.field("leaf");
    const int material = solver_.field("material");
    const Vec3 o = origin();
    const MatterView view = MatterView::of(f);
    const VoxelPalette palette;
    FallingPiece piece;
    Vec3 sum{};
    double mass = 0.0;
    const auto add = [&](std::size_t i, bool is_leaf) {
        const int z = static_cast<int>(i / static_cast<std::size_t>(g.nx * g.ny));
        const int rest = static_cast<int>(i % static_cast<std::size_t>(g.nx * g.ny));
        const int y = rest / g.nx;
        const int x = rest % g.nx;
        const VoxelCell cell = classify(view, x, y, z, palette);
        FallingCell c;
        c.x = static_cast<float>((x + 0.5) * g.cell);
        c.y = static_cast<float>((y + 0.5) * g.cell);
        c.z = static_cast<float>((z + 0.5) * g.cell);
        c.abgr = 0xff000000u | (static_cast<std::uint32_t>(cell.b) << 16) | (static_cast<std::uint32_t>(cell.g) << 8) | cell.r;
        c.leaf = is_leaf;
        c.material = material >= 0 ? static_cast<short>(f.values(material)[i]) : 0;
        piece.cells.push_back(c);
        const double m = is_leaf ? 0.15 : 1.0;
        sum = sum + Vec3{c.x, c.y, c.z} * m;
        mass += m;
    };
    for (const std::size_t i : trunk) {
        add(i, false);
    }
    for (const std::size_t i : crown) {
        add(i, true);
    }
    const Vec3 com = sum * (1.0 / mass);
    // What it carried and the sum of what was done to it: the push of the law where it stands, and any blow that just landed.
    Vec3 pushed{};
    Vec3 lowest{1e30, 1e30, 1e30};
    double inertia[3][3] = {};
    const double half = g.cell * 0.5;
    for (FallingCell& c : piece.cells) {
        const Vec3 at{c.x, c.y, c.z};
        if (trunk.empty() && at.z < lowest.z) {
            lowest = at;
        }
        const double m = c.leaf ? 0.15 : 1.0;
        pushed = pushed + Vec3{sample(push_[0], at), sample(push_[1], at), sample(push_[2], at)} * m;
        if (!c.leaf && at.z < lowest.z) {
            lowest = at;
        }
        const Vec3 r = at - com;
        c.x = static_cast<float>(r.x);
        c.y = static_cast<float>(r.y);
        c.z = static_cast<float>(r.z);
        const double own = m * (2.0 * half * half / 3.0);
        inertia[0][0] += m * (r.y * r.y + r.z * r.z) + own;
        inertia[1][1] += m * (r.x * r.x + r.z * r.z) + own;
        inertia[2][2] += m * (r.x * r.x + r.y * r.y) + own;
        inertia[0][1] -= m * r.x * r.y;
        inertia[0][2] -= m * r.x * r.z;
        inertia[1][2] -= m * r.y * r.z;
    }
    inertia[1][0] = inertia[0][1];
    inertia[2][0] = inertia[0][2];
    inertia[2][1] = inertia[1][2];
    invert(inertia, piece.inverse);
    piece.mass = mass;
    piece.at = com + o;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            piece.turned[i][j] = i == j ? 1.0 : 0.0;
        }
    }
    // Which faces look out: nothing inside is drawn.
    std::map<std::tuple<int, int, int>, int> places;
    for (std::size_t c = 0; c < piece.cells.size(); ++c) {
        places[{static_cast<int>(std::lround((piece.cells[c].x + com.x) / g.cell - 0.5)), static_cast<int>(std::lround((piece.cells[c].y + com.y) / g.cell - 0.5)), static_cast<int>(std::lround((piece.cells[c].z + com.z) / g.cell - 0.5))}] = static_cast<int>(c);
    }
    for (auto& [key, index] : places) {
        const auto [x, y, z] = key;
        std::uint8_t faces = 0;
        faces |= places.count({x + 1, y, z}) ? 0 : 1;
        faces |= places.count({x - 1, y, z}) ? 0 : 2;
        faces |= places.count({x, y + 1, z}) ? 0 : 4;
        faces |= places.count({x, y - 1, z}) ? 0 : 8;
        faces |= places.count({x, y, z + 1}) ? 0 : 16;
        faces |= places.count({x, y, z - 1}) ? 0 : 32;
        piece.cells[static_cast<std::size_t>(index)].faces = faces;
    }
    // What bodies run into: its own cells, in its own frame.
    {
        int lo[3] = {1 << 30, 1 << 30, 1 << 30};
        int hi[3] = {-(1 << 30), -(1 << 30), -(1 << 30)};
        const auto index_of = [&](const FallingCell& c, int k) {
            const double v = k == 0 ? c.x + com.x : k == 1 ? c.y + com.y : c.z + com.z;
            return static_cast<int>(std::lround(v / g.cell - 0.5));
        };
        for (const FallingCell& c : piece.cells) {
            for (int k = 0; k < 3; ++k) {
                lo[k] = std::min(lo[k], index_of(c, k));
                hi[k] = std::max(hi[k], index_of(c, k));
            }
        }
        for (int k = 0; k < 3; ++k) {
            piece.grid[k] = hi[k] - lo[k] + 1;
        }
        // The corner of cell (0, 0, 0) of it, about the centre of mass.
        piece.grid_from = Vec3{lo[0] * g.cell - com.x, lo[1] * g.cell - com.y, lo[2] * g.cell - com.z};
        piece.filled.assign(static_cast<std::size_t>(piece.grid[0]) * piece.grid[1] * piece.grid[2], 0);
        for (const FallingCell& c : piece.cells) {
            const std::size_t at = static_cast<std::size_t>(index_of(c, 0) - lo[0]) + static_cast<std::size_t>(index_of(c, 1) - lo[1]) * piece.grid[0] + static_cast<std::size_t>(index_of(c, 2) - lo[2]) * piece.grid[0] * piece.grid[1];
            piece.filled[at] = 1;
        }
    }
    // Where to feel for the ground: all the wood there is few of, and some of the leaves, which give.
    int every = std::max(1, static_cast<int>(piece.cells.size()) / 160);
    for (std::size_t c = 0; c < piece.cells.size(); ++c) {
        if (static_cast<int>(c) % every == 0 && (!piece.cells[c].leaf || static_cast<int>(c) % (every * 3) == 0)) {
            piece.feel.push_back(static_cast<int>(c));
        }
    }
    // What the law pushes it with acts on all of it, as any force does.
    // (Not if a tool has just cut it: what a heavy blow shook out of the ground is not what brings a tree down.)
    Vec3 impulse = tick_ - last_force_tick_ < 60u && length(last_force_at_ - (lowest + o)) < 8.0 ? Vec3{} : pushed * 0.04;
    Vec3 at = lowest - com;
    piece.velocity = impulse * (1.0 / mass);
    piece.spin = apply(piece.inverse, cross(at, impulse));
    // A blow only cuts. It takes wood from the side it comes from, so the cut opens on that side, what is left under the trunk is
    // on the far side, and it goes over about that: toward where it was struck from, against the force. How fast is the blow
    // over what it has to turn about its foot.
    const Vec3 up_from_foot = com - lowest;
    const double reach = length(up_from_foot);
    if (tick_ - last_force_tick_ < 60u && length(last_force_at_ - (lowest + o)) < 8.0 && reach > 0.1) {
        const Vec3 blow = horizontal_of(last_force_);
        const double hard = length(blow);
        if (hard > 1e-3) {
            const Vec3 away = blow * (-1.0 / hard);
            const double turn = std::clamp(hard * reach / (mass * reach * reach * 0.34), 0.35, 0.9);
            piece.spin = piece.spin + cross(up_from_foot, away) * (turn / reach);
            // What it stands on would take that straight back out of it: it leans into the cut until it has gone too far to stop.
            piece.tip = cross(up_from_foot, away) * (2.0 * turn / reach);
            piece.tip_left = 0.7;
            piece.velocity = piece.velocity + away * (turn * reach);
        }
    }
    // Whatever else, it is never perfectly balanced: its own lean takes it, or a breath of wind.
    const Vec3 lean{com.x - lowest.x, com.y - lowest.y, 0.0};
    if (length(piece.velocity) < 1e-3 && length(lean) < 0.05) {
        const double a = 6.2831853 * std::fmod(static_cast<double>(tick_) * 0.618, 1.0);
        piece.spin = piece.spin + cross(up_from_foot, Vec3{std::cos(a), std::sin(a), 0.0}) * (0.1 / std::max(reach, 0.1));
    }
    // Gone from the fields.
    for (const std::size_t i : trunk) {
        add_to_cell(wood, i, -f.values(wood)[i]);
    }
    for (const std::size_t i : crown) {
        add_to_cell(leaf, i, -f.values(leaf)[i]);
    }
    pieces_.push_back(std::move(piece));
    note("Something comes down.");
}

void Playground::pieces_turn()
{
    if (pieces_.empty()) {
        return;
    }
    const FieldSet& f = fields();
    const Surface s = surface();
    const Vec3 o = origin();
    const double dt = kTick;
    const double radius = f.shape().cell * 0.45;
    std::vector<FallingPiece> kept;
    std::vector<PlayBody*> bodies = all_bodies();
    // In the open world a piece that leaves the lattice sideways lands on the map beyond it (ground_at), not in nothing.
    const GridShape& lattice = f.shape();
    const auto beyond = [&](Vec3 here) {
        return config_.open_world && (here.x < 0.0 || here.y < 0.0 || here.x > lattice.nx * lattice.cell || here.y > lattice.ny * lattice.cell);
    };
    for (FallingPiece& p : pieces_) {
        p.age += dt;
        p.hit_wait = std::max(0.0, p.hit_wait - dt);
        // Contacts are looked at where the piece is, so a tick that carries a cube further than a wall is thick would carry
        // it through. Fast (a thrown ball, not a falling tree), it moves in as many substeps as keep each under a third of
        // a voxel.
        double outermost = 0.0;
        for (const int index : p.feel) {
            const FallingCell& c = p.cells[static_cast<std::size_t>(index)];
            outermost = std::max(outermost, length(Vec3{c.x, c.y, c.z}));
        }
        const double cell = f.shape().cell;
        const double travel = (length(p.velocity) + length(p.spin) * outermost) * dt;
        const int substeps = travel > 0.5 * cell ? std::clamp(static_cast<int>(std::ceil(travel / (0.35 * cell))), 1, 16) : 1;
        const double h = dt / substeps;
        double hardest = 0.0;
        for (int sub = 0; sub < substeps; ++sub) {
            p.velocity.z = std::max(p.velocity.z - config_.gravity * h, -config_.fall_limit);
            if (p.tip_left > 0.0) {
                p.tip_left -= h;
                p.spin = p.spin + p.tip * h;
            }
            // Inverse inertia, as it is turned now.
            const auto inv_apply = [&](Vec3 v) { return apply(p.turned, apply(p.inverse, apply_transposed(p.turned, v))); };
            // Contacts, each with its own impulse, twice round.
            for (int round = 0; round < 2; ++round) {
                for (const int index : p.feel) {
                    const FallingCell& c = p.cells[static_cast<std::size_t>(index)];
                    const Vec3 r = apply(p.turned, Vec3{c.x, c.y, c.z});
                    const Vec3 here = p.at + r - o;
                    const double e = f.shape().cell * 0.5;
                    double d = 0.0;
                    Vec3 n{};
                    if (beyond(here)) {
                        // Out of the window: the map's heights, and their slope.
                        const Vec3 w = here + o;
                        d = w.z - ground_at(w.x, w.y);
                        if (d >= radius) {
                            continue;
                        }
                        n = Vec3{ground_at(w.x - e, w.y) - ground_at(w.x + e, w.y), ground_at(w.x, w.y - e) - ground_at(w.x, w.y + e), 2.0 * e};
                    } else {
                        d = surface_distance(f, s, here);
                        if (d >= radius) {
                            continue;
                        }
                        n = Vec3{surface_distance(f, s, here + Vec3{e, 0.0, 0.0}) - surface_distance(f, s, here - Vec3{e, 0.0, 0.0}),
                            surface_distance(f, s, here + Vec3{0.0, e, 0.0}) - surface_distance(f, s, here - Vec3{0.0, e, 0.0}),
                            surface_distance(f, s, here + Vec3{0.0, 0.0, e}) - surface_distance(f, s, here - Vec3{0.0, 0.0, e})};
                    }
                    n = length(n) > 1e-6 ? normalize(n) : Vec3{0.0, 0.0, 1.0};
                    const Vec3 velocity = p.velocity + cross(p.spin, r);
                    const double into = dot(velocity, n);
                    // It is pushed out a little of what it has gone into.
                    if (round == 0) {
                        p.at = p.at + n * (std::min(radius - d, 0.25) * (c.leaf ? 0.03 : 0.15));
                    }
                    if (into >= 0.0) {
                        continue;
                    }
                    hardest = std::max(hardest, -into);
                    const double scale = c.leaf ? 0.25 : 1.0; // leaves give
                    const double resist = 1.0 / p.mass + dot(n, cross(inv_apply(cross(r, n)), r));
                    const double j = -(1.0 + (c.leaf ? 0.0 : p.bounce)) * into * scale / resist;
                    Vec3 tangent = velocity - n * into;
                    const double slide = length(tangent);
                    Vec3 impulse = n * j;
                    if (slide > 1e-5) {
                        tangent = tangent * (1.0 / slide);
                        const double t_resist = 1.0 / p.mass + dot(tangent, cross(inv_apply(cross(r, tangent)), r));
                        impulse = impulse - tangent * std::min(slide / t_resist, 0.8 * j);
                    }
                    p.velocity = p.velocity + impulse * (1.0 / p.mass);
                    p.spin = p.spin + inv_apply(cross(r, impulse));
                }
            }
            // A little of it goes, always: the air, and everything that creaks. It has fallen by now: what is left is only settling.
            const double settling = p.age > 3.0 ? 4.0 : 0.15;
            p.spin = p.spin * (1.0 - settling * h);
            if (p.age > 3.0) {
                p.velocity = p.velocity * (1.0 - settling * h);
            }
            p.at = p.at + p.velocity * h;
            // Turned by what it spins at (a small rotation, and straightened again).
            const Vec3 w = p.spin * h;
            const double m[3][3] = {{1.0, -w.z, w.y}, {w.z, 1.0, -w.x}, {-w.y, w.x, 1.0}};
            double next[3][3];
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    next[i][j] = m[i][0] * p.turned[0][j] + m[i][1] * p.turned[1][j] + m[i][2] * p.turned[2][j];
                }
            }
            Vec3 a{next[0][0], next[1][0], next[2][0]};
            Vec3 b{next[0][1], next[1][1], next[2][1]};
            a = normalize(a);
            b = normalize(b - a * dot(a, b));
            const Vec3 c = cross(a, b);
            for (int i = 0; i < 3; ++i) {
                p.turned[i][0] = i == 0 ? a.x : i == 1 ? a.y : a.z;
                p.turned[i][1] = i == 0 ? b.x : i == 1 ? b.y : b.z;
                p.turned[i][2] = i == 0 ? c.x : i == 1 ? c.y : c.z;
            }
        }
        if (hardest > 5.0 && !p.landed) {
            p.landed = true;
            event(PlayEvent::Kind::Landing, -1, p.at, p.at, hardest * 1.5);
        }
        // What it falls on.
        if (p.hit_wait <= 0.0) {
            for (PlayBody* body : bodies) {
                if (body->parked) {
                    continue;
                }
                const Vec3 world = body->position + o;
                if (length(world - p.at) > 30.0) {
                    continue;
                }
                for (const int index : p.feel) {
                    const FallingCell& cell = p.cells[static_cast<std::size_t>(index)];
                    if (cell.leaf) {
                        continue;
                    }
                    const Vec3 r = apply(p.turned, Vec3{cell.x, cell.y, cell.z});
                    if (length(world - (p.at + r)) > body->radius + radius + 0.1) {
                        continue;
                    }
                    const Vec3 v = p.velocity + cross(p.spin, r);
                    const double speed = length(v);
                    if (speed > 3.0) {
                        body->hp = std::max(0.0, body->hp - speed * p.crush);
                        body->velocity = body->velocity + v * 0.5 + Vec3{0.0, 0.0, 2.0};
                        body->grounded = false;
                        body->stagger = std::max(body->stagger, 0.5);
                        p.hit_wait = 0.4;
                    }
                    break;
                }
            }
        }
        const double moving = length(p.velocity) + length(p.spin) * 3.0;
        p.quiet = moving < 1.0 ? p.quiet + dt : 0.0;
        if (p.quiet > 0.25 || p.age > 6.0) {
            set_piece_down(p);
        } else {
            kept.push_back(std::move(p));
        }
    }
    pieces_ = std::move(kept);
}

// A body meets what is falling as it meets anything else: the same sphere, against the cells of it, in its own frame.
bool Playground::out_of_pieces(PlayBody& body) const
{
    bool stands = false;
    const Vec3 o = origin();
    const double cell = fields().shape().cell;
    for (const FallingPiece& p : pieces_) {
        const Vec3 off = body.position + o - p.at;
        const Vec3 local = apply_transposed(p.turned, off);
        Vec3 here = local - p.grid_from;
        const int reach = static_cast<int>(std::ceil(body.radius / cell));
        const int cx = static_cast<int>(std::floor(here.x / cell));
        const int cy = static_cast<int>(std::floor(here.y / cell));
        const int cz = static_cast<int>(std::floor(here.z / cell));
        if (cx < -reach || cy < -reach || cz < -reach || cx >= p.grid[0] + reach || cy >= p.grid[1] + reach || cz >= p.grid[2] + reach) {
            continue;
        }
        // The nearest of what it fills, out of what the sphere can reach.
        double nearest = body.radius;
        Vec3 out{};
        for (int z = std::max(cz - reach, 0); z <= std::min(cz + reach, p.grid[2] - 1); ++z) {
            for (int y = std::max(cy - reach, 0); y <= std::min(cy + reach, p.grid[1] - 1); ++y) {
                for (int x = std::max(cx - reach, 0); x <= std::min(cx + reach, p.grid[0] - 1); ++x) {
                    if (p.filled[static_cast<std::size_t>(x) + static_cast<std::size_t>(y) * p.grid[0] + static_cast<std::size_t>(z) * p.grid[0] * p.grid[1]] == 0) {
                        continue;
                    }
                    const Vec3 closest{std::clamp(here.x, x * cell, (x + 1) * cell), std::clamp(here.y, y * cell, (y + 1) * cell), std::clamp(here.z, z * cell, (z + 1) * cell)};
                    const Vec3 away = here - closest;
                    const double d = length(away);
                    if (d < nearest) {
                        nearest = d;
                        out = d > 1e-6 ? away * (1.0 / d) : apply_transposed(p.turned, Vec3{0.0, 0.0, 1.0});
                    }
                }
            }
        }
        if (nearest >= body.radius) {
            continue;
        }
        const Vec3 way = apply(p.turned, out);
        body.position = body.position + way * std::min(body.radius - nearest, 0.2);
        const double into = dot(body.velocity, way);
        if (into < 0.0) {
            body.velocity = body.velocity - way * into;
        }
        stands = stands || way.z > 0.5;
    }
    return stands;
}

// Voxels again, in the cells it lies in. Only what fits: two of it cannot be in one cell, and a few leaves are lost in the fall.
void Playground::set_piece_down(const FallingPiece& p)
{
    const FieldSet& f = fields();
    const GridShape& g = f.shape();
    const Vec3 o = origin();
    const int wood = solver_.field("wood");
    const int leaf = solver_.field("leaf");
    const int solid = solver_.field("solid");
    const int ice = solver_.field("ice");
    const int occ = occ_;
    const int material = solver_.field("material");
    const int held = solver_.field("held");
    // (Two set down in one moment do not see each other in `occ`, which is worked out with the law's next step.)
    if (tick_ - set_down_tick_ > 8u) {
        set_down_.clear();
    }
    set_down_tick_ = tick_;
    std::set<std::size_t>& taken = set_down_;
    std::vector<BeyondCell> beyond;
    for (const FallingCell& c : p.cells) {
        const Vec3 here = p.at + apply(p.turned, Vec3{c.x, c.y, c.z}) - o;
        int x = static_cast<int>(std::floor(here.x / g.cell));
        int y = static_cast<int>(std::floor(here.y / g.cell));
        int z = static_cast<int>(std::floor(here.z / g.cell));
        if (config_.open_world && (x < 0 || y < 0 || x >= g.nx || y >= g.ny || z >= g.nz)) {
            // Out of the window: into the chunk it lies in, as the store keeps it.
            const Vec3 w = here + o;
            beyond.push_back(BeyondCell{.wx = static_cast<int>(std::floor(w.x / g.cell)), .wy = static_cast<int>(std::floor(w.y / g.cell)),
                .wz = static_cast<int>(std::floor(w.z / g.cell)), .matter = static_cast<std::uint8_t>(c.leaf ? 3 : c.matter), .material = c.material});
            continue;
        }
        bool put = false;
        for (int up = 0; up < 3 && !put; ++up) {
            const int zz = z + up;
            if (x < 0 || y < 0 || zz < 0 || x >= g.nx || y >= g.ny || zz >= g.nz) {
                break;
            }
            const std::size_t i = g.index(x, y, zz);
            if (taken.count(i) != 0 || (occ >= 0 && f.values(occ)[i] > 0.5f)) {
                if (c.leaf) {
                    break;
                }
                continue;
            }
            taken.insert(i);
            const int into = c.leaf ? leaf : c.matter == 1 && solid >= 0 ? solid : c.matter == 2 && ice >= 0 ? ice : wood;
            add_to_cell(into, i, 1.0f - f.values(into)[i]);
            if (material >= 0 && c.matter != 2) {
                add_to_cell(material, i, static_cast<float>(c.material) - f.values(material)[i]);
            }
            if (held >= 0) {
                add_to_cell(held, i, 1.0f - f.values(held)[i]);
            }
            put = true;
        }
    }
    set_down_beyond(beyond);
}

// A ball of voxels, thrown. It is a piece like any other: the same contacts, the same setting down, only it was never in the fields.
bool Playground::throw_ball(Vec3 at, Vec3 velocity, double radius, BallKind kind, int material)
{
    if (pieces_.size() >= kMostPieces) {
        return false;
    }
    const double cell = fields().shape().cell;
    // Cells whose centres are within the radius, about the centre of the ball: (i + 0.5) * cell for i in [-reach, reach).
    const int reach = std::max(1, static_cast<int>(std::ceil(radius / cell)));
    const VoxelPalette palette;
    const int stone_index = std::clamp(material, 0, static_cast<int>(palette.stone.size()) - 1);
    const std::array<float, 3> base = kind == BallKind::Stone ? palette.stone[static_cast<std::size_t>(stone_index)]
        : kind == BallKind::Ice                                 ? palette.ice
                                                                : palette.wood;
    FallingPiece piece;
    std::vector<std::array<int, 3>> where;
    for (int k = -reach; k < reach; ++k) {
        for (int j = -reach; j < reach; ++j) {
            for (int i = -reach; i < reach; ++i) {
                const double x = (i + 0.5) * cell;
                const double y = (j + 0.5) * cell;
                const double z = (k + 0.5) * cell;
                if (reach > 1 && x * x + y * y + z * z > radius * radius) {
                    continue;
                }
                // Each cube a shade of its own (a fixed hash of where it is in the ball), so you can see it turn.
                std::uint32_t h = static_cast<std::uint32_t>((i + 64) * 73856093) ^ static_cast<std::uint32_t>((j + 64) * 19349663) ^
                    static_cast<std::uint32_t>((k + 64) * 83492791);
                h ^= h >> 13;
                h *= 0x5bd1e995u;
                const float shade = 0.86f + 0.22f * static_cast<float>(h & 255u) / 255.0f;
                const auto channel = [&](float c) { return static_cast<std::uint32_t>(std::clamp(c * shade, 0.0f, 1.0f) * 255.0f); };
                FallingCell c;
                c.x = static_cast<float>(x);
                c.y = static_cast<float>(y);
                c.z = static_cast<float>(z);
                c.abgr = 0xff000000u | (channel(base[2]) << 16) | (channel(base[1]) << 8) | channel(base[0]);
                c.matter = kind == BallKind::Stone ? 1 : kind == BallKind::Ice ? 2 : 0;
                c.material = static_cast<short>(kind == BallKind::Stone ? stone_index : kind == BallKind::Wood ? 13 : 0);
                piece.cells.push_back(c);
                where.push_back({i, j, k});
            }
        }
    }
    // The centre of mass is the centre of the ball, by symmetry; its moment of inertia is summed over its cubes.
    const double half = cell * 0.5;
    double inertia[3][3] = {};
    for (const FallingCell& c : piece.cells) {
        const Vec3 r{c.x, c.y, c.z};
        const double own = 2.0 * half * half / 3.0;
        inertia[0][0] += r.y * r.y + r.z * r.z + own;
        inertia[1][1] += r.x * r.x + r.z * r.z + own;
        inertia[2][2] += r.x * r.x + r.y * r.y + own;
        inertia[0][1] -= r.x * r.y;
        inertia[0][2] -= r.x * r.z;
        inertia[1][2] -= r.y * r.z;
    }
    inertia[1][0] = inertia[0][1];
    inertia[2][0] = inertia[0][2];
    inertia[2][1] = inertia[1][2];
    invert(inertia, piece.inverse);
    piece.mass = static_cast<double>(piece.cells.size());
    // Which faces look out, and which cubes feel for the ground: those on the outside.
    std::set<std::array<int, 3>> places(where.begin(), where.end());
    for (std::size_t c = 0; c < piece.cells.size(); ++c) {
        const auto [i, j, k] = where[c];
        std::uint8_t faces = 0;
        faces |= places.count({i + 1, j, k}) ? 0 : 1;
        faces |= places.count({i - 1, j, k}) ? 0 : 2;
        faces |= places.count({i, j + 1, k}) ? 0 : 4;
        faces |= places.count({i, j - 1, k}) ? 0 : 8;
        faces |= places.count({i, j, k + 1}) ? 0 : 16;
        faces |= places.count({i, j, k - 1}) ? 0 : 32;
        piece.cells[c].faces = faces;
        if (faces != 0) {
            piece.feel.push_back(static_cast<int>(c));
        }
    }
    // What bodies run into: its own cells, in its own frame.
    for (int k = 0; k < 3; ++k) {
        piece.grid[k] = 2 * reach;
    }
    piece.grid_from = Vec3{-reach * cell, -reach * cell, -reach * cell};
    piece.filled.assign(static_cast<std::size_t>(piece.grid[0]) * piece.grid[1] * piece.grid[2], 0);
    for (const auto& [i, j, k] : where) {
        piece.filled[static_cast<std::size_t>(i + reach) + static_cast<std::size_t>(j + reach) * piece.grid[0] +
            static_cast<std::size_t>(k + reach) * piece.grid[0] * piece.grid[1]] = 1;
    }
    piece.at = at;
    piece.velocity = velocity;
    // Thrown, it turns a little about the way it flies.
    piece.spin = Vec3{-velocity.y, velocity.x, 0.0} * (0.4 / std::max(radius, cell));
    piece.bounce = kind == BallKind::Wood ? 0.3 : kind == BallKind::Ice ? 0.15 : 0.22;
    piece.crush = kind == BallKind::Stone ? 3.0 : 1.5;
    pieces_.push_back(std::move(piece));
    return true;
}

void Playground::mesh_pieces(std::vector<MeshVertex>& out) const
{
    const double half = fields().shape().cell * 0.5;
    static const double kShade[6] = {0.82, 0.62, 0.74, 0.58, 1.0, 0.45};
    static const int kCorners[6][4][3] = {
        {{1, -1, -1}, {1, 1, -1}, {1, 1, 1}, {1, -1, 1}},
        {{-1, 1, -1}, {-1, -1, -1}, {-1, -1, 1}, {-1, 1, 1}},
        {{1, 1, -1}, {-1, 1, -1}, {-1, 1, 1}, {1, 1, 1}},
        {{-1, -1, -1}, {1, -1, -1}, {1, -1, 1}, {-1, -1, 1}},
        {{-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}},
        {{-1, 1, -1}, {1, 1, -1}, {1, -1, -1}, {-1, -1, -1}}};
    for (const FallingPiece& p : pieces_) {
        for (const FallingCell& c : p.cells) {
            for (int side = 0; side < 6; ++side) {
                if ((c.faces & (1u << side)) == 0) {
                    continue;
                }
                Vec3 corner[4];
                for (int k = 0; k < 4; ++k) {
                    const Vec3 local{c.x + kCorners[side][k][0] * half, c.y + kCorners[side][k][1] * half, c.z + kCorners[side][k][2] * half};
                    corner[k] = p.at + apply(p.turned, local);
                }
                const double shade = kShade[side];
                const std::uint32_t r = static_cast<std::uint32_t>((c.abgr & 0xffu) * shade);
                const std::uint32_t g = static_cast<std::uint32_t>(((c.abgr >> 8) & 0xffu) * shade);
                const std::uint32_t b = static_cast<std::uint32_t>(((c.abgr >> 16) & 0xffu) * shade);
                const std::uint32_t colour = 0xff000000u | (b << 16) | (g << 8) | r;
                const auto vertex = [&](const Vec3& v) { out.push_back(MeshVertex{static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), colour}); };
                vertex(corner[0]);
                vertex(corner[1]);
                vertex(corner[2]);
                vertex(corner[0]);
                vertex(corner[2]);
                vertex(corner[3]);
            }
        }
    }
}

} // namespace fieldengine::field
