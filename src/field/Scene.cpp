#include "field/Scene.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fieldengine::field {

bool load_scene(const std::string& path, Scene& scene, std::string& error)
{
    std::ifstream in(path);
    if (!in) {
        error = "no scene at " + path;
        return false;
    }
    std::string line;
    int number = 0;
    while (std::getline(in, line)) {
        ++number;
        if (const std::size_t hash = line.find('#'); hash != std::string::npos) {
            line.erase(hash);
        }
        std::istringstream words(line);
        std::string kind;
        if (!(words >> kind)) {
            continue;
        }
        const std::string where = path + ":" + std::to_string(number) + ": ";
        if (kind == "grid") {
            if (!(words >> scene.grid.nx >> scene.grid.ny >> scene.grid.nz >> scene.grid.cell) || scene.grid.nx < 1 || scene.grid.ny < 1 ||
                scene.grid.nz < 1 || scene.grid.cell <= 0.0) {
                error = where + "expected 'grid NX NY NZ METRES'";
                return false;
            }
        } else if (kind == "law") {
            std::string law;
            words >> law;
            scene.laws.push_back(law);
        } else if (kind == "param") {
            std::string name;
            double v = 0.0;
            if (!(words >> name >> v)) {
                error = where + "expected 'param NAME VALUE'";
                return false;
            }
            scene.params.emplace_back(name, v);
        } else if (kind == "fill" || kind == "box" || kind == "ball" || kind == "noise") {
            Scene::Op op;
            op.kind = kind;
            const int count = kind == "fill" ? 0 : (kind == "box" ? 6 : (kind == "ball" ? 4 : 1));
            if (!(words >> op.field >> op.value)) {
                error = where + "expected '" + kind + " FIELD VALUE ...'";
                return false;
            }
            for (int a = 0; a < count; ++a) {
                if (!(words >> op.args[a])) {
                    error = where + kind + " needs " + std::to_string(count) + " numbers after the value";
                    return false;
                }
            }
            scene.ops.push_back(op);
        } else {
            error = where + "unknown line '" + kind + "' (grid, law, param, fill, box, ball, noise)";
            return false;
        }
    }
    return true;
}

bool apply_scene(const Scene& scene, Solver& solver, std::string& error)
{
    for (const auto& [name, value] : scene.params) {
        if (!solver.set_param(name, value)) {
            error = "the law has no param '" + name + "'";
            return false;
        }
    }
    FieldSet& f = solver.fields();
    const GridShape& g = f.shape();
    for (const Scene::Op& op : scene.ops) {
        const int id = f.find(op.field);
        if (id < 0) {
            error = "the law has no field '" + op.field + "'";
            return false;
        }
        std::vector<float>& v = f.values(id);
        for (int z = 0; z < g.nz; ++z) {
            for (int y = 0; y < g.ny; ++y) {
                for (int x = 0; x < g.nx; ++x) {
                    float& cell = v[g.index(x, y, z)];
                    if (op.kind == "fill") {
                        cell = static_cast<float>(op.value);
                    } else if (op.kind == "box") {
                        if (x >= op.args[0] && y >= op.args[1] && z >= op.args[2] && x < op.args[3] && y < op.args[4] && z < op.args[5]) {
                            cell = static_cast<float>(op.value);
                        }
                    } else if (op.kind == "ball") {
                        const double dx = x - op.args[0];
                        const double dy = y - op.args[1];
                        const double dz = z - op.args[2];
                        if (dx * dx + dy * dy + dz * dz <= op.args[3] * op.args[3]) {
                            cell = static_cast<float>(op.value);
                        }
                    } else { // noise
                        std::uint64_t h = static_cast<std::uint64_t>(op.args[0]) * 0x9e3779b97f4a7c15ull ^ g.index(x, y, z) * 0xbf58476d1ce4e5b9ull;
                        h ^= h >> 31;
                        h *= 0x94d049bb133111ebull;
                        h ^= h >> 29;
                        cell += static_cast<float>(op.value * static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0));
                    }
                }
            }
        }
    }
    solver.refresh_derived();
    return true;
}

bool dump_fields(const FieldSet& fields, const std::string& dir, std::string& error)
{
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const GridShape& g = fields.shape();
    std::ofstream meta(dir + "/meta.json");
    if (!meta) {
        error = "cannot write " + dir;
        return false;
    }
    meta << "{\"nx\":" << g.nx << ",\"ny\":" << g.ny << ",\"nz\":" << g.nz << ",\"h\":" << g.cell << ",\"fields\":[";
    for (std::size_t i = 0; i < fields.count(); ++i) {
        const Field& field = fields.at(static_cast<int>(i));
        meta << (i ? "," : "") << '"' << field.name << '"';
        std::ofstream out(dir + "/" + field.name + ".bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(field.values.data()), static_cast<std::streamsize>(field.values.size() * 4));
        if (!out.flush()) {
            error = "cannot write " + dir + "/" + field.name + ".bin (disk full?)";
            return false;
        }
    }
    meta << "]}\n";
    return true;
}

bool load_fields(FieldSet& fields, const std::string& dir, std::string& error)
{
    if (!std::filesystem::is_directory(dir)) {
        error = "no state at " + dir;
        return false;
    }
    for (std::size_t i = 0; i < fields.count(); ++i) {
        std::vector<float>& values = fields.values(static_cast<int>(i));
        const std::string path = dir + "/" + fields.at(static_cast<int>(i)).name + ".bin";
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in) {
            continue;
        }
        const std::streamsize bytes = in.tellg();
        if (bytes != static_cast<std::streamsize>(values.size() * 4)) {
            error = path + " holds " + std::to_string(bytes / 4) + " floats, the lattice " + std::to_string(values.size());
            return false;
        }
        in.seekg(0);
        in.read(reinterpret_cast<char*>(values.data()), bytes);
    }
    return true;
}

} // namespace fieldengine::field
