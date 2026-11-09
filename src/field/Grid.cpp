#include "field/Grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace fieldengine::field {

int FieldSet::add(const std::string& name, float initial)
{
    if (const int existing = find(name); existing >= 0) {
        return existing;
    }
    fields_.push_back(Field{.name = name, .values = std::vector<float>(shape_.cells(), initial)});
    return static_cast<int>(fields_.size()) - 1;
}

int FieldSet::find(const std::string& name) const
{
    for (std::size_t i = 0; i < fields_.size(); ++i) {
        if (fields_[i].name == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

float FieldSet::get_clamped(int id, int x, int y, int z) const
{
    x = std::clamp(x, 0, shape_.nx - 1);
    y = std::clamp(y, 0, shape_.ny - 1);
    z = std::clamp(z, 0, shape_.nz - 1);
    return values(id)[shape_.index(x, y, z)];
}

float FieldSet::sample(int id, double px, double py, double pz) const
{
    // Cell centres sit at (i + 0.5) * cell.
    const double gx = px / shape_.cell - 0.5;
    const double gy = py / shape_.cell - 0.5;
    const double gz = pz / shape_.cell - 0.5;
    const int x0 = static_cast<int>(std::floor(gx));
    const int y0 = static_cast<int>(std::floor(gy));
    const int z0 = static_cast<int>(std::floor(gz));
    const double fx = gx - x0;
    const double fy = gy - y0;
    const double fz = gz - z0;
    auto v = [&](int dx, int dy, int dz) {
        return static_cast<double>(get_clamped(id, x0 + dx, y0 + dy, z0 + dz));
    };
    const double c00 = v(0, 0, 0) * (1.0 - fx) + v(1, 0, 0) * fx;
    const double c10 = v(0, 1, 0) * (1.0 - fx) + v(1, 1, 0) * fx;
    const double c01 = v(0, 0, 1) * (1.0 - fx) + v(1, 0, 1) * fx;
    const double c11 = v(0, 1, 1) * (1.0 - fx) + v(1, 1, 1) * fx;
    const double c0 = c00 * (1.0 - fy) + c10 * fy;
    const double c1 = c01 * (1.0 - fy) + c11 * fy;
    return static_cast<float>(c0 * (1.0 - fz) + c1 * fz);
}

std::uint64_t FieldSet::digest() const
{
    std::uint64_t hash = 1469598103934665603ull;
    auto mix = [&](const void* data, std::size_t bytes) {
        const auto* p = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < bytes; ++i) {
            hash ^= p[i];
            hash *= 1099511628211ull;
        }
    };
    for (const Field& field : fields_) {
        mix(field.name.data(), field.name.size());
        mix(field.values.data(), field.values.size() * sizeof(float));
    }
    return hash;
}

} // namespace fieldengine::field
