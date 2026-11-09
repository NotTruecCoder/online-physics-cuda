#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// THE WORLD IS A SET OF FIELDS ON ONE VOXEL LATTICE.
//
// Physics, light, collision, navigation and perception all read the same cells, so the shape of the world
// is kept in exactly one place: a field. Geometry is the `solid` field; the distance to it is derived from it;
// the picture is meshed from it; a body is pushed out along it.
namespace fieldengine::field {

struct GridShape {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    double cell = 1.0; // metres per voxel edge

    [[nodiscard]] std::size_t cells() const
    {
        return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    }
    [[nodiscard]] std::size_t index(int x, int y, int z) const
    {
        return (static_cast<std::size_t>(z) * static_cast<std::size_t>(ny) + static_cast<std::size_t>(y)) *
                static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(x);
    }
    [[nodiscard]] bool inside(int x, int y, int z) const
    {
        return x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz;
    }
};

// A named scalar field. Storage is float: the lattice is the memory budget, and determinism comes
// from a fixed evaluation order, not from width.
struct Field {
    std::string name;
    std::vector<float> values;
};

// Every field on one lattice, addressed by a dense id (its declaration order — like the property
// manifest, ids are positions).
class FieldSet {
public:
    FieldSet() = default;
    explicit FieldSet(GridShape shape) : shape_(shape) {}

    [[nodiscard]] const GridShape& shape() const { return shape_; }

    // Adds a field filled with `initial`, or returns the existing id.
    int add(const std::string& name, float initial = 0.0f);
    [[nodiscard]] int find(const std::string& name) const; // -1 if absent
    [[nodiscard]] std::size_t count() const { return fields_.size(); }

    [[nodiscard]] Field& at(int id) { return fields_[static_cast<std::size_t>(id)]; }
    [[nodiscard]] const Field& at(int id) const { return fields_[static_cast<std::size_t>(id)]; }
    [[nodiscard]] std::vector<float>& values(int id) { return at(id).values; }
    [[nodiscard]] const std::vector<float>& values(int id) const { return at(id).values; }

    // Clamped lattice read (zero-flux boundary: the edge cell repeats).
    [[nodiscard]] float get_clamped(int id, int x, int y, int z) const;
    // Trilinear sample at a position in metres. This is what a body, a ray or a sensor reads.
    [[nodiscard]] float sample(int id, double px, double py, double pz) const;

    // FNV-1a over every field's bits in declaration order. Two runs of the same equations from the
    // same start must agree on this to the bit.
    [[nodiscard]] std::uint64_t digest() const;

private:
    GridShape shape_{};
    std::vector<Field> fields_;
};

} // namespace fieldengine::field
