#pragma once

#include "field/Body.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// WHAT THERE IS TO HOLD, AS TEXT (assets/world/items.world). An item is a line; nothing in the engine knows a sword from a loaf
// but what that line says it does.
namespace fieldengine::field {

enum class ItemKind : std::uint8_t { Weapon, Food, Potion, Resource, Clothes };
enum WornSlot { kWornHead, kWornBody, kWornLegs, kWornFeet, kWornBack, kWornHand, kWornSlots };

struct ItemDef {
    std::string name;
    std::string label;
    std::string about;
    ItemKind kind = ItemKind::Resource;
    int stack = 20;
    float colour[3] = {0.7f, 0.7f, 0.7f};
    std::string weapon;      // the weapon of moves.world it fights as
    double heal = 0.0;       // HP at once
    double mana = 0.0;
    std::string effect;      // speed strength stone regen flow leap: for `effect_time` seconds
    double effect_amount = 0.0;
    double effect_time = 0.0;
    int slot = -1;           // WornSlot, of clothes
    std::string model;       // assets/world/clothes/NAME.model
    double armour = 0.0;
    double warmth = 0.0;     // -1 for the heat, +1 for the cold: who wears it where
    std::string for_job;     // who gets it, in a town
    int price = 0;           // in coins; 0, worked out from what it does
    std::string good;        // the q_ state of people.law whose price moves it
};

struct ItemStack {
    int item = -1;
    int count = 0;
};

// What someone has.
struct Inventory {
    static constexpr int kBag = 30;
    std::array<ItemStack, kBag> bag{};
    std::array<int, kWornSlots> worn{-1, -1, -1, -1, -1, -1};
    // What fits goes in, what does not is the answer.
    int add(const std::vector<ItemDef>& book, int item, int count);
    int count_of(int item) const;
    bool take(int item, int count);
};

// An item lying about.
struct WorldItem {
    int item = 0;
    int count = 1;
    Vec3 at{};       // world metres
    Vec3 velocity{};
    double age = 0.0;
    bool lying = false;
};

// An effect on you, ticking down.
struct Effect {
    std::string name;
    double amount = 0.0;
    double left = 0.0;
};

struct ItemBook {
    std::vector<ItemDef> items;
    std::map<int, int> drops;  // material index -> item
    int wood = -1;
    int leaf = -1;
    int ore = -1;
    std::vector<std::pair<int, int>> start;    // what you begin with
    std::vector<int> start_worn;
    std::map<std::string, std::vector<std::pair<int, int>>> loot; // creature -> what comes off it
    int coin = -1;
    [[nodiscard]] int find(const std::string& name) const;
};

ItemBook parse_items(const std::string& text, std::vector<std::string>& errors);
ItemBook load_items(std::vector<std::string>& errors);

} // namespace fieldengine::field
