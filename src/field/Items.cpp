#include "field/Items.hpp"

#include "field/WorldMap.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace fieldengine::field {

int Inventory::add(const std::vector<ItemDef>& book, int item, int count)
{
    const int most = book[static_cast<std::size_t>(item)].stack;
    for (ItemStack& s : bag) {
        if (s.item == item && s.count < most) {
            const int put = std::min(count, most - s.count);
            s.count += put;
            count -= put;
            if (count == 0) {
                return 0;
            }
        }
    }
    for (ItemStack& s : bag) {
        if (s.item < 0) {
            const int put = std::min(count, most);
            s = ItemStack{item, put};
            count -= put;
            if (count == 0) {
                return 0;
            }
        }
    }
    return count;
}

int Inventory::count_of(int item) const
{
    int sum = 0;
    for (const ItemStack& s : bag) {
        sum += s.item == item ? s.count : 0;
    }
    return sum;
}

bool Inventory::take(int item, int count)
{
    if (count_of(item) < count) {
        return false;
    }
    for (ItemStack& s : bag) {
        if (s.item == item) {
            const int got = std::min(count, s.count);
            s.count -= got;
            count -= got;
            if (s.count == 0) {
                s = ItemStack{};
            }
            if (count == 0) {
                break;
            }
        }
    }
    return true;
}

int ItemBook::find(const std::string& name) const
{
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (items[i].name == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

ItemBook parse_items(const std::string& text, std::vector<std::string>& errors)
{
    ItemBook book;
    struct Pending {
        std::string item;
        int material = -1;
    };
    std::vector<std::vector<std::string>> later;
    std::istringstream stream(text);
    std::string line;
    int number = 0;
    while (std::getline(stream, line)) {
        ++number;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const std::string where = "items.world:" + std::to_string(number) + ": ";
        std::string body = line.substr(0, line.find('#'));
        std::string label;
        std::string about;
        if (const auto bar = body.find('|'); bar != std::string::npos) {
            label = body.substr(bar + 1);
            body = body.substr(0, bar);
            if (const auto second = label.find('|'); second != std::string::npos) {
                about = label.substr(second + 1);
                label = label.substr(0, second);
            }
        }
        const auto trim = [](std::string& s) {
            s.erase(0, s.find_first_not_of(' '));
            s.erase(s.find_last_not_of(' ') + 1);
        };
        trim(label);
        trim(about);
        std::istringstream in(body);
        std::vector<std::string> w;
        for (std::string word; in >> word;) {
            w.push_back(word);
        }
        if (w.empty()) {
            continue;
        }
        if (w[0] == "drop" || w[0] == "start" || w[0] == "wear" || w[0] == "loot") {
            later.push_back(w);
            continue;
        }
        if (w[0] != "item" || w.size() < 3) {
            errors.push_back(where + "expected item, drop, start or wear");
            continue;
        }
        ItemDef def;
        def.name = w[1];
        def.label = label.empty() ? w[1] : label;
        def.about = about;
        if (w[2] == "weapon") {
            def.kind = ItemKind::Weapon;
            def.stack = 1;
        } else if (w[2] == "food") {
            def.kind = ItemKind::Food;
            def.stack = 10;
        } else if (w[2] == "potion") {
            def.kind = ItemKind::Potion;
            def.stack = 5;
        } else if (w[2] == "resource") {
            def.kind = ItemKind::Resource;
            def.stack = 40;
        } else if (w[2] == "clothes") {
            def.kind = ItemKind::Clothes;
            def.stack = 1;
        } else {
            errors.push_back(where + "no such kind: " + w[2]);
            continue;
        }
        try {
            for (std::size_t k = 3; k + 1 < w.size();) {
                const std::string& key = w[k];
                const std::string& value = w[k + 1];
                if (key == "colour" && k + 3 < w.size()) {
                    def.colour[0] = std::stof(w[k + 1]) / 255.0f;
                    def.colour[1] = std::stof(w[k + 2]) / 255.0f;
                    def.colour[2] = std::stof(w[k + 3]) / 255.0f;
                    k += 4;
                    continue;
                }
                if (key == "effect" && k + 3 < w.size()) {
                    def.effect = w[k + 1];
                    def.effect_amount = std::stod(w[k + 2]);
                    def.effect_time = std::stod(w[k + 3]);
                    k += 4;
                    continue;
                }
                if (key == "fights") {
                    def.weapon = value;
                } else if (key == "heal") {
                    def.heal = std::stod(value);
                } else if (key == "mana") {
                    def.mana = std::stod(value);
                } else if (key == "stack") {
                    def.stack = std::stoi(value);
                } else if (key == "armour") {
                    def.armour = std::stod(value);
                } else if (key == "warmth") {
                    def.warmth = std::stod(value);
                } else if (key == "model") {
                    def.model = value;
                } else if (key == "job") {
                    def.for_job = value;
                } else if (key == "price") {
                    def.price = std::stoi(value);
                } else if (key == "good") {
                    def.good = value;
                } else if (key == "slot") {
                    static const char* const kNames[kWornSlots] = {"head", "body", "legs", "feet", "back", "hand"};
                    for (int s = 0; s < kWornSlots; ++s) {
                        def.slot = value == kNames[s] ? s : def.slot;
                    }
                    if (def.slot < 0) {
                        errors.push_back(where + "no such slot: " + value);
                    }
                } else {
                    errors.push_back(where + "no such thing as '" + key + "'");
                }
                k += 2;
            }
        } catch (...) {
            errors.push_back(where + "a number was expected");
        }
        if (def.kind == ItemKind::Clothes && (def.slot < 0 || def.model.empty())) {
            errors.push_back(where + "clothes need a slot and a model");
            continue;
        }
        if (def.kind == ItemKind::Weapon && def.weapon.empty()) {
            errors.push_back(where + "a weapon needs 'fights WEAPON'");
            continue;
        }
        if (def.kind == ItemKind::Weapon) {
            def.slot = kWornHand;
        }
        book.items.push_back(std::move(def));
    }
    for (const std::vector<std::string>& w : later) {
        if (w[0] == "drop" && w.size() == 3) {
            const int item = book.find(w[2]);
            if (item < 0) {
                errors.push_back("items.world: drop of an unknown item: " + w[2]);
            } else if (w[1] == "wood") {
                book.wood = item;
            } else if (w[1] == "leaf") {
                book.leaf = item;
            } else if (w[1] == "ore") {
                book.ore = item;
            } else {
                const int m = material_id(w[1]);
                if (m < 0) {
                    errors.push_back("items.world: no such material: " + w[1]);
                } else {
                    book.drops[m] = item;
                }
            }
        } else if (w[0] == "loot" && w.size() >= 4 && w.size() % 2 == 0) {
            for (std::size_t k = 2; k + 1 < w.size(); k += 2) {
                const int item = book.find(w[k]);
                if (item < 0) {
                    errors.push_back("items.world: loot of an unknown item: " + w[k]);
                } else {
                    book.loot[w[1]].push_back({item, std::atoi(w[k + 1].c_str())});
                }
            }
        } else if (w[0] == "start" && w.size() == 3) {
            const int item = book.find(w[1]);
            if (item < 0) {
                errors.push_back("items.world: start with an unknown item: " + w[1]);
            } else {
                book.start.push_back({item, std::atoi(w[2].c_str())});
            }
        } else if (w[0] == "wear" && w.size() == 2) {
            const int item = book.find(w[1]);
            if (item < 0) {
                errors.push_back("items.world: wear of an unknown item: " + w[1]);
            } else {
                book.start_worn.push_back(item);
            }
        }
    }
    return book;
}

ItemBook load_items(std::vector<std::string>& errors)
{
    for (const std::string root : {"", "../", "../../"}) {
        std::ifstream in(root + "assets/world/items.world");
        if (in) {
            std::ostringstream all;
            all << in.rdbuf();
            return parse_items(all.str(), errors);
        }
    }
    errors.push_back("assets/world/items.world not found");
    return ItemBook{};
}

} // namespace fieldengine::field
