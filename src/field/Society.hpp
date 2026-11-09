#pragma once

#include "field/Law.hpp"
#include "field/WorldMap.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// THE PEOPLE OF THE WORLD, MOST OF THEM AS NUMBERS. A town is a set of states and the equations of `creature town`
// (assets/law/people.law), one step a month. Nobody in a town is stored:
//   a person is its town and an index. Its name, house, character and work come from a hash of that, and its work and its
//   camp from the town's shares, so the same person comes back the same while the town changes around it;
//   what it knows is hash(person, fact) < how much of the town has heard it. So a rumour is one number a town, and the people
//   who have heard it stay the people who have heard it while it grows.
// What only a real person can have, a blow taken, a thing seen, what it holds against you, is in a record of its own (`Record`).
// A fact is an entry of `facts()` with one wave of its own over the graph of the roads, and a town has room for so many.
namespace fieldengine::field {

struct Fact {
    std::string text;
    int town = -1;          // where it happened
    float weight = 0.5f;    // how much it matters, 0..1
    float about_you = 0.0f; // for you (+) or against you (-)
    int month = 0;
    std::string kind;       // an event's or a deed's name; "" for what the history of the map says
    std::uint64_t who = 0;  // whom it was about, if anyone: pid + 1
    int count = 1;          // of a deed that was done again
    bool local = false;     // it stays with those who were there, and whom they tell: no wave on the roads
    std::string quote;      // of words: what was said, as it was said
    int good = -1;          // of a price: which good
    double value = 0.0;     // and what it was said to be then
};

// SOMEONE WHO CARRIES THINGS TO WHERE THEY HAVE HEARD THEY ARE WANTED. It goes by what it knows, which is the rumours of the place it is
// in and what it was told, and that may be old or false: it gets there and sells for what it finds. Where it passes, what it knows goes too.
struct Caravan {
    std::uint64_t pid = 0;
    int home = -1;
    int index = 0;      // who it is, of its home
    int town = -1;      // where it is or last was
    int next = -1;      // the next town on its way; -1, it is in `town`
    int goal = -1;      // where it means to end up
    double x = 0.0;     // world cells
    double y = 0.0;
    int good = -1;
    double load = 0.0;  // of a town's stock
    double coins = 60.0;
    double paid = 0.0;  // a unit, for what it carries
    double wait = 0.0;  // seconds before it thinks of going
    bool body = false;  // walking about near you: its body says where it is
    bool gone = false;  // lost on the road
    double gone_left = 0.0;
    int trips = 0;
    double earned = 0.0;
};

// Something worth telling someone who is nearby.
struct MarketNews {
    int town = -1;
    std::string text;
};

struct FactRow {
    int fact = -1;
    float now = 0.0f;    // of the town, how much has heard it
    float before = 0.0f; // the last step, for the wave
    float weight = 0.0f; // how much it matters here: less, further from where it happened
};

struct Town {
    int settlement = 0; // index into the map's
    std::string name;
    std::string culture;
    std::vector<double> locals;
    std::vector<char> crossed; // by event: it is past its line, and has been told of
    std::vector<FactRow> rows;
    static constexpr int kBeliefs = 32;
    std::array<float, kBeliefs> beliefs{}; // how many hold what: the opinions of the place, summing to 1
    std::vector<std::pair<int, float>> roads; // town, how good a road
    int wanted = 0;      // bodies, when it is near you
};

struct Person {
    std::uint64_t pid = 0;
    int town = -1;
    int index = 0;
    std::string name;
    int house = 0;
    int good = 0;      // which q_ state it works in
    std::string job;
    double bold = 0.5;
    double temper = 0.5;
    double kind = 0.5;
    double belief = 0.5; // where it stands in the town's opinions
    int camp = 0;        // which of the town's camps that is
};

// What only this person has.
struct Record {
    bool dead = false;
    double own = 0.0;  // what it holds for or against you
    double hp = -1.0;  // below 0: whole
    std::vector<int> saw; // facts it was there for, or was told to its face
    std::vector<int> said; // what it heard you say: it keeps these longer than anything
    std::map<std::uint64_t, double> others; // what it holds against (-) or for (+) someone else, by pid
};

class Society {
public:
    static constexpr int kRows = 32;
    static constexpr std::size_t kSawKept = 32;   // what one person keeps of what it saw and was told
    static constexpr std::size_t kSaidKept = 64;  // and of what it heard you say

    Society(const WorldMap& map, const Law& law, const std::string& words);

    // The law can move with the world that owns it.
    void use(const Law& law) { law_ = &law; }

    // Reads the words; nothing there is fatal, a line with no words is just not said.
    static std::string read_words();
    void set_words(const std::string& text);

    // The law changed: towns keep their states by name.
    void rebind(const Law& law);

    // One tick of play: some of the towns take their month, and every so often the rumours take a step.
    void tick(double dt);

    [[nodiscard]] const std::vector<Town>& towns() const { return towns_; }
    [[nodiscard]] const std::vector<Fact>& facts() const { return facts_; }
    [[nodiscard]] const std::vector<std::string>& goods() const { return goods_; }
    [[nodiscard]] int town_at(const Settlement& settlement) const;
    [[nodiscard]] int month() const { return month_; }
    [[nodiscard]] double state(const Town& town, const char* name) const;
    [[nodiscard]] double share(const Town& town, int good) const;
    [[nodiscard]] double price(const Town& town, int good) const;
    [[nodiscard]] int people_in(const Town& town) const;
    [[nodiscard]] int guards_in(const Town& town) const;
    [[nodiscard]] std::vector<std::pair<float, float>> camps_of(const Town& town) const; // where, how many of them

    [[nodiscard]] Person person(int town, int index) const;
    [[nodiscard]] bool dead(std::uint64_t pid) const;
    [[nodiscard]] const Record* record(std::uint64_t pid) const;
    Record& record_of(std::uint64_t pid);

    [[nodiscard]] bool knows(std::uint64_t pid, int town, int fact) const;
    // What it knows, what matters most first.
    [[nodiscard]] std::vector<int> known(std::uint64_t pid, int town) const;
    // What it thinks of you: -1 to 1. `because` is what weighs most in that, if anything.
    [[nodiscard]] double opinion(const Person& person, int* because = nullptr) const;

    // Something you did. It is known at once to those who were there (`witnesses`) and to a share of the town.
    int deed(const std::string& kind, int town, const std::string& name, double weight, double about_you, const std::vector<std::uint64_t>& witnesses, double town_share, std::uint64_t who = 0, const std::string& other = "");
    // You told a person something. True if that is news to it.
    bool told(const Person& person, int fact);
    // Something that happened here, among them. The words come from the `local` lines of the words.
    int local_fact(const std::string& kind, int town, const std::string& name, const std::string& other, double weight, double about_you, const std::vector<std::uint64_t>& witnesses, std::uint64_t who = 0);
    // Something you said, and who heard it.
    int said_fact(const std::string& words, int town, double about_you, const std::vector<std::uint64_t>& hearers);
    // Is this word on that list (`word LIST ...`)?
    [[nodiscard]] bool word_on(const std::string& list, const std::string& word) const;
    void months_pass(double seconds);
    // ---- the market --------------------------------------------------------------------------------------------------------------
    [[nodiscard]] double stock(const Town& town, int good) const;
    void add_stock(int town, int good, double amount);
    [[nodiscard]] int good_index(const std::string& name) const;
    [[nodiscard]] double unit() const { return param("goods_unit", 0.02); }
    [[nodiscard]] const std::vector<Caravan>& caravans() const { return caravans_; }
    [[nodiscard]] Caravan& caravan(int i) { return caravans_[static_cast<std::size_t>(i)]; }
    [[nodiscard]] bool on_the_road(std::uint64_t pid) const;
    void caravan_lost(int i);
    // What it comes to: someone says a price, true or not, and it is a rumour like any other.
    int price_fact(int good, int town, double value);
    std::vector<MarketNews> take_news();
    // That one died; a town is that many fewer.
    void lost(int town);

    // Words.
    [[nodiscard]] std::string words(const std::string& topic, std::uint64_t seed) const;
    [[nodiscard]] std::string word_of(const std::string& list, const std::string& key) const;
    [[nodiscard]] std::string say(const std::string& topic, const Person& person, std::uint64_t seed, const std::string& fact = "", const std::string& good = "", const std::string& other = "", int count = 0) const;

private:
    void take_month(Town& town);
    void wave_step();
    void beliefs_step(Town& town);
    void bring_land(Town& town, const WorldMap& map, const Settlement& s, const Law& law);
    void add_row(Town& town, int fact, float now, float weight);
    int add_fact(Fact fact);
    int local(const char* name) const;

    const WorldMap* map_;
    const Law* law_;
    int def_ = -1;
    std::vector<std::string> locals_now_;
    std::vector<Town> towns_;
    std::vector<int> town_of_settlement_;
    std::vector<Fact> facts_;
    std::vector<std::string> goods_; // the q_ states, by name
    std::vector<int> q_, p_, res_, cult_;
    std::map<std::uint64_t, Record> records_;
    std::map<std::string, std::vector<std::string>> words_;
    std::map<std::string, std::vector<std::string>> lists_;
    std::vector<std::string> syllables_;
    double owed_months_ = 0.0;
    double wave_left_ = 0.0;
    std::size_t next_town_ = 0;
    int month_ = 0;
    int i_people_ = -1, i_wealth_ = -1, i_camps_ = -1, i_ill_ = -1, i_good_ = -1, i_tolerance_ = -1, i_guard_ = -1;
    double param(const char* name, double fallback) const;
    void make_caravans();
    void caravans_step(double dt);
    void caravan_arrives(Caravan& c);
    void caravan_decides(Caravan& c);
    [[nodiscard]] int hop_toward(int from, int to) const;
    std::vector<Caravan> caravans_;
    std::vector<MarketNews> news_;
    std::vector<int> stock_;
    double caravan_owed_ = 0.0;
    bool caravans_made_ = false;
    double month_seconds_ = 20.0, rumour_every_ = 6.0, rumour_damp_ = 0.3, rumour_speed_ = 0.25, rumour_retell_ = 0.25, rumour_line_ = 0.05, rumour_forget_ = 0.015, rumour_far_ = 0.8, rumour_age_ = 0.0006, belief_mix_ = 0.02;
    void bind();
};

} // namespace fieldengine::field
