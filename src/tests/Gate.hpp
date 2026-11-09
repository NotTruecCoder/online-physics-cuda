#pragma once

#include <iostream>
#include <string>

// THE SKELETON EVERY GATE IN THIS PROJECT REPEATS.
//
// The convention a gate is expected to follow, and why:
//
//   * NAME THE PROPERTY, NOT THE CODE. "a body shoved into stone leaves through the wall's side"
//     is a claim about the world that survives a refactor; "resolve_solid_overlaps returns true"
//     is a claim about a function that will not.
//
//   * SAY WHAT WOULD FALSIFY IT. Every gate here carries a comment block listing the mutations that
//     make it fail, and those mutations are actually run once before the gate is trusted. A check
//     that has never been seen to fail is a check nobody has tested. Two of this project's gates
//     passed their own falsification on the first attempt — both were measuring something adjacent
//     to what they claimed, and only the mutant showed it.
//
//   * DO NOT DERIVE THE CHECK FROM ITS SUBJECT. A roll call written out by hand catches a system
//     that vanished; a roll call generated from the system catches nothing. This bit us twice.
namespace fieldengine::gate {

inline int& failures()
{
    static int count = 0;
    return count;
}

inline void check(bool condition, const std::string& label)
{
    if (!condition) {
        std::cerr << "FAIL: " << label << "\n";
        ++failures();
    }
}

// Standard tail of a gate's main(): prints the verdict and returns the process exit code.
inline int report(const std::string& gate_name)
{
    if (failures() != 0) {
        std::cerr << failures() << " " << gate_name << " check(s) failed.\n";
        return 1;
    }
    std::cout << gate_name << " smoke ok\n";
    return 0;
}

} // namespace fieldengine::gate
