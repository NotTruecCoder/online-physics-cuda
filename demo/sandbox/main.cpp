#include "sandbox/Sandbox.hpp"

int main(int, char**)
{
    fieldengine::sandbox::Sandbox sandbox;
    if (!sandbox.initialize()) {
        sandbox.shutdown();
        return 1;
    }
    sandbox.run();
    sandbox.shutdown();
    return 0;
}
