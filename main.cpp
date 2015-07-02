#include "testUtils.hpp"
#include "testCustomTypes.hpp"
#include "testFile.hpp"

int main(int argc, char** argv) {
    using namespace foobarTest;
    init();
    visualizeBase();

    testCustomTypes();
    testFile();

    finalize();
    return 0;
}

