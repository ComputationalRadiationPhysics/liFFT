#include "testUtils.hpp"
#include "testCustomTypes.hpp"
#include "testFile.hpp"
#include "testPlainPtr.hpp"

int main(int argc, char** argv) {
    using namespace foobarTest;
    init();
    visualizeBase();

    testCustomTypes();
    testFile();
    testPlainPtr();

    finalize();
    return 0;
}

