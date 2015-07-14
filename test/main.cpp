#include "testUtils.hpp"
#include "testCustomTypes.hpp"
#include "testFile.hpp"
#include "testPlainPtr.hpp"
#include "testZip.hpp"

int main(int argc, char** argv) {
    using namespace foobarTest;
    init();
    visualizeBase();

    testZip();

    testCustomTypes();
    testFile();
    testPlainPtr();

    finalize();
    return 0;
}

