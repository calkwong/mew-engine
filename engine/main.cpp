#include "vk_engine.h"

int main(int argc, char** argv)
{
    VulkanEngine engine{};

    if (argc < 2)
    {
        return 1;
    }

    engine.init(argc, argv);

    engine.run();

    engine.cleanup();

    return 0;
}
