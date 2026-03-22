#include "vk_engine.h"

#include <string>
#include <vector>

int main(int argc, char** argv)
{
	VulkanEngine engine{};

	if (argc != 2)
	{
		return 1;
	}

	engine.init(argv[1]);

	engine.run();

	engine.cleanup();

	return 0;
}