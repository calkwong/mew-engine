#include "vk_engine.h"

#include <string>
#include <vector>

int main(int argc, char** argv)
{
	VulkanEngine engine{};

	if (argc < 2)
	{
		return 1;
	}

	std::vector<std::string> file_paths{};
	for (int i = 1; i < argc; i++)
	{
		file_paths.push_back(argv[i]);
	}

	engine.init(file_paths);

	engine.run();

	engine.cleanup();

	return 0;
}