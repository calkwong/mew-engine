#include "vk_engine.h"

int main(int argc, char** argv)
{
	VulkanEngine engine{};

	// only support single gltf file for now
	if (argc != 2)
		return 1; 
	
	std::string file_path = argv[1];

	engine.init(file_path);

	engine.run();

	engine.cleanup();

	return 0;
}