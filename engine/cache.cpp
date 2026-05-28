#include "cache.h"
#include "common.h"
#include "pipelines.h"

#include <filesystem>
#include <memory>
#include <string>

ShaderProgram* ShaderCache::operator[](const std::string& key)
{
    return data[key].get();
}

void ShaderCache::add_shader(VkDevice device, const char* path)
{
    const auto it = data.find(path);

    std::string shader_path{ "shaders/compiled/" };
    shader_path += path;
    shader_path += ".spv";

    if (it == data.end())
    {
        VkShaderModule module{};
        load_shader_module(shader_path.c_str(), device, &module);
        auto time = std::filesystem::last_write_time(shader_path);
        data[path] = std::make_unique<ShaderProgram>(module, path, time);
    }
}
