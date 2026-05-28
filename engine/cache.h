#pragma once

#include "common.h"
#include "pipelines.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

struct ShaderCache
{
    std::unordered_map<std::string, std::unique_ptr<ShaderProgram>> data{};

    ShaderProgram* operator[](const std::string& key);

    void add_shader(VkDevice device, const char* path);
};

struct Bindless
{
    uint32_t depth_pyramid_uav{};
    uint32_t skybox_uav{};
    uint32_t draw_uav{};
    uint32_t accum_uav{};
    uint32_t irradiance_uav{};
    uint32_t prefiltered_uav{};
    uint32_t brdf_uav{};

    uint32_t draw_srv{};
    uint32_t gbuffer_srv{};
    uint32_t vbuffer_srv{};
    uint32_t depth_srv{};
    uint32_t depth_pyramid_srv{};
    uint32_t shadowmap_srv{};
    uint32_t accum_srv{};
    uint32_t skybox_srv{};
    uint32_t hdri_srv{};
    uint32_t irradiance_srv{};
    uint32_t prefiltered_srv{};
    uint32_t brdf_srv{};

    uint32_t oit{};
};
