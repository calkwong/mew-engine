#pragma once

#include "math.h"

#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vk_mem_alloc.h>
#include <fmt/core.h>

#include <vector>
#include <array>

#define VK_CHECK(x)                                                     \
    do {                                                                \
        VkResult err = x;                                               \
        if (err) {                                                      \
            fmt::println("Detected Vulkan error: {}", string_VkResult(err)); \
            abort();                                                    \
        }                                                               \
    } while (0)

struct AllocatedImage
{
    VkImage image{};
    VkImageView view{};
    VmaAllocation allocation{};
    VkExtent3D extent{};
    VkFormat format{};
};

struct AllocatedBuffer
{
    VkBuffer buffer{};
    VmaAllocation allocation{};
    VmaAllocationInfo info{};
};

struct Vertex
{
    glm::vec3 position{};
    float uv_x{};
    glm::vec3 normal{};
    float uv_y{};
    glm::vec4 tangent{};
};

struct GPUMeshBuffers
{
    AllocatedBuffer index_buffer{};
    AllocatedBuffer vertex_buffer{};
    VkDeviceAddress vertex_buffer_address{};
};

enum class MaterialPass : uint32_t
{
    Opaque,
    Mask,
    Blend
};

struct Bounds
{
    glm::vec3 origin{};
    float radius{};
};

struct MeshLod
{
    uint32_t first_index{};
    uint32_t count{};
    float error{};
    uint32_t meshlet_offset{};
    uint32_t meshlet_count{};
};

struct alignas(16) Meshlet
{
    glm::vec3 center{};
    float radius{};
    uint32_t data_offset{}; // aka index first count
    uint32_t vertex_count{};
    uint32_t triangle_count{};
};

struct MaterialData
{
    glm::vec4 base_color_factor{ glm::vec4(1.0f) };
    float metallic_factor{ 1.0f };
    float roughness_factor{ 1.0f };
    uint32_t diffuse_id{};
    uint32_t metal_roughness_id{};
    uint32_t normal_id{};
    uint32_t occlusion_id{};
    uint32_t emissive_id{};
    uint32_t padding{}; 
};

struct SceneData
{
    glm::mat4 view{};
    glm::mat4 proj{};
    glm::mat4 viewproj{};
    std::array<glm::mat4, 4> shadow_transforms{};
    glm::vec4 cascade_splits{}; 
    glm::vec4 camera_pos{};
    glm::vec4 sunlight_color{};
    glm::vec4 sunlight_dir{};
    glm::vec4 textures{}; // irradiance, prefiltered, brdf, shadow
};

struct CascadeData
{
    AllocatedImage shadow_map{};
    glm::mat4 viewproj{};
    float split_ratio{};
};

template <typename T>
struct Handle
{
    uint32_t handle{};
};