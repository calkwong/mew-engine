#pragma once

#include "vk_math.h"
#include "resources.h"

#include <array>
#include <vector>

struct Material; // TODO: not declared/anywhere
struct ShaderPass;

struct Vertex
{
    uint16_t px{};
    uint16_t py{};
    uint16_t pz{};

    uint16_t tangent{};

    uint32_t normal{};

    uint16_t uv_x{};
    uint16_t uv_y{};
};

enum class MaterialPass : uint32_t
{
    Opaque,
    Mask,
    Blend
};

struct MeshLod
{
    uint32_t first_index{};
    uint32_t count{};
    float error{};
    uint32_t meshlet_offset{};
    uint32_t meshlet_count{};
};

struct Meshlet
{
    uint16_t cx{};
    uint16_t cy{};
    uint16_t cz{};
    uint16_t radius{};

    // int8_t cone_axis[3]{};
    // int8_t cone_cutoff{};

    uint32_t data_offset{}; // aka index first count

    uint8_t vertex_count{};
    uint8_t triangle_count{};
};

struct MaterialData
{
    glm::vec4 base_color_factor{ glm::vec4(1.0f) };
    float metallic_factor{ 1.0f };
    float roughness_factor{ 1.0f };
    uint32_t diffuse_id{};
    uint32_t metal_roughness_id{};
    uint32_t normal_id{};
    glm::vec3 emissive_factor{};
    uint32_t emissive_id{};
    uint32_t occlusion_id{};
};

struct SceneData
{
    glm::mat4 view{};
    glm::mat4 proj{};
    glm::mat4 viewproj{};
    glm::mat4 inverse_viewproj{};
    glm::mat4 previous_viewproj{};
    glm::mat4 light_rot{};
    std::array<glm::mat4, 4> shadow_transforms{};
    glm::vec4 cascade_splits{};
    glm::vec4 camera_pos{};
    glm::vec4 sunlight_color{};
    glm::vec4 sunlight_dir{};
    glm::vec4 textures{}; // cubemap/skybox, irradiance, prefiltered, brdf
    std::array<glm::mat4, 4> shadow_views{}; // for shadow_cull
    std::array<float, 4> shadow_widths{}; // TODO: move to pc?
};

struct CascadeData
{
    AllocatedImage shadow_map{};
    glm::mat4 viewproj{};
    float split_ratio{};
};

struct PointLight
{
    glm::vec4 pos{}; // pos & radius
    glm::vec4 color{};
};

struct ClusterAABB
{
    glm::vec4 min{};
    glm::vec4 max{};
};

struct LightGrid
{
    uint32_t offset{};
    uint32_t count{};
};

struct OITData
{
    glm::uvec4 colors{};
    glm::uvec4 depths{};
    glm::vec4 transmissions{};
};

struct Mesh
{
    glm::vec3 center{};
    float radius{};

    uint32_t lod_count{};
    uint32_t vertex_offset{};

    std::array<MeshLod, 8> mesh_lods{};
};

struct RenderObject
{
    glm::vec3 translation{};
    float scale{};
    glm::quat orientation{};

    uint32_t mesh_id{};
    uint32_t material_id{};
    uint32_t meshlet_bits{};
    uint32_t post_pass{};
};

struct ObjectData
{
    glm::vec3 translation{};
    float scale{};
    glm::quat orientation{};

    uint32_t mesh_id{};
    uint32_t material_id{};
    uint32_t meshlet_bit_offset{};
    uint32_t post_pass{};
};

struct PrefixSumData
{
    uint32_t instance_id{};
    uint32_t sum{};
    uint32_t lod_offset{};
};

struct PrefixSum
{
    uint64_t count{};
    PrefixSumData* data;
};

struct MeshAsset;

struct RenderScene
{
    enum class MeshPassType
    {
        Opaque,
        Mask,
        Transparent
    };

    struct MeshPass
    {
        std::vector<uint32_t> unbatched_objects{}; // handles for renderables
        uint32_t indices_offset{};
        MeshPassType type{};
    };

    std::vector<RenderObject> renderables{};
    std::vector<Mesh> meshes{};
    std::unordered_map<MeshAsset*, uint32_t> mesh_cache{};

    AllocatedBuffer vertex_buffer{};
    AllocatedBuffer index_buffer{};
    AllocatedBuffer indices_buffer{}; // an indirection buffer - for indexing into the right RenderObject
    AllocatedBuffer object_buffer{};
    AllocatedBuffer mesh_buffer{};
    AllocatedBuffer meshlet_buffer{};
    AllocatedBuffer meshlet_indices{};
    AllocatedBuffer material_buffer{};

    AllocatedBuffer draw_indirect_buffer{};
    AllocatedBuffer dispatch_buffer{};
    AllocatedBuffer vis_buffer{};
    AllocatedBuffer meshlet_vis_buffer{};

    AllocatedBuffer meshlet_dispatch_buffer{};
    AllocatedBuffer cluster_indices{};
    AllocatedBuffer oit_buffer{};

    AllocatedBuffer sh_buffer{};
    AllocatedBuffer luminance_buffer{};
    AllocatedBuffer luminance_avg_buffer{};

    AllocatedBuffer prefix_sum_buffer{};

    AllocatedBuffer spd_counter_buffer{};

    MeshPass opaque_pass{};
    MeshPass mask_pass{};
    MeshPass transparent_pass{};
    uint32_t total_meshlets_bits{};
};
