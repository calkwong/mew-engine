#pragma once

#include "common.h"
#include "vk_math.h"
#include "glm/ext/vector_uint2.hpp"

#include <array>
#include <cstdint>

struct CompactDispatchPushConstants
{
    VkDeviceAddress prefix_sum_buffer{};
    VkDeviceAddress dispatch_buffer{};
};

struct IBLPushConstants
{
    glm::vec2 image_size{};
    uint32_t texture_id{};
    uint32_t image_id{};
    float roughness{};
};

struct LuminanceBinsPushConstants
{
    VkDeviceAddress luminance_buffer{};
    VkDeviceAddress luminance_avg_buffer{};
    glm::vec2 screen_size{};
    uint32_t image_id{};
    float min_log_luminance{};
    float one_over_log_luminance_range{};
    uint32_t pixel_count{};
    float tau{};
    float delta_time{};
};

struct TonemapPushConstants
{
    VkDeviceAddress luminance_avg_buffer{};
    glm::vec2 screen_size{};
    uint32_t src_id{};
    uint32_t dst_id{};
    uint32_t autoexposure{};
    uint32_t tonemap_func{};
};

struct SHPushConstants
{
    VkDeviceAddress sh_buffer_address{};
    uint32_t cubemap_id{};
};

struct GPUPushConstants
{
    VkDeviceAddress object_buffer_address{};
    VkDeviceAddress vertex_buffer_address{};
    VkDeviceAddress meshlet_buffer_address{};
    VkDeviceAddress meshlet_indices_buffer_address{};
    VkDeviceAddress cluster_indices_address{};
    VkDeviceAddress material_buffer_address{};
    VkDeviceAddress prefix_sum_buffer{};
    glm::uvec2 screen_size{};
    glm::vec4 jitter_offset{};
};

struct OITPushConstants
{
    VkDeviceAddress object_buffer_address{};
    VkDeviceAddress vertex_buffer_address{};
    VkDeviceAddress meshlet_buffer_address{};
    VkDeviceAddress meshlet_indices_buffer_address{};
    VkDeviceAddress cluster_indices_address{};
    VkDeviceAddress material_buffer_address{};
    VkDeviceAddress prefix_sum_buffer{};
    VkDeviceAddress sh_buffer{};
    glm::uvec2 screen_size{};
    float max_prefiltered_lod{};
    uint32_t framebuffer_id{};
    uint32_t volume{};
};

struct DeferredPushConstants
{
    glm::vec4 cluster_size{}; // xyz are cluster data structure dimensions, w is a single cluster's dimension
    glm::vec2 screen_size{};
    VkDeviceAddress light_buffer_address{};
    VkDeviceAddress light_index_buffer_address{};
    VkDeviceAddress light_grid_buffer_address{};
    VkDeviceAddress meshlet_indices_address{}; // TESTING FOR VIS BUFFER ONLY
    VkDeviceAddress meshlet_buffer_address{}; // TESTING FOR VIS BUFFER ONLY
    VkDeviceAddress vertex_buffer_address{}; // TESTING FOR VIS BUFFER ONLY
    VkDeviceAddress object_buffer_address{}; // TESTING FOR VIS BUFFER ONLY
    VkDeviceAddress material_buffer_address{}; // TESTING FOR VIS BUFFER ONLY
    VkDeviceAddress index_buffer_address{};
    VkDeviceAddress mesh_buffer_address{};
    VkDeviceAddress sh_buffer_address{};
    uint32_t draw_id;
    uint32_t depth_id{};
    uint32_t gbuffer_id{};
    uint32_t shadow_id{};
    uint32_t light_culling{}; // for toggling light culling between naive and proper implementation
    float near{};
    float scale{};
    float bias{};
    uint32_t shadows{};
    uint32_t shadows_rt{};
    float max_prefiltered_lod{};
    uint32_t debug{};
};

// rasterize shadows
struct ShadowPushConstants
{
    glm::mat4 viewproj{};
    VkDeviceAddress material_buffer_address{};
    VkDeviceAddress object_buffer_address{};
    VkDeviceAddress vertex_buffer_address{};
};

struct ShadowCullPushConstants
{
    VkDeviceAddress object_buffer_address{};
    VkDeviceAddress mesh_buffer_address{};
    VkDeviceAddress indices_buffer_address{};
    VkDeviceAddress draw_buffer_address{};
    uint32_t count{};
    uint32_t lod_enabled{};
};

struct SkyboxPushConstants
{
    glm::mat4 inverse_viewproj{};
    uint32_t texture_id{};
};

struct TAAPushConstants
{
    glm::vec4 jitter_offset{};
    glm::vec2 screen_size{};
    uint32_t current_id{};
    uint32_t history_id{};
    uint32_t resolve_id{};
    uint32_t depth_id{};
    uint32_t velocity_id{};
    uint32_t variance_clipping{};
    uint32_t history_filter{};
    uint32_t local_filter{};
    uint32_t ycocg{};
    uint32_t valid_history{};
    uint32_t dynamic{};
};

struct SpdPushConstants
{
    VkDeviceAddress spd_counter_buffer{};
    glm::vec2 rcp_resolution{};
    uint32_t mips{};
    uint32_t num_wgs{};
    uint32_t src_id{}; // texture to sample
    uint32_t dst_id{}; // image to write to, offset accordingly!
    uint32_t sampler_id{};
    // uint32_t wg_offset; // note: for subregion downsampling, not implemented for now
};

struct DepthPyramidPushConstants
{
    std::array<int32_t, 2> image_size{};
    uint32_t texture_id{};
    uint32_t image_id{};
    uint32_t lod{};
};

struct ClusterGridPushConstants
{
    glm::mat4 inverse_proj{};
    VkDeviceAddress light_cluster_buffer_address{};
    glm::vec2 screen_size{};
    glm::vec2 cluster_dim{};
    float near{};
    float far{};
    uint32_t depth_slices{};
};

struct LightCullingPushConstants
{
    glm::mat4 view{};
    glm::mat4 light_rot{};
    VkDeviceAddress light_cluster_buffer_address{};
    VkDeviceAddress light_buffer_address{};
    VkDeviceAddress light_index_buffer_address{};
    VkDeviceAddress light_grid_buffer_address{};
    VkDeviceAddress light_count_buffer_address{};
};

struct CullData
{
    glm::mat4 view{};
    glm::vec4 frustum_planes{};
    VkDeviceAddress object_buffer_address{};
    VkDeviceAddress mesh_buffer_address{};
    VkDeviceAddress indices_buffer_address{};
    VkDeviceAddress draw_indirect_address{};
    VkDeviceAddress dispatch_buffer_address{};
    VkDeviceAddress vis_buffer_address{};
    VkDeviceAddress prefix_sum_buffer{};
    uint32_t count{};
    uint32_t late{};
    uint32_t texture_id{};
    uint32_t occlusion_enabled{};

    float p00{};
    float p11{};
    float near{};
    float far{};

    glm::vec2 resolution{};
    float texture_lod{};
    float lod_distance_factor{};
    uint32_t lod_enabled{};
    uint32_t task_submit{};
    uint32_t post_pass{};
};

struct ClusterCullData
{
    glm::mat4 view{};
    glm::vec4 frustum_planes{};
    VkDeviceAddress object_buffer_address{};
    VkDeviceAddress meshlet_buffer_address{};
    VkDeviceAddress cluster_indices_address{};
    VkDeviceAddress meshlet_dispatch_address{};
    VkDeviceAddress cluster_vis_address{};
    VkDeviceAddress prefix_sum_buffer{};
    uint32_t count{};
    uint32_t late{};
    uint32_t texture_id{};
    uint32_t occlusion_enabled{};

    float p00{};
    float p11{};
    float near{};
    float far{};

    glm::vec2 resolution{};
    float texture_lod{};
    float lod_distance_factor{};
    uint32_t lod_enabled{};
    uint32_t task_submit{};
    uint32_t post_pass{};
};
