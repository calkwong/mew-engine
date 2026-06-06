#pragma once

#include "common.h"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

struct BDATable
{
    VkDeviceAddress light_buffer{};
    VkDeviceAddress light_cluster_buffer{};
    VkDeviceAddress light_index_buffer{};
    VkDeviceAddress light_grid_buffer{};
    VkDeviceAddress light_count_buffer{};
    VkDeviceAddress vertex_buffer{};
    VkDeviceAddress index_buffer{};
    VkDeviceAddress indices_buffer{};
    VkDeviceAddress object_buffer{};
    VkDeviceAddress mesh_buffer{};
    VkDeviceAddress meshlet_buffer{};
    VkDeviceAddress meshlet_indices{};
    VkDeviceAddress material_buffer{};
    VkDeviceAddress draw_indirect_buffer{};
    VkDeviceAddress dispatch_buffer{};
    VkDeviceAddress vis_buffer{};
    VkDeviceAddress meshlet_vis_buffer{};
    VkDeviceAddress meshlet_dispatch_buffer{};
    VkDeviceAddress cluster_indices{};
    VkDeviceAddress oit_buffer{};
    VkDeviceAddress sh_buffer{};
    VkDeviceAddress luminance_buffer{};
    VkDeviceAddress luminance_avg_buffer{};
    VkDeviceAddress prefix_sum_buffer{};
    VkDeviceAddress spd_counter_buffer{};
};

class RenderGraph;

struct Pass
{
public:
    uint32_t get_resource_index(const std::string& name, VkImage image = VK_NULL_HANDLE);

    void add_depth_stencil_output(const std::string& name, VkImage image);
    void add_color_output(const std::string& name, VkImage image);
    void add_image_read(const std::string& name, VkImage image);
    void add_image_write(const std::string& name, VkImage image);
    void add_storage_buffer_read(const std::string& name);
    void add_storage_buffer_write(const std::string& name);
    void add_indirect_buffer_read(const std::string& name);

    enum PassType
    {
        GraphicsPass,
        ComputePass,
    };

    struct Barrier
    {
        uint32_t index{};
        VkPipelineStageFlags2 stages{};
    };

    RenderGraph* graph{};
    std::string name{};
    PassType pass_type{};
    std::function<void()> callback{};

    std::vector<Barrier> flushes{};
    std::vector<Barrier> invalidates{};
};

// Rendergraph is intentionally simple and dumb. It performs UNDEFINED -> GENERAL transitions on startup, and emits gigabarriers for each pass.
// It takes advantage of VK_KHR_unified_image_layouts.
// Future work: sorting, culling, pass reordering, merging passes
// More adventurous work: aliasing, transient, cross-queue sync
class RenderGraph
{
public:
    void add_pass(const std::string& name, Pass::PassType pass_type, std::function<void(Pass& pass)> setup, std::function<void()> execute);
    void bake();
    void execute(VkCommandBuffer cmd);

    std::unordered_map<std::string, uint32_t> pass_indices{};
    std::unordered_map<std::string, uint32_t> resource_indices{};

    void add_resource(VkImage image = VK_NULL_HANDLE);
    uint32_t get_resources_size() const;
    void print() const;

private:
    // TODO: write only images that discard between pass executions not currently supported as we don't have such cases
    void build_barriers();

    // note: this is kinda redundant and never used for buffers
    struct TrackedResource
    {
        VkImage image = VK_NULL_HANDLE;
    };

    std::vector<Pass> passes{};
    std::vector<TrackedResource> resources{};
    std::vector<uint32_t> early_discards{};
    std::vector<uint32_t> early_depth_discards{};
};
