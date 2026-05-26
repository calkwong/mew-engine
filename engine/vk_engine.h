#pragma once

#include "common.h"
#include "vk_math.h"
#include "cache.h"
#include "inputs.h"
#include "resources.h"
#include "vk_loader.h"
#include "pipelines.h"
#include "vk_scene.h"
#include "swapchain.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

constexpr unsigned int FRAME_OVERLAP = 2;

// TODO: move to cpp
struct DeletionQueue
{
    std::vector<std::function<void()>> deletors{};

    void push_function(std::function<void()>&& function)
    {
        deletors.emplace_back(function);
    }

    void flush()
    {
        for (auto it = deletors.rbegin(); it != deletors.rend(); ++it)
        {
            (*it)();
        }

        deletors.clear();
    }
};

struct FrameData
{
    VkCommandPool command_pool{};
    VkCommandBuffer main_command_buffer{};

    VkQueryPool query_pool_timestamps{};
    VkQueryPool query_pool_pipelines{};
    VkQueryPool query_pool_mesh_primitives{};

    VkSemaphore image_acquired_semaphore{};
    VkFence render_fence{};

    AllocatedBuffer scene_buffer{};

    DeletionQueue deletion_queue{};
};

struct EngineStats
{
    unsigned int triangle_count{};
    double deltatime{};
    double cpu_time{};
    double gpu_time{};
    double early_cull{};
    double late_cull{};
    double mask_cull{};
    double hiz{};
    double early_indirect{};
    double late_indirect{};
    double mask_indirect{};
    double deferred_shading{};
    double light_culling{};
    double transparent_cull{};
    double transparent_render{};
    double shadow_cull{};
    double shadow_render{};
    double taa_resolve{};
    unsigned int cascade0{};
    unsigned int cascade1{};
    unsigned int cascade2{};
    unsigned int cascade3{};
};

struct SDL_Window;
struct CullData;
struct ClusterCullData;
struct CVarSystem;

class VulkanEngine
{
public:
    bool is_initialized{ false };
    uint32_t frame_number{ 0 };
    bool freeze_camera{ false };
    bool first_frame{ true };
    glm::mat4 last_view{};
    glm::mat4 last_proj{};

    VkExtent2D window_extent{ 1700, 900 };

    VkInstance instance{}; // vulkan library handle
    VkDebugUtilsMessengerEXT debug_messenger{}; // vulkan debug output handle
    VkPhysicalDevice physical_device{};
    VkSurfaceKHR surface{}; // vulkan window surface
    SDL_Window* window{};

    VkQueue graphics_queue{};
    uint32_t graphics_queue_family{};
    VkPhysicalDeviceProperties2 device_properties{};
    VkPhysicalDeviceDescriptorHeapPropertiesEXT desc_heap_properties{};
    VkDevice device{};

    Swapchain swapchain{};

    FrameData frames[FRAME_OVERLAP]{};

    FrameData& get_current_frame()
    {
        return frames[frame_number % FRAME_OVERLAP];
    }

    std::vector<VkSemaphore> render_done_semaphores{};
    DeletionQueue main_deletion_queue{};

    VmaAllocator allocator{};

    VkFence imm_fence{};
    VkCommandBuffer imm_command_buffer{};
    VkCommandPool imm_command_pool{};

    std::unique_ptr<LoadedGLTF> loaded_scene{};

    Camera main_camera{};
    SceneData scene_data{};
    std::array<CascadeData, 4> cascade_data{};
    std::array<glm::vec2, 8> jitter_offset{};
    EngineStats stats{};

    TextureCache texture_cache{};
    ImageCache image_cache{};
    ShaderCache shader_cache{};

    std::unordered_map<std::string, std::unique_ptr<ShaderPass>> shader_passes{};

    AllocatedImage draw_image{};
    AllocatedImage depth_image{};
    AllocatedImage visibility_buffer{};
    std::array<AllocatedImage, 2> accumulation_buffers{};
    std::vector<AllocatedImage> gbuffers{};
    AllocatedImage depth_pyramid{};
    AllocatedImage hdri{};
    AllocatedImage hdri_cubemap{};
    AllocatedImage irradiance_cubemap{}; // for SH reference
    AllocatedImage prefiltered_envmap{};
    AllocatedImage brdf_lut{};
    AllocatedImage shadow_map{};

    AllocatedBuffer light_buffer{};
    AllocatedBuffer light_cluster_buffer{};
    AllocatedBuffer light_index_buffer{};
    AllocatedBuffer light_grid_buffer{};
    AllocatedBuffer light_count_buffer{};

    AllocatedBuffer blas_buffer{};
    AllocatedBuffer tlas_buffer{};
    AllocatedBuffer tlas_instance_buffer{};

    AllocatedBuffer resource_heap{};
    AllocatedBuffer sampler_heap{};

    std::vector<DescriptorImageInfo> sampled_textures{};
    std::vector<DescriptorImageInfo> rw_images{};

    uint32_t sampled_textures_offset{};
    uint32_t rw_images_offset{};
    uint32_t depth_pyramid_level_count{};

    VkAccelerationStructureKHR tlas_as{};

    CVarSystem* cvar_system{};
    RenderScene render_scene{};

    // tracy::VkCtx* tracy_ctx{};

    static VulkanEngine& get();

    void init(int argc, char** argv);
    void cleanup();
    void draw();
    void run();

    void update_scene();
    void register_object(const Node* node, const glm::mat4& top_matrix);
    void resolve_taa(VkCommandBuffer cmd);
    void update_cascade();
    void draw_imgui(VkCommandBuffer cmd, VkImageView swapchain_view);
    void upload_buffers();
    void ready_mesh_cull(RenderScene::MeshPass& pass, CullData& cull_data, glm::mat4& proj);
    void ready_meshlet_cull(RenderScene::MeshPass& pass, ClusterCullData& cull_data, glm::mat4& proj);
    void execute_compact_dispatch(VkCommandBuffer cmd);
    void execute_compute_cull(VkCommandBuffer cmd, RenderScene::MeshPass& pass, CullData& cull_data, bool late, uint32_t post_pass);
    void execute_compute_cull(VkCommandBuffer cmd, ClusterCullData& cull_data, VkBuffer dispatch_buffer, uint32_t offset, bool late, uint32_t post_pass);
    void execute_shadow_cull(VkCommandBuffer cmd);
    void render(VkCommandBuffer cmd, bool late, uint32_t post_pass, uint32_t query);
    void render_transparent(VkCommandBuffer cmd, uint32_t query);
    void render_shadows(VkCommandBuffer cmd, uint32_t cascade_idx, uint32_t query);
    void execute_hiz_spd(VkCommandBuffer cmd);
    void execute_hiz(VkCommandBuffer cmd);
    void execute_light_culling(VkCommandBuffer cmd);
    void execute_shading(VkCommandBuffer cmd);
    void create_acceleration_structures();
    void update_descriptor_heap();
    void refresh_sampled_textures();
    void refresh_rw_images();

private:
    void init_vulkan();
    void init_commands();
    void init_sync_structures();
    void init_descriptors();
    void init_shaders();
    void init_pipelines();
    void init_resources();
    void init_renderables(int argc, char** argv);
    void execute_baked_gi();
    void init_imgui();
    void build_cluster_grid();
};
