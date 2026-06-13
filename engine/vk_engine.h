#pragma once

#include "common.h"
#include "vk_math.h"
#include "inputs.h"
#include "resources.h"
#include "vk_loader.h"
#include "pipelines.h"
#include "vk_scene.h"
#include "swapchain.h"
#include "descriptors.h"
#include "queries.h"
#include "config.h"
#include "rendergraph.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

struct DeletionQueue
{
    std::vector<std::function<void()>> deletors{};

    void push_function(std::function<void()>&& function);
    void flush();
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
    double deltatime{};
    double cpu_time{};
    double gpu_time{};
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

struct SDL_Window;
struct CullData;
struct ClusterCullData;
struct CVarSystem;

class VulkanEngine
{
public:
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
    VkPhysicalDeviceAccelerationStructurePropertiesKHR as_properties{};
    VkDevice device{};

    Swapchain swapchain{};

    FrameData frames[MAX_FRAMES_IN_FLIGHT]{};

    FrameData& get_current_frame()
    {
        return frames[frame_number % MAX_FRAMES_IN_FLIGHT];
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

    ShaderCache shader_cache{};

    std::unordered_map<std::string, std::unique_ptr<ShaderPass>> shader_passes{};

    AllocatedImage draw_image{};
    AllocatedImage depth_image{};
    AllocatedImage visibility_buffer{};
    std::array<AllocatedImage, 2> accumulation_buffers{};
    std::vector<AllocatedImage> gbuffers{};
    AllocatedImage depth_pyramid{};
    AllocatedImage hdri{};
    AllocatedImage skybox_cubemap{};
    AllocatedImage irradiance_cubemap{}; // for SH reference
    AllocatedImage prefiltered_envmap{};
    AllocatedImage brdf_lut{};
    AllocatedImage shadow_map{};

    AllocatedImage perlin_noise{};
    std::array<AllocatedImage, 2> scattering_extinction_tex{};
    AllocatedImage light_scattering_tex{};
    AllocatedImage integrated_light_scattering_tex{};
    AllocatedImage blue_noise_tex{};

    AllocatedBuffer light_buffer{};
    AllocatedBuffer light_cluster_buffer{};
    AllocatedBuffer light_index_buffer{};
    AllocatedBuffer light_grid_buffer{};
    AllocatedBuffer light_count_buffer{};

    AllocatedBuffer vertex_buffer{};
    AllocatedBuffer index_buffer{};
    AllocatedBuffer indices_buffer{}; // an indirection buffer - for indexing into the right MeshData
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

    AllocatedBuffer blas_buffer{};
    AllocatedBuffer tlas_buffer{};
    AllocatedBuffer tlas_instance_buffer{};

    AllocatedBuffer resource_heap_buffer{};
    AllocatedBuffer sampler_heap_buffer{};

    ResourceHeapManager resource_heap_manager{};
    SamplerHeapManager sampler_heap_manager{};
    std::vector<VkDescriptorSetAndBindingMappingEXT> desc_mappings{};
    uint32_t depth_pyramid_level_count{};

    VkAccelerationStructureKHR tlas_as{};

    CVarSystem* cvar_system{};
    BDATable bda_table{};
    RenderScene render_scene{};

    TimestampManager timestamp_manager{};
    PipelineQueryManager query_manager{};

    // tracy::VkCtx* tracy_ctx{};

    static VulkanEngine& get();

    void init(int file_count, char** file_paths);
    void cleanup();
    void draw();
    void run();

    // TODO: implement RWTexture ids as Texture+1 so we track a single id - careful with handling ping pong textures that also have RW
    // TODO: this requires moving to untyped pointers, using DescriptorHandle<T>? overall cleaner to manage
    struct Bindless
    {
        uint32_t depth_pyramid_uav{};
        uint32_t skybox_uav{};
        uint32_t draw_uav{};
        uint32_t accum_uav{};
        uint32_t irradiance_uav{};
        uint32_t prefiltered_uav{};
        uint32_t brdf_uav{};
        uint32_t perlin_uav{};
        uint32_t scattering_extinction_uav{};
        uint32_t light_scattering_uav{};
        uint32_t integrated_light_scattering_uav{};

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
        uint32_t perlin_srv{};
        uint32_t scattering_extinction_srv{};
        uint32_t light_scattering_srv{};
        uint32_t integrated_light_scattering_srv{};
        uint32_t blue_noise_srv{};

        uint32_t oit{};
    };

    Bindless bindless{};
private:
    void init_vulkan();
    void init_commands();
    void init_sync_structures();
    void init_descriptors();
    void init_shaders();
    void init_pipelines();
    void init_resources();
    void init_renderables(int file_count, char** file_paths);
    void execute_runtime_setup();
    void init_imgui();
    void build_cluster_grid();

    void update_scene();
    void register_object(const Node* node, const glm::mat4& top_matrix);
    void resolve_taa(VkCommandBuffer cmd);
    void update_cascade();
    void draw_imgui(VkCommandBuffer cmd, VkImageView swapchain_view);
    void upload_scene_data_to_buffers();
    void ready_cull_mesh(RenderScene::MeshPass& pass, CullData& cull_data, glm::mat4& proj);
    void ready_cull_meshlet(RenderScene::MeshPass& pass, ClusterCullData& cull_data, glm::mat4& proj);
    void execute_compact_dispatch(VkCommandBuffer cmd);
    void execute_cull_mesh(VkCommandBuffer cmd, const RenderScene::MeshPass& pass, CullData& cull_data, bool late, uint32_t post_pass);
    void execute_cull_meshlet(VkCommandBuffer cmd, ClusterCullData& cull_data, VkBuffer dispatch_buffer, uint32_t offset, bool late, uint32_t post_pass);
    void execute_shadow_cull(VkCommandBuffer cmd);
    void render(VkCommandBuffer cmd, bool late, uint32_t post_pass);
    void render_transparent(VkCommandBuffer cmd);
    void render_shadows(VkCommandBuffer cmd, uint32_t cascade_idx);
    void execute_hiz_spd(VkCommandBuffer cmd);
    void execute_hiz(VkCommandBuffer cmd);
    void execute_light_culling(VkCommandBuffer cmd);
    void execute_shading(VkCommandBuffer cmd);
    void create_acceleration_structures();
    void register_queries_with_imgui();
    void register_bda_table();
};
