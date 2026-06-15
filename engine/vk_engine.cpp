#include "vk_engine.h"
#include "common.h"
#include "config.h"
#include "vk_math.h"
#include "cvars.h"
#include "inputs.h"
#include "resources.h"
#include "vk_loader.h"
#include "pipelines.h"
#include "vk_scene.h"
#include "rendergraph.h"
#include "swapchain.h"
#include "descriptors.h"
#include "queries.h"

#include <stb_image.h>
#include <vk_mem_alloc.h>
#include <VkBootstrap.h>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/matrix.hpp>
#include <glm/trigonometric.hpp>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_video.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_timer.h>
#include <SDL3/SDL_vulkan.h>
#include <SDL3/SDL_hints.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
// #include <tracy/Tracy.hpp>
// #include <tracy/TracyVulkan.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

VulkanEngine* loaded_engine{};

VulkanEngine& VulkanEngine::get()
{
    return *loaded_engine;
}

// #ifdef NDEBUG
// constexpr bool USE_VALIDATION_LAYERS = false;
// #else
constexpr bool USE_VALIDATION_LAYERS = true;
// #endif

// #define STRESS_TEST // uncomment if loading a proper scene

AutoCVar_Int CVAR_IMGUI{ "imgui", "Imgui", CVarFlags::EditCheckbox | CVarFlags::EditHide, 1 };
AutoCVar_Int CVAR_DISABLE_CAMERA{ "disable_camera", "Disable camera", CVarFlags::EditCheckbox | CVarFlags::EditHide, 0 };
AutoCVar_Int CVAR_HOT_RELOAD{ "hot_reload", "Hot reload shaders", CVarFlags::EditCheckbox | CVarFlags::EditHide, 0 };

AutoCVar_Float CVAR_VOLUMETRIC_NOISE_POS{ "volumetric.noise_pos_mult", "Volumetric noise pos mult", CVarFlags::EditDragFloat, 0.0, 0.0, 1.0, 0.05 };
AutoCVar_Float CVAR_VOLUMETRIC_NOISE_SPEED{ "volumetric.noise_speed_mult", "Volumetric noise speed mult", CVarFlags::EditDragFloat, 0.0, 0.0, 1.0, 0.05 };
AutoCVar_Float CVAR_VOLUMETRIC_FOG_DENSITY{ "volumetric.fog_density", "Volumetric fog density", CVarFlags::EditDragFloat, 0.0, 0.0, 1.0, 0.05 };
AutoCVar_Float CVAR_VOLUMETRIC_HEIGHT_FOG_DENSITY{ "volumetric.height_fog_density", "Volumetric height fog density", CVarFlags::EditDragFloat, 0.8, 0.0, 10.0, 0.5 };
AutoCVar_Float CVAR_VOLUMETRIC_SCATTERING_FACTOR{ "volumetric.scattering_factor", "Volumetric scattering factor", CVarFlags::EditDragFloat, 0.4, 0.0, 1.0, 0.05 };
AutoCVar_Float CVAR_VOLUMETRIC_HEIGHT_FOG_FALLOFF{ "volumetric.height_fog_falloff", "Volumetric height fog falloff", CVarFlags::EditDragFloat, 0.1, 0.0, 10.0, 0.5 };
AutoCVar_Float CVAR_VOLUMETRIC_PHASE_ANISOTROPY{ "volumetric.phase_anisotropy", "Volumetric phase anisotropy", CVarFlags::EditDragFloat, 0.2, 0.0, 1.0, 0.05 };
AutoCVar_Int CVAR_VOLUMETRIC_SPATIAL_FILTERING{ "volumetric.spatial_filtering", "Volumetric spatial filtering", CVarFlags::EditCheckbox, 1 };
AutoCVar_Int CVAR_VOLUMETRIC_TEMPORAL_FILTERING{ "volumetric.temporal_filtering", "Volumetric temporal filtering", CVarFlags::EditCheckbox, 0 };
AutoCVar_Float CVAR_VOLUMETRIC_FAR_PLANE{ "volumetric.far_plane", "Volumetric far plane", CVarFlags::EditDragFloat, 60.0, 50.0, 150.0, 10.0 };
AutoCVar_Int CVAR_DEBUG_3D{ "debug.3d", "Debug 3d texture", CVarFlags::EditCheckbox, 0 };

AutoCVar_Int CVAR_Z_SLICE{ "z_slice", "Noise z", CVarFlags::EditSliderInt, 127, 0, 127, 1 };
AutoCVar_Int CVAR_VBUFFER{ "vbuffer", "Vbuffer path", CVarFlags::EditCheckbox, 1 };
AutoCVar_Int CVAR_MESH_SHADERS{ "mesh_shaders", "Mesh shaders path", CVarFlags::EditCheckbox, 1 };
AutoCVar_Int CVAR_ALPHACLIP{ "alphaclip", "Alphaclip", CVarFlags::EditCheckbox, 1 };
AutoCVar_Int CVAR_TRANSPARENT{ "transparent", "Transparent", CVarFlags::EditCheckbox, 0 };
AutoCVar_Int CVAR_POINT_LIGHTS{ "point_lights", "Point lights", CVarFlags::EditCheckbox, 0 };
AutoCVar_Int CVAR_OCCLUSION_CULLING{ "occlusion_culling", "Occlusion culling", CVarFlags::EditCheckbox, 1 };
AutoCVar_Int CVAR_LOD{ "lod", "LODs", CVarFlags::EditCheckbox, 1 };
AutoCVar_Int CVAR_SHADOWS{ "shadows", "Shadows", CVarFlags::EditCheckbox, 1 };
AutoCVar_Int CVAR_SHADOWS_RT{ "shadows_rt", "Ray traced shadows", CVarFlags::EditCheckbox, 0 };
AutoCVar_Int CVAR_VOLUMETRIC_FOG{ "volumetric_fog", "Volumetric fog", CVarFlags::EditCheckbox, 0 };
AutoCVar_Int CVAR_TAA{ "taa", "TAA", CVarFlags::EditCheckbox, 0 };
AutoCVar_Int CVAR_RT{ "rt", "RT", CVarFlags::EditCheckbox, 0 };
AutoCVar_Int CVAR_VOLUME{ "volume", "KHR_volume", CVarFlags::EditCheckbox, 1 };

AutoCVar_Float CVAR_SHADOWS_CASCADE_SPLIT{ "shadows.cascade_split", "Cascades log factor", CVarFlags::EditDragFloat, 0.95f, 0.f, 1.f, 0.005f };
AutoCVar_Int CVAR_SHADOWS_DISTANCE{ "shadows.distance", "Shadow draw distance", CVarFlags::EditSliderInt, 48, 20, 200, 5 };

AutoCVar_Int CVAR_DEBUG_TEXTURES{ "debug.textures", "Debug textures", CVarFlags::EditSliderInt, 0, 0, DEBUG_COUNT, 1 };

AutoCVar_Int CVAR_DRAW_DISTANCE{ "draw_distance", "Draw distance", CVarFlags::EditSliderInt, 1000, 100, 1000, 100 };
AutoCVar_Int CVAR_AUTOEXPOSURE{ "autoexposure", "Autoexposure", CVarFlags::EditCheckbox, 0 };
AutoCVar_Int CVAR_TONEMAPPING{ "tonemapping", "Tonemapping", CVarFlags::EditCheckbox, 1 };
AutoCVar_Int CVAR_TONEMAPPING_FUNC{ "tonemapping_func", "Tonemapping function", CVarFlags::EditSliderInt, 0, 0, 3, 1 };
AutoCVar_Int CVAR_FREEZE_CAMERA{ "freeze_camera", "Freeze camera", CVarFlags::EditCheckbox, 0 };
AutoCVar_Int CVAR_HIZ_SPD{ "hiz_spd", "HiZ SPD", CVarFlags::EditCheckbox, 1 };

AutoCVar_Int CVAR_TAA_VARIANCE_CLIP{ "taa.variance_clip", "Variance clipping", CVarFlags::EditCheckbox, 1 };
AutoCVar_Int CVAR_TAA_CATMULL_ROM{ "taa.catmull_rom", "Catmull filter", CVarFlags::EditCheckbox, 1 };
AutoCVar_Int CVAR_TAA_MITCHELL{ "taa.mitchell", "Mitchell filter", CVarFlags::EditCheckbox, 0 };
AutoCVar_Int CVAR_TAA_YCOCG{ "taa.ycocg", "YCoCg", CVarFlags::EditCheckbox, 1 };
AutoCVar_Int CVAR_TAA_DYNAMIC{ "taa.dynamic", "Dynamic luma weights", CVarFlags::EditCheckbox, 0 };

namespace
{
uint32_t nearest_pow2(uint32_t extent)
{
    return 1 << static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(extent))));
}

uint32_t next_pow2(uint32_t extent)
{
    return nearest_pow2(extent) << 1;
}

float Halton(uint32_t i, uint32_t b)
{
    float f = 1.0f;
    float r = 0.0f;

    while (i > 0)
    {
        f /= static_cast<float>(b);
        r = r + f * static_cast<float>(i % b);
        auto ratio = static_cast<float>(i) / static_cast<float>(b);
        i = static_cast<uint32_t>(std::floor(ratio));
    }

    return r;
}

uint32_t get_groupcount(uint32_t size, uint32_t threads)
{
    return (size + threads - 1) / threads;
}

float size_in_bytes(uint64_t size)
{
    return static_cast<float>(size) * 1e-6f;
}

// taken directly from https://github.com/zeux/niagara/blob/master/src/scene.cpp
void decompose_transform(const glm::mat4& m, glm::vec3& t, glm::vec3& s, glm::vec4& rotation)
{
    t.x = m[3][0];
    t.y = m[3][1];
    t.z = m[3][2];

    float det = glm::determinant(glm::mat3(m));
    float sign = (det < 0.0f) ? -1.0f : 1.0f;

    s.x = std::sqrt(m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[0][2] * m[0][2]) * sign;
    s.y = std::sqrt(m[1][0] * m[1][0] + m[1][1] * m[1][1] + m[1][2] * m[1][2]) * sign;
    s.z = std::sqrt(m[2][0] * m[2][0] + m[2][1] * m[2][1] + m[2][2] * m[2][2]) * sign;

    float rsx = (s[0] == 0.f) ? 0.f : 1.f / s[0];
    float rsy = (s[1] == 0.f) ? 0.f : 1.f / s[1];
    float rsz = (s[2] == 0.f) ? 0.f : 1.f / s[2];

    // mat = rotation * scale, we want a pure rotation matrix hence normalize axes
    float r00 = m[0][0] * rsx, r10 = m[1][0] * rsy, r20 = m[2][0] * rsz;
    float r01 = m[0][1] * rsx, r11 = m[1][1] * rsy, r21 = m[2][1] * rsz;
    float r02 = m[0][2] * rsx, r12 = m[1][2] * rsy, r22 = m[2][2] * rsz;

    // "branchless" version of Mike Day's matrix to quaternion conversion, no attempt was made to understand quats :)
    int qc = r22 < 0 ? (r00 > r11 ? 0 : 1) : (r00 < -r11 ? 2 : 3);
    float qs1 = qc & 2 ? -1.f : 1.f;
    float qs2 = qc & 1 ? -1.f : 1.f;
    float qs3 = (qc - 1) & 2 ? -1.f : 1.f;

    float qt = 1.f - qs3 * r00 - qs2 * r11 - qs1 * r22;
    float qs = 0.5f / sqrtf(qt);

    rotation[qc ^ 0] = qs * qt;
    rotation[qc ^ 1] = qs * (r01 + qs1 * r10);
    rotation[qc ^ 2] = qs * (r20 + qs2 * r02);
    rotation[qc ^ 3] = qs * (r12 + qs3 * r21);
}

VkBool32 custom_debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
    void* p_user_data
)
{
    auto ms = vkb::to_string_message_severity(message_severity);
    auto mt = vkb::to_string_message_type(message_type);
    if (message_type & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)
    {
        if (strcmp(p_callback_data->pMessageIdName, "VUID-RuntimeSpirv-OpVariable-08746") == 0)
            return VK_FALSE;
        fmt::println("[{}: {}] - {}\n{}\n", ms, mt, p_callback_data->pMessageIdName, p_callback_data->pMessage);
    }
    else
        fmt::println("[{}: {}]\n{}\n", ms, mt, p_callback_data->pMessage);

    return VK_FALSE;
}
} // namespace

void DeletionQueue::push_function(std::function<void()>&& function)
{
    deletors.emplace_back(function);
}

void DeletionQueue::flush()
{
    for (auto it = deletors.rbegin(); it != deletors.rend(); ++it)
    {
        (*it)();
    }

    deletors.clear();
}

void VulkanEngine::init(int file_count, char** file_paths)
{
    assert(loaded_engine == nullptr);
    loaded_engine = this;
    cvar_system = CVarSystem::get();

    VK_CHECK(volkInitialize());

    SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "wayland");
    // SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "x11");
    SDL_SetHint(SDL_HINT_APP_ID, "mew-engine");
    SDL_Init(SDL_INIT_VIDEO);

    auto window_flags = static_cast<SDL_WindowFlags>(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    window = SDL_CreateWindow(
        "Vulkan Engine",
        static_cast<int>(window_extent.width),
        static_cast<int>(window_extent.height),
        window_flags
    );

    // SDL_SetRelativeMouseMode(true);
    SDL_SetWindowRelativeMouseMode(window, true);

    init_vulkan();
    init_commands();
    init_sync_structures();
    init_resources();
    init_renderables(file_count, file_paths);
    create_acceleration_structures();
    register_bda_table();
    init_descriptors(); // after scene creation!
    init_shaders();
    init_pipelines();

    init_imgui();
    main_camera.position = glm::vec3(0, 0, 5);
    main_camera.far = static_cast<float>(cvar_system->get_int_cvar("draw_distance"));
    main_camera.near = 0.01f;
    main_camera.fov = 70.0f;
    main_camera.set_perspective_matrix(glm::radians(main_camera.fov), static_cast<float>(swapchain.extent.width) / static_cast<float>(swapchain.extent.height), main_camera.near);

    // TODO: support draw distance change and rebuilding
    // note: camera must already be set prior to building cluster grid
    build_cluster_grid();
    execute_runtime_setup();

    auto create_query_pool_info = [&](VkQueryType query_type, uint32_t query_count, VkQueryPipelineStatisticFlags pipeline_statistics)
    {
        VkQueryPoolCreateInfo query_pool_info{};
        query_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        query_pool_info.queryType = query_type;
        query_pool_info.queryCount = query_count;
        query_pool_info.pipelineStatistics = pipeline_statistics;

        return query_pool_info;
    };

    auto create_query_pool = [&](VkQueryPoolCreateInfo* p_pool_info, VkQueryPool* p_pool)
    {
        VK_CHECK(vkCreateQueryPool(device, p_pool_info, nullptr, p_pool));
    };

    for (auto& frame : frames)
    {
        auto timestamp_info = create_query_pool_info(VK_QUERY_TYPE_TIMESTAMP, QUERY_COUNT, 0);
        auto pipeline_info = create_query_pool_info(VK_QUERY_TYPE_PIPELINE_STATISTICS, QUERY_COUNT, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT);
        auto mesh_primitives_info = create_query_pool_info(VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT, QUERY_COUNT, VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT);

        VK_CHECK(vkCreateQueryPool(device, &timestamp_info, nullptr, &frame.query_pool_timestamps));
        VK_CHECK(vkCreateQueryPool(device, &pipeline_info, nullptr, &frame.query_pool_pipelines));
        VK_CHECK(vkCreateQueryPool(device, &mesh_primitives_info, nullptr, &frame.query_pool_mesh_primitives));

        vkResetQueryPool(device, frame.query_pool_timestamps, 0, QUERY_COUNT);
        vkResetQueryPool(device, frame.query_pool_pipelines, 0, QUERY_COUNT);
        vkResetQueryPool(device, frame.query_pool_mesh_primitives, 0, QUERY_COUNT);
    }

    // initialize jitter offsets
    for (int i = 0; i < jitter_offset.size(); i++)
    {
        float halton_x = 2.0f * Halton(i + 1, 2) - 1.0f;
        float halton_y = 2.0f * Halton(i + 1, 3) - 1.0f;
        fog_jitter_offset[i] = glm::vec2(halton_x, halton_y);

        float x = halton_x / static_cast<float>(swapchain.extent.width);
        float y = halton_y / static_cast<float>(swapchain.extent.height);
        jitter_offset[i] = glm::vec2(x, y);
    }
}

void VulkanEngine::cleanup()
{
    vkDeviceWaitIdle(device);

    // TracyVkDestroy(tracy_ctx);

    loaded_scene.reset();

    for (auto& frame : frames)
    {
        vkDestroyCommandPool(device, frame.command_pool, nullptr);

        vkDestroyFence(device, frame.render_fence, nullptr);
        vkDestroySemaphore(device, frame.image_acquired_semaphore, nullptr);

        destroy_buffer(allocator, frame.scene_buffer);

        frame.deletion_queue.flush();

        vkDestroyQueryPool(device, frame.query_pool_timestamps, nullptr);
        vkDestroyQueryPool(device, frame.query_pool_pipelines, nullptr);
        vkDestroyQueryPool(device, frame.query_pool_mesh_primitives, nullptr);
    }

    for (auto& sem : render_done_semaphores)
    {
        vkDestroySemaphore(device, sem, nullptr);
    }

    for (const auto& [_, shader_program] : shader_cache.data)
    {
        vkDestroyShaderModule(device, shader_program.get()->module, nullptr);
    }

    destroy_buffer(allocator, object_buffer);
    destroy_buffer(allocator, mesh_buffer);
    destroy_buffer(allocator, meshlet_buffer);
    destroy_buffer(allocator, meshlet_indices);
    destroy_buffer(allocator, material_buffer);
    destroy_buffer(allocator, vertex_buffer);
    destroy_buffer(allocator, index_buffer);

    destroy_buffer(allocator, draw_indirect_buffer);
    destroy_buffer(allocator, dispatch_buffer);
    destroy_buffer(allocator, vis_buffer);
    destroy_buffer(allocator, meshlet_vis_buffer);
    destroy_buffer(allocator, meshlet_dispatch_buffer);
    destroy_buffer(allocator, cluster_indices);

    destroy_buffer(allocator, oit_buffer);
    destroy_buffer(allocator, indices_buffer);

    destroy_buffer(allocator, sh_buffer);
    destroy_buffer(allocator, luminance_buffer);
    destroy_buffer(allocator, luminance_avg_buffer);

    destroy_buffer(allocator, prefix_sum_buffer);
    destroy_buffer(allocator, spd_counter_buffer);

    destroy_buffer(allocator, resource_heap_buffer);
    destroy_buffer(allocator, sampler_heap_buffer);

    for (const auto& [_, shader] : shader_passes)
        vkDestroyPipeline(device, shader->pipeline, nullptr);

    {
        destroy_image(device, allocator, draw_image);
        destroy_image(device, allocator, visibility_buffer);
        for (size_t i = 0; i < gbuffers.size(); ++i)
            destroy_image(device, allocator, gbuffers[i]);
        destroy_image(device, allocator, depth_image);
        destroy_image(device, allocator, accumulation_buffers[0]);
        destroy_image(device, allocator, accumulation_buffers[1]);
        destroy_image(device, allocator, depth_pyramid);
    }

    main_deletion_queue.flush();

    destroy_swapchain(swapchain, device);

    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyDevice(device, nullptr);

    vkb::destroy_debug_utils_messenger(instance, debug_messenger);
    vkDestroyInstance(instance, nullptr);

    SDL_DestroyWindow(window);

    volkFinalize();

    loaded_engine = nullptr;
}

void VulkanEngine::execute_runtime_setup()
{
    VK_CHECK(vkResetFences(device, 1, &imm_fence));
    VK_CHECK(vkResetCommandPool(device, imm_command_pool, 0));

    VkCommandBufferBeginInfo cmd_begin_info{};
    cmd_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmd_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK(vkBeginCommandBuffer(imm_command_buffer, &cmd_begin_info));

    VkBindHeapInfoEXT bind_resource_heap_info{};
    bind_resource_heap_info.sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT;
    bind_resource_heap_info.heapRange = { get_buffer_address(device, resource_heap_buffer.buffer), resource_heap_buffer.size };
    bind_resource_heap_info.reservedRangeOffset = resource_heap_buffer.size - desc_heap_properties.minResourceHeapReservedRange;
    bind_resource_heap_info.reservedRangeSize = desc_heap_properties.minResourceHeapReservedRange;
    vkCmdBindResourceHeapEXT(imm_command_buffer, &bind_resource_heap_info);

    VkBindHeapInfoEXT bind_sampler_heap_info{};
    bind_sampler_heap_info.sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT;
    bind_sampler_heap_info.heapRange = { get_buffer_address(device, sampler_heap_buffer.buffer), sampler_heap_buffer.size };
    bind_sampler_heap_info.reservedRangeOffset = sampler_heap_buffer.size - desc_heap_properties.minSamplerHeapReservedRange;
    bind_sampler_heap_info.reservedRangeSize = desc_heap_properties.minSamplerHeapReservedRange;
    vkCmdBindSamplerHeapEXT(imm_command_buffer, &bind_sampler_heap_info);

    RenderGraph graph{};

    graph.add_pass(
        "skybox",
        Pass::PassType::ComputePass,
        [&](Pass& pass)
        {
            pass.add_image_write("skybox", skybox_cubemap.image);
        },
        [&]()
        {
            ShaderPass current_pass = *shader_passes["equirectangular_to_cubemap"];
            struct PushConstant
            {
                glm::vec2 image_size{};
                uint32_t texture_id{};
                uint32_t image_id{};
            } pc;
            pc.image_size = glm::vec2(skybox_cubemap.extent.width, skybox_cubemap.extent.height);
            pc.texture_id = bindless.hdri_srv;
            pc.image_id = bindless.skybox_uav;

            vkCmdBindPipeline(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
            VkPushDataInfoEXT push_data_info{};
            push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
            push_data_info.data = { &pc, sizeof(PushConstant) };
            vkCmdPushDataEXT(imm_command_buffer, &push_data_info);
            auto groupcount_x = get_groupcount(skybox_cubemap.extent.width, WARP_SIZE);
            auto groupcount_y = get_groupcount(skybox_cubemap.extent.height, WARP_SIZE);
            vkCmdDispatch(imm_command_buffer, groupcount_x, groupcount_y, 6);
        }
    );

    graph.add_pass(
        "skybox_mipmap",
        Pass::PassType::ComputePass,
        [&](Pass& pass)
        {
            pass.add_image_read("skybox", skybox_cubemap.image);
            pass.add_image_write("skybox", skybox_cubemap.image);
        },
        [&]()
        {
            vkutil::generate_mipmaps(imm_command_buffer, skybox_cubemap.image, VkExtent2D(skybox_cubemap.extent.width, skybox_cubemap.extent.height), 6);
        }
    );

    graph.add_pass(
        "spherical_harmonics",
        Pass::PassType::ComputePass,
        [&](Pass& pass)
        {
            pass.add_image_read("skybox", skybox_cubemap.image);
            pass.add_storage_buffer_write("sh");
        },
        [&]()
        {
            ShaderPass current_pass = *shader_passes["spherical_harmonics"];
            struct PushConstant
            {
                VkDeviceAddress sh_buffer_address{};
                uint32_t cubemap_id{};
            } pc;
            pc.sh_buffer_address = bda_table.sh_buffer;
            pc.cubemap_id = scene_data.skybox_id;

            vkCmdBindPipeline(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
            VkPushDataInfoEXT push_data_info{};
            push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
            push_data_info.data = { &pc, sizeof(PushConstant) };
            vkCmdPushDataEXT(imm_command_buffer, &push_data_info);

            vkCmdDispatch(imm_command_buffer, 1, 1, 1);
        }
    );

    graph.add_pass(
        "irradiance",
        Pass::PassType::ComputePass,
        [&](Pass& pass)
        {
            pass.add_image_read("skybox", skybox_cubemap.image);
            pass.add_image_write("irradiance", irradiance_cubemap.image);
        },
        [&]()
        {
            ShaderPass current_pass = *shader_passes["irradiance"];
            struct PushConstant
            {
                glm::vec2 image_size{};
                uint32_t texture_id{};
                uint32_t image_id{};
            } pc;
            pc.image_size = glm::vec2(irradiance_cubemap.extent.width, irradiance_cubemap.extent.height);
            pc.texture_id = scene_data.skybox_id;
            pc.image_id = bindless.irradiance_uav;

            vkCmdBindPipeline(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
            VkPushDataInfoEXT push_data_info{};
            push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
            push_data_info.data = { &pc, sizeof(PushConstant) };
            vkCmdPushDataEXT(imm_command_buffer, &push_data_info);
            auto groupcount_x = get_groupcount(irradiance_cubemap.extent.width, WARP_SIZE);
            auto groupcount_y = get_groupcount(irradiance_cubemap.extent.height, WARP_SIZE);
            vkCmdDispatch(imm_command_buffer, groupcount_x, groupcount_y, 6);
        }
    );

    graph.add_pass(
        "prefiltered",
        Pass::PassType::ComputePass,
        [&](Pass& pass)
        {
            pass.add_image_read("skybox", skybox_cubemap.image);
            pass.add_image_write("prefiltered", prefiltered_envmap.image);
        },
        // TODO: fix - we are dispatching wg_size that is more than necessary here
        [&]()
        {
            ShaderPass current_pass = *shader_passes["prefiltered"];
            vkCmdBindPipeline(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
            struct PushConstant
            {
                uint32_t texture_id{};
                uint32_t image_id{};
                float roughness{};
            } pc;
            pc.texture_id = scene_data.skybox_id;

            auto mips = static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(std::max(prefiltered_envmap.extent.width, prefiltered_envmap.extent.height))))) + 1;
            for (uint32_t i = 0; i < mips; i++)
            {
                pc.image_id = bindless.prefiltered_uav + i;
                pc.roughness = static_cast<float>(i) / static_cast<float>(mips);

                VkPushDataInfoEXT push_data_info{};
                push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
                push_data_info.data = { &pc, sizeof(PushConstant) };
                vkCmdPushDataEXT(imm_command_buffer, &push_data_info);
                auto groupcount_x = get_groupcount(prefiltered_envmap.extent.width, WARP_SIZE);
                auto groupcount_y = get_groupcount(prefiltered_envmap.extent.height, WARP_SIZE);
                vkCmdDispatch(imm_command_buffer, groupcount_x, groupcount_y, 6);
            }
        }
    );

    graph.add_pass(
        "brdf_lut",
        Pass::PassType::ComputePass,
        [&](Pass& pass)
        {
            pass.add_image_write("brdf_lut", brdf_lut.image);
        },
        [&]()
        {
            ShaderPass current_pass = *shader_passes["brdf"];
            struct PushConstant
            {
                glm::vec2 image_size{};
                uint32_t image_id{};
            } pc;
            pc.image_size = glm::vec2(brdf_lut.extent.width, brdf_lut.extent.height);
            pc.image_id = bindless.brdf_uav;

            vkCmdBindPipeline(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
            VkPushDataInfoEXT push_data_info{};
            push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
            push_data_info.data = { &pc, sizeof(PushConstant) };
            vkCmdPushDataEXT(imm_command_buffer, &push_data_info);
            auto groupcount_x = get_groupcount(brdf_lut.extent.width, WARP_SIZE);
            auto groupcount_y = get_groupcount(brdf_lut.extent.height, WARP_SIZE);
            vkCmdDispatch(imm_command_buffer, groupcount_x, groupcount_y, 1);
        }
    );

    graph.add_pass(
        "perlin",
        Pass::PassType::ComputePass,
        [&](Pass& pass)
        {
            pass.add_image_write("perlin", perlin_noise.image);
        },
        [&]()
        {
            struct PushConstant
            {
                glm::uvec3 resolution{};
                uint32_t noise_id{};
            } pc;

            pc.resolution = glm::uvec3(perlin_noise.extent.width, perlin_noise.extent.height, perlin_noise.extent.depth);
            pc.noise_id = bindless.perlin_uav;

            VkPushDataInfoEXT push_data_info{};
            push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
            push_data_info.data = { &pc, sizeof(PushConstant) };
            vkCmdPushDataEXT(imm_command_buffer, &push_data_info);

            ShaderPass current_pass = *shader_passes["perlin"];
            vkCmdBindPipeline(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
            auto groupcount_x = get_groupcount(pc.resolution.x, WARP_SIZE);
            auto groupcount_y = get_groupcount(pc.resolution.y, WARP_SIZE);
            auto groupcount_z = get_groupcount(pc.resolution.z, 1);
            vkCmdDispatch(imm_command_buffer, groupcount_x, groupcount_y, groupcount_z);
        }
    );

    graph.bake();
    graph.execute(imm_command_buffer);

    VK_CHECK(vkEndCommandBuffer(imm_command_buffer));

    VkCommandBufferSubmitInfo cmd_info{};
    cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cmd_info.commandBuffer = imm_command_buffer;

    VkSubmitInfo2 submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submit.pCommandBufferInfos = &cmd_info;
    submit.commandBufferInfoCount = 1;

    VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit, imm_fence));
    VK_CHECK(vkWaitForFences(device, 1, &imm_fence, true, 9999999999));
}

void VulkanEngine::draw()
{
    VK_CHECK(vkWaitForFences(device, 1, &get_current_frame().render_fence, true, 1000000000));
    VK_CHECK(vkResetFences(device, 1, &get_current_frame().render_fence));

    get_current_frame().deletion_queue.flush();

    auto* scene_uniform_data = static_cast<SceneData*>(get_current_frame().scene_buffer.info.pMappedData);
    *scene_uniform_data = scene_data;

    // TODO: potential hazard, we should probably allocate FIF bindings for UBO and offset accordingly
    // TODO: use proper bufferdescriptorsize
    void* descriptor = static_cast<uint8_t*>(resource_heap_buffer.info.pMappedData) + 0 * desc_heap_properties.imageDescriptorSize;
    write_buffer_descriptor(
        device,
        descriptor,
        get_buffer_address(device, get_current_frame().scene_buffer.buffer),
        get_current_frame().scene_buffer.size,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        desc_heap_properties.imageDescriptorSize
    );

    CullData forward_mesh_cull_data{};
    ClusterCullData forward_cluster_cull_data{};

    {
        auto proj = freeze_camera ? last_proj : scene_data.proj;

        ready_cull_mesh(render_scene.opaque_pass, forward_mesh_cull_data, proj);
        ready_cull_meshlet(render_scene.opaque_pass, forward_cluster_cull_data, proj);
    }

    uint32_t swapchain_image_idx{};
    VkResult e = vkAcquireNextImageKHR(device, swapchain.swapchain, 1000000000, get_current_frame().image_acquired_semaphore, nullptr, &swapchain_image_idx);
    if (e == VK_ERROR_OUT_OF_DATE_KHR)
    {
        swapchain.dirty = true;
        return;
    }

    {
        timestamp_manager.get_query_pool_results(frame_number, device, get_current_frame().query_pool_timestamps);
        timestamp_manager.get_render_time(frame_number, device_properties.properties.limits.timestampPeriod);
        timestamp_manager.lerp_timestamp(frame_number, "gpu_time", stats.gpu_time);

        bool mesh_shaders = cvar_system->get_int_cvar("mesh_shaders");
        if (mesh_shaders)
            query_manager.get_query_pool_results(frame_number, device, get_current_frame().query_pool_mesh_primitives, PipelineQueryType::Mesh);
        if (!mesh_shaders || (cvar_system->get_int_cvar("shadows") && !cvar_system->get_int_cvar("shadows_rt")))
            query_manager.get_query_pool_results(frame_number, device, get_current_frame().query_pool_pipelines, PipelineQueryType::Vertex);

        register_queries_with_imgui();

        timestamp_manager.reset(frame_number);
        query_manager.reset(frame_number);
    }

    vkResetQueryPool(device, get_current_frame().query_pool_timestamps, 0, QUERY_COUNT);
    vkResetQueryPool(device, get_current_frame().query_pool_pipelines, 0, QUERY_COUNT);
    vkResetQueryPool(device, get_current_frame().query_pool_mesh_primitives, 0, QUERY_COUNT);

    VkCommandBuffer cmd = get_current_frame().main_command_buffer;
    VK_CHECK(vkResetCommandPool(device, get_current_frame().command_pool, 0));

    VkCommandBufferBeginInfo cmd_begin_info{};
    cmd_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmd_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));

    {
        auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, "gpu_time");

        VkBindHeapInfoEXT bind_resource_heap_info{};
        bind_resource_heap_info.sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT;
        bind_resource_heap_info.heapRange = { get_buffer_address(device, resource_heap_buffer.buffer), resource_heap_buffer.size };
        bind_resource_heap_info.reservedRangeOffset = resource_heap_buffer.size - desc_heap_properties.minResourceHeapReservedRange;
        bind_resource_heap_info.reservedRangeSize = desc_heap_properties.minResourceHeapReservedRange;
        vkCmdBindResourceHeapEXT(cmd, &bind_resource_heap_info);

        VkBindHeapInfoEXT bind_sampler_heap_info{};
        bind_sampler_heap_info.sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT;
        bind_sampler_heap_info.heapRange = { get_buffer_address(device, sampler_heap_buffer.buffer), sampler_heap_buffer.size };
        bind_sampler_heap_info.reservedRangeOffset = sampler_heap_buffer.size - desc_heap_properties.minSamplerHeapReservedRange;
        bind_sampler_heap_info.reservedRangeSize = desc_heap_properties.minSamplerHeapReservedRange;
        vkCmdBindSamplerHeapEXT(cmd, &bind_sampler_heap_info);

        auto zero_buffers = [&]()
        {
            if (cvar_system->get_int_cvar("mesh_shaders"))
            {
                vkCmdFillBuffer(cmd, dispatch_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
                vkCmdFillBuffer(cmd, meshlet_dispatch_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
            }
            vkCmdFillBuffer(cmd, draw_indirect_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(cmd, prefix_sum_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
        };

        auto cull_and_render = [&](
                                   RenderGraph& graph,
                                   const std::string& prefix,
                                   RenderScene::MeshPass& mesh_pass,
                                   uint32_t offset,
                                   bool late,
                                   uint32_t post_pass,
                                   bool clear = false
                               )
        {
            bool mesh_rendering = cvar_system->get_int_cvar("mesh_shaders");
            bool visibility_rendering = cvar_system->get_int_cvar("vbuffer") && mesh_rendering;

            graph.add_pass(
                prefix + "zero_buffers",
                Pass::PassType::ComputePass,
                [&](Pass& pass)
                {
                    if (mesh_rendering)
                    {
                        pass.add_storage_buffer_write("dispatch");
                        pass.add_storage_buffer_write("meshlet_dispatch");
                    }
                    pass.add_storage_buffer_write("draw_indirect");
                    pass.add_storage_buffer_write("prefix_sum");
                },
                [&]()
                {
                    zero_buffers();
                }
            );

            graph.add_pass(
                prefix + "cull_meshes",
                Pass::PassType::ComputePass,
                [&](Pass& pass)
                {
                    pass.add_storage_buffer_read("object");
                    pass.add_storage_buffer_read("mesh");
                    pass.add_storage_buffer_read("indices");
                    if (!mesh_rendering)
                    {
                        pass.add_storage_buffer_write("draw_indirect");
                    }
                    pass.add_storage_buffer_write("dispatch");
                    pass.add_storage_buffer_read("vis");
                    pass.add_storage_buffer_write("prefix_sum");
                    if (late)
                    {
                        pass.add_storage_buffer_write("vis");
                        pass.add_image_read("hiz", depth_pyramid.image);
                    }
                },
                [&, mesh_pass, late, post_pass, prefix]()
                {
                    auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, prefix + "cull_meshes");
                    execute_cull_mesh(cmd, mesh_pass, forward_mesh_cull_data, late, post_pass);
                }
            );

            if (cvar_system->get_int_cvar("mesh_shaders"))
            {
                graph.add_pass(
                    prefix + "compact_dispatch",
                    Pass::PassType::ComputePass,
                    [&](Pass& pass)
                    {
                        pass.add_storage_buffer_write("dispatch");
                        pass.add_storage_buffer_read("prefix_sum");
                    },
                    [&]()
                    {
                        execute_compact_dispatch(cmd);
                    }
                );

                graph.add_pass(
                    prefix + "cull_meshlets",
                    Pass::PassType::ComputePass,
                    [&](Pass& pass)
                    {
                        pass.add_storage_buffer_read("object");
                        pass.add_storage_buffer_read("meshlet");
                        pass.add_storage_buffer_write("cluster_indices");
                        pass.add_storage_buffer_write("meshlet_dispatch");
                        pass.add_storage_buffer_read("cluster_vis");
                        pass.add_storage_buffer_read("prefix_sum");
                        if (late)
                        {
                            pass.add_storage_buffer_write("cluster_vis");
                            pass.add_image_read("hiz", depth_pyramid.image);
                        }
                    },
                    [&, offset, late, post_pass, prefix]()
                    {
                        auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, prefix + "cull_meshlets");
                        execute_cull_meshlet(cmd, forward_cluster_cull_data, dispatch_buffer.buffer, offset, late, post_pass);
                    }
                );
            }

            graph.add_pass(
                prefix + "rasterization",
                Pass::PassType::GraphicsPass,
                [&, clear](Pass& pass)
                {
                    pass.add_storage_buffer_read("object");
                    pass.add_storage_buffer_read("meshlet");
                    pass.add_storage_buffer_read("meshlet_indices");
                    pass.add_storage_buffer_read("cluster_indices");
                    pass.add_storage_buffer_read("material");
                    pass.add_storage_buffer_read("prefix_sum");
                    pass.add_depth_stencil_output("depth", depth_image.image);
                    if (visibility_rendering)
                    {
                        pass.add_color_output("vis_buffer", visibility_buffer.image);
                        if (!clear)
                        {
                            pass.add_image_read("vis_buffer", visibility_buffer.image);
                        }
                    }
                    else
                    {
                        for (uint32_t i = 0; i < gbuffers.size(); i++)
                        {
                            const std::string name = "gbuffer" + std::to_string(i);
                            pass.add_color_output(name, gbuffers[i].image);
                            if (!clear)
                            {
                                pass.add_image_read(name, gbuffers[i].image);
                            }
                        }
                    }
                },
                [&, late, post_pass, prefix]()
                {
                    auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, prefix + "rasterization");
                    render(cmd, late, post_pass);
                }
            );
        };

        auto cull_and_render_transparent = [&](
                                               RenderGraph& graph,
                                               const std::string& prefix,
                                               uint32_t offset,
                                               bool late,
                                               uint32_t post_pass
                                           )
        {
            bool mesh_rendering = cvar_system->get_int_cvar("mesh_shaders");

            graph.add_pass(
                prefix + "zero_buffers",
                Pass::PassType::ComputePass,
                [&](Pass& pass)
                {
                    if (mesh_rendering)
                    {
                        pass.add_storage_buffer_write("dispatch");
                        pass.add_storage_buffer_write("meshlet_dispatch");
                    }
                    pass.add_storage_buffer_write("draw_indirect");
                    pass.add_storage_buffer_write("prefix_sum");
                },
                [&]()
                {
                    zero_buffers();
                }
            );

            graph.add_pass(
                prefix + "cull_meshes",
                Pass::PassType::ComputePass,
                [&](Pass& pass)
                {
                    pass.add_storage_buffer_read("object");
                    pass.add_storage_buffer_read("mesh");
                    pass.add_storage_buffer_read("indices");
                    if (!mesh_rendering)
                    {
                        pass.add_storage_buffer_write("draw_indirect");
                    }
                    pass.add_storage_buffer_write("dispatch");
                    pass.add_storage_buffer_read("vis");
                    pass.add_storage_buffer_write("prefix_sum");
                    // for transparent, it is always late
                    if (late)
                    {
                        pass.add_storage_buffer_write("vis");
                        pass.add_image_read("hiz", depth_pyramid.image);
                    }
                },
                [&, late, post_pass, prefix]()
                {
                    auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, prefix + "cull_meshes");
                    execute_cull_mesh(cmd, render_scene.transparent_pass, forward_mesh_cull_data, late, post_pass);
                }
            );

            if (cvar_system->get_int_cvar("mesh_shaders"))
            {
                graph.add_pass(
                    prefix + "compact_dispatch",
                    Pass::PassType::ComputePass,
                    [&](Pass& pass)
                    {
                        pass.add_storage_buffer_write("dispatch");
                        pass.add_storage_buffer_read("prefix_sum");
                    },
                    [&]()
                    {
                        execute_compact_dispatch(cmd);
                    }
                );

                graph.add_pass(
                    prefix + "cull_meshlets",
                    Pass::PassType::ComputePass,
                    [&](Pass& pass)
                    {
                        pass.add_storage_buffer_read("object");
                        pass.add_storage_buffer_read("meshlet");
                        pass.add_storage_buffer_write("cluster_indices");
                        pass.add_storage_buffer_write("meshlet_dispatch");
                        pass.add_storage_buffer_read("cluster_vis");
                        pass.add_storage_buffer_read("prefix_sum");
                        if (late)
                        {
                            pass.add_storage_buffer_write("cluster_vis");
                            pass.add_image_read("hiz", depth_pyramid.image);
                        }
                    },
                    [&, offset, late, post_pass, prefix]()
                    {
                        auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, prefix + "cull_meshlets");
                        execute_cull_meshlet(cmd, forward_cluster_cull_data, dispatch_buffer.buffer, offset, late, post_pass);
                    }
                );
            }

            graph.add_pass(
                "mlab",
                Pass::PassType::GraphicsPass,
                [&](Pass& pass)
                {
                    pass.add_storage_buffer_read("oit");
                    pass.add_storage_buffer_write("oit");
                    pass.add_storage_buffer_read("object");
                    pass.add_storage_buffer_read("meshlet");
                    pass.add_storage_buffer_read("meshlet_indices");
                    pass.add_storage_buffer_read("cluster_indices");
                    pass.add_storage_buffer_read("material");
                    pass.add_storage_buffer_read("prefix_sum");
                    pass.add_storage_buffer_read("sh");
                    pass.add_image_read("depth", depth_image.image);
                },
                [&]()
                {
                    auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, "mlab");
                    render_transparent(cmd);
                }
            );
        };

        RenderGraph graph{};
        {
            cull_and_render(graph, "opaque_early_", render_scene.opaque_pass, 0, false, 0, true);

            if (!freeze_camera)
            {
                graph.add_pass(
                    "hiz",
                    Pass::PassType::ComputePass,
                    [&](Pass& pass)
                    {
                        if (cvar_system->get_int_cvar("hiz_spd"))
                        {
                            pass.add_storage_buffer_write("spd_counter");
                        }
                        pass.add_image_read("depth", depth_image.image);
                        pass.add_image_write("hiz", depth_pyramid.image);
                    },
                    [&]()
                    {
                        auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, "hiz");
                        if (cvar_system->get_int_cvar("hiz_spd"))
                            execute_hiz_spd(cmd);
                        else
                            execute_hiz(cmd);
                    }
                );
            }

            cull_and_render(graph, "opaque_late_", render_scene.opaque_pass, 0, true, 0);

            // alphaclip postpass only, this is using early pass hiz for culling
            if (cvar_system->get_int_cvar("alphaclip"))
                cull_and_render(graph, "alphaclip_late_", render_scene.mask_pass, 0, true, 1);

            if (cvar_system->get_int_cvar("point_lights"))
            {
                // TODO: combine this somewhere
                // TODO: handle as Transfer instead of setting to Compute?
                graph.add_pass(
                    "zero_light_buffers",
                    Pass::PassType::ComputePass,
                    [&](Pass& pass)
                    {
                        pass.add_storage_buffer_write("light_count");
                    },
                    [&]()
                    {
                        vkCmdFillBuffer(cmd, light_count_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
                    }
                );

                graph.add_pass(
                    "light_culling",
                    Pass::PassType::ComputePass,
                    [&](Pass& pass)
                    {
                        pass.add_storage_buffer_read("light_cluster");
                        pass.add_storage_buffer_read("light");
                        pass.add_storage_buffer_write("light_index");
                        pass.add_storage_buffer_write("light_grid");
                        pass.add_storage_buffer_write("light_count");
                    },
                    [&]()
                    {
                        auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, "light_culling");
                        execute_light_culling(cmd);
                    }
                );
            }

            if (cvar_system->get_int_cvar("shadows") && !cvar_system->get_int_cvar("shadows_rt"))
            {
                graph.add_pass(
                    "zero_shadow_buffers",
                    Pass::PassType::ComputePass,
                    [&](Pass& pass)
                    {
                        pass.add_storage_buffer_write("draw_indirect");
                        pass.add_storage_buffer_write("dispatch");
                    },
                    [&]()
                    {
                        vkCmdFillBuffer(cmd, dispatch_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
                        vkCmdFillBuffer(cmd, draw_indirect_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
                    }
                );

                graph.add_pass(
                    "cull_shadow",
                    Pass::PassType::ComputePass,
                    [&](Pass& pass)
                    {
                        pass.add_storage_buffer_read("object");
                        pass.add_storage_buffer_read("mesh");
                        pass.add_storage_buffer_read("indices");
                        pass.add_storage_buffer_write("draw_indirect");
                    },
                    [&]()
                    {
                        auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, "cull_shadow");
                        execute_shadow_cull(cmd);
                    }
                );

                graph.add_pass(
                    "render_shadows",
                    Pass::PassType::GraphicsPass,
                    [&](Pass& pass)
                    {
                        pass.add_storage_buffer_read("material");
                        pass.add_storage_buffer_read("object");
                        for (size_t i = 0; i < cascade_data.size(); i++)
                            pass.add_depth_stencil_output("shadowmap_" + std::to_string(i), cascade_data[i].shadow_map.image);
                    },
                    [&]()
                    {
                        auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, "render_shadows");
                        for (size_t i = 0; i < cascade_data.size(); i++)
                            render_shadows(cmd, static_cast<uint32_t>(i));
                    }
                );

                graph.add_pass(
                    "scattering_extinction",
                    Pass::PassType::ComputePass,
                    [&](Pass& pass)
                    {
                        pass.add_image_read("perlin_noise", perlin_noise.image);
                        pass.add_image_write("scattering_extinction", scattering_extinction_tex[frame_number % 2].image); // always use current
                        pass.add_image_read("blue_noise", blue_noise_tex.image);
                    },
                    [&]()
                    {
                        auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, "scattering_extinction");

                        struct PushConstants
                        {
                            glm::mat4 inverse_view_proj{};
                            glm::uvec3 froxel_dimensions{};
                            uint32_t current_frame{};
                            glm::vec2 halton{};
                            float near{};
                            float far{};
                            uint32_t perlin_noise_tex{};
                            uint32_t blue_noise_tex{};
                            uint32_t scattering_extinction_tex{};
                            float volumetric_noise_pos_mult{};
                            float volumetric_noise_speed_mult{};
                            float fog_density_modifier{};
                            float height_fog_density_modifier{};
                            float scattering_factor{};
                            float height_fog_falloff{};
                        } pc;

                        pc.inverse_view_proj = scene_data.inverse_viewproj;
                        pc.froxel_dimensions = glm::uvec3(VOLUMETRIC_FROXEL_X, VOLUMETRIC_FROXEL_Y, VOLUMETRIC_FROXEL_Z);
                        pc.current_frame = frame_number;
                        pc.halton = fog_jitter_offset[frame_number % fog_jitter_offset.size()];
                        pc.near = main_camera.near;
                        pc.far = cvar_system->get_float_cvar("volumetric.far_plane");
                        pc.perlin_noise_tex = bindless.perlin_srv;
                        pc.blue_noise_tex = bindless.blue_noise_uav;
                        pc.scattering_extinction_tex = bindless.scattering_extinction_uav + (frame_number % 2);
                        pc.volumetric_noise_pos_mult = cvar_system->get_float_cvar("volumetric.noise_pos_mult");
                        pc.volumetric_noise_speed_mult = cvar_system->get_float_cvar("volumetric.noise_speed_mult");
                        pc.fog_density_modifier = cvar_system->get_float_cvar("volumetric.fog_density");
                        pc.height_fog_density_modifier = cvar_system->get_float_cvar("volumetric.height_fog_density");
                        pc.scattering_factor = cvar_system->get_float_cvar("volumetric.scattering_factor");
                        pc.height_fog_falloff = cvar_system->get_float_cvar("volumetric.height_fog_falloff");

                        VkPushDataInfoEXT push_data_info{};
                        push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
                        push_data_info.data = { &pc, sizeof(PushConstants) };

                        vkCmdPushDataEXT(cmd, &push_data_info);

                        ShaderPass current_pass = *shader_passes["scattering_extinction"];
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
                        auto groupcount_x = get_groupcount(VOLUMETRIC_FROXEL_X, 8);
                        auto groupcount_y = get_groupcount(VOLUMETRIC_FROXEL_Y, 8);
                        auto groupcount_z = get_groupcount(VOLUMETRIC_FROXEL_Z, 1);

                        vkCmdDispatch(cmd, groupcount_x, groupcount_y, groupcount_z);
                    }
                );

                graph.add_pass(
                    "light_scattering",
                    Pass::PassType::ComputePass,
                    [&](Pass& pass)
                    {
                        pass.add_image_read("scattering_extinction", scattering_extinction_tex[frame_number % 2].image); // always use current
                        pass.add_image_write("light_scattering", light_scattering_tex.image);
                        for (size_t i = 0; i < cascade_data.size(); i++)
                            pass.add_image_read("shadowmap_" + std::to_string(i), cascade_data[i].shadow_map.image);
                        pass.add_storage_buffer_read("light");
                        pass.add_storage_buffer_read("light_index");
                        pass.add_storage_buffer_read("light_grid");
                        pass.add_image_read("blue_noise", blue_noise_tex.image);
                    },
                    [&]()
                    {
                        auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, "light_scattering");

                        struct PushConstants
                        {
                            glm::mat4 inverse_view_proj{};
                            VkDeviceAddress light_buffer{};
                            VkDeviceAddress light_index_buffer{};
                            VkDeviceAddress light_grid_buffer{};
                            float near{};
                            float far{};
                            glm::uvec3 froxel_dimensions{};
                            uint32_t scattering_extinction_tex{};
                            uint32_t light_scattering_tex{};
                            uint32_t shadowmap_id{};
                            float light_cluster_scale{};
                            float light_cluster_bias{};
                            glm::vec2 cluster_dim{};
                            glm::vec2 halton{};
                            float phase_anisotropy{};
                            uint32_t blue_noise_tex{};
                            uint32_t current_frame{};
                            uint32_t point_lights{};
                        } pc;

                        pc.inverse_view_proj = scene_data.inverse_viewproj;
                        pc.light_buffer = bda_table.light_buffer;
                        pc.light_index_buffer = bda_table.light_index_buffer;
                        pc.light_grid_buffer = bda_table.light_grid_buffer;
                        pc.near = main_camera.near;
                        pc.far = cvar_system->get_float_cvar("volumetric.far_plane");
                        pc.froxel_dimensions = glm::uvec3(VOLUMETRIC_FROXEL_X, VOLUMETRIC_FROXEL_Y, VOLUMETRIC_FROXEL_Z);
                        pc.scattering_extinction_tex = bindless.scattering_extinction_srv + (frame_number % 2);
                        pc.light_scattering_tex = bindless.light_scattering_uav;
                        pc.shadowmap_id = bindless.shadowmap_srv;

                        const float ratio = main_camera.far / main_camera.near;
                        pc.light_cluster_scale = static_cast<float>(CLUSTER_DEPTH_SLICES) / std::log(ratio);
                        pc.light_cluster_bias = static_cast<float>(CLUSTER_DEPTH_SLICES) * std::log(main_camera.near) / std::log(ratio);
                        auto cluster_x = ceil(static_cast<float>(VOLUMETRIC_FROXEL_X) / CLUSTER_X); // # cluster dim
                        auto cluster_y = ceil(static_cast<float>(VOLUMETRIC_FROXEL_Y) / CLUSTER_Y); // # cluster dim
                        pc.cluster_dim = glm::vec2(cluster_x, cluster_y);
                        pc.halton = fog_jitter_offset[frame_number % fog_jitter_offset.size()];
                        pc.phase_anisotropy = cvar_system->get_float_cvar("volumetric.phase_anisotropy");
                        pc.blue_noise_tex = bindless.blue_noise_uav;
                        pc.current_frame = frame_number;
                        pc.point_lights = cvar_system->get_int_cvar("point_lights");

                        VkPushDataInfoEXT push_data_info{};
                        push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
                        push_data_info.data = { &pc, sizeof(PushConstants) };

                        vkCmdPushDataEXT(cmd, &push_data_info);

                        ShaderPass current_pass = *shader_passes["light_scattering"];
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
                        auto groupcount_x = get_groupcount(VOLUMETRIC_FROXEL_X, 8);
                        auto groupcount_y = get_groupcount(VOLUMETRIC_FROXEL_Y, 8);
                        auto groupcount_z = get_groupcount(VOLUMETRIC_FROXEL_Z, 1);

                        vkCmdDispatch(cmd, groupcount_x, groupcount_y, groupcount_z);
                    }
                );

                if (cvar_system->get_int_cvar("volumetric.spatial_filtering"))
                {
                    graph.add_pass(
                        "fog_spatial_filtering",
                        Pass::PassType::ComputePass,
                        [&](Pass& pass)
                        {
                            pass.add_image_read("light_scattering", light_scattering_tex.image);
                            pass.add_image_write("scattering_extinction", scattering_extinction_tex[frame_number % 2].image); // always use current
                        },
                        [&]()
                        {
                            auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, "fog_spatial_filtering");

                            struct PushConstants
                            {
                                glm::uvec3 froxel_dims{};
                                uint32_t light_scattering_tex{};
                                uint32_t scattering_extinction_tex{};
                            } pc;

                            pc.froxel_dims = glm::uvec3(VOLUMETRIC_FROXEL_X, VOLUMETRIC_FROXEL_Y, VOLUMETRIC_FROXEL_Z);
                            pc.light_scattering_tex = bindless.light_scattering_srv;
                            pc.scattering_extinction_tex = bindless.scattering_extinction_uav + (frame_number % 2);

                            VkPushDataInfoEXT push_data_info{};
                            push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
                            push_data_info.data = { &pc, sizeof(PushConstants) };
                            vkCmdPushDataEXT(cmd, &push_data_info);

                            ShaderPass current_pass = *shader_passes["fog_spatial_filtering"];
                            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
                            auto groupcount_x = get_groupcount(VOLUMETRIC_FROXEL_X, 8);
                            auto groupcount_y = get_groupcount(VOLUMETRIC_FROXEL_Y, 8);
                            auto groupcount_z = get_groupcount(VOLUMETRIC_FROXEL_Z, 1);
                            vkCmdDispatch(cmd, groupcount_x, groupcount_y, groupcount_z);
                        }
                    );

                    /*
                if (cvar_system->get_int_cvar("volumetric.temporal_filtering"))
                {
                    graph.add_pass(
                        "fog_temporal_filtering",
                        Pass::PassType::ComputePass,
                        [&](Pass& pass)
                        {
                            // TODO: this is a hack to prevent UB behaviour from UNDEFINED -> GENERAL for previous_scattering_extinction
                            // pass.add_image_read("previous_scattering_extinction", scattering_extinction_tex[(frame_number + 1) % 2].image);
                            pass.add_image_read("scattering_extinction", scattering_extinction_tex[frame_number % 2].image);
                            pass.add_image_write("scattering_extinction", scattering_extinction_tex[frame_number % 2].image);
                            pass.add_image_read("blue_noise", blue_noise_tex.image);
                        },
                        [&]()
                        {
                            auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, "fog_temporal_filtering");

                            struct PushConstants
                            {
                                glm::mat4 inverse_view_proj{};
                                glm::mat4 previous_view_proj{};
                                glm::vec3 froxel_dims{};
                                uint32_t first_frame{};
                                float near{};
                                float far{};
                                uint32_t previous_scattering_extinction_tex{};
                                uint32_t scattering_extinction_tex{};
                                uint32_t rw_scattering_extinction_tex{};
                                float volumetrics_scale{};
                                float volumetrics_bias{};
                                float reprojection_factor{};
                                glm::vec2 halton{};
                                uint32_t blue_noise_tex{};
                                uint32_t current_frame{};
                            } pc;

                            pc.inverse_view_proj = scene_data.inverse_viewproj;
                            pc.previous_view_proj = scene_data.previous_viewproj;
                            pc.froxel_dims = glm::uvec3(VOLUMETRIC_FROXEL_X, VOLUMETRIC_FROXEL_Y, VOLUMETRIC_FROXEL_Z);
                            pc.first_frame = first_frame ? 1 : 0;
                            // TODO: temporary hack - this needs to be set elsewhere properly, does not play well with TAA at the moment
                            first_frame = false;
                            pc.near = main_camera.near;
                            pc.far = main_camera.far;
                            pc.previous_scattering_extinction_tex = bindless.scattering_extinction_srv + ((frame_number + 1) % 2);
                            pc.scattering_extinction_tex = bindless.scattering_extinction_srv + (frame_number % 2);
                            pc.rw_scattering_extinction_tex = bindless.scattering_extinction_uav + (frame_number % 2);
                            const float ratio = main_camera.far / main_camera.near;
                            float volumetrics_slices = static_cast<float>(VOLUMETRIC_FROXEL_Z);
                            pc.volumetrics_scale = volumetrics_slices / std::log(ratio);
                            pc.volumetrics_bias = volumetrics_slices * std::log(main_camera.near) / std::log(ratio);
                            pc.reprojection_factor = 0.9;
                            pc.halton = fog_jitter_offset[frame_number % fog_jitter_offset.size()];
                            pc.blue_noise_tex = bindless.blue_noise_uav;
                            pc.current_frame = frame_number;

                            VkPushDataInfoEXT push_data_info{};
                            push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
                            push_data_info.data = { &pc, sizeof(PushConstants) };
                            vkCmdPushDataEXT(cmd, &push_data_info);

                            ShaderPass current_pass = *shader_passes["fog_temporal_filtering"];
                            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
                            auto groupcount_x = get_groupcount(VOLUMETRIC_FROXEL_X, 8);
                            auto groupcount_y = get_groupcount(VOLUMETRIC_FROXEL_Y, 8);
                            auto groupcount_z = get_groupcount(VOLUMETRIC_FROXEL_Z, 1);
                            vkCmdDispatch(cmd, groupcount_x, groupcount_y, groupcount_z);
                        }
                    );
                }
                */
                }
            }

            graph.add_pass(
                "light_integration",
                Pass::PassType::ComputePass,
                [&](Pass& pass)
                {
                    if (cvar_system->get_int_cvar("volumetric.spatial_filtering"))
                        pass.add_image_read("scattering_extinction", scattering_extinction_tex[frame_number % 2].image);
                    else
                        pass.add_image_read("light_scattering", light_scattering_tex.image);
                    pass.add_image_write("integrated_light_scattering", integrated_light_scattering_tex.image);
                },
                [&]()
                {
                    auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, "light_integration");

                    struct PushConstants
                    {
                        glm::mat4 inverse_view_proj{};
                        glm::uvec3 froxel_dimensions{};
                        float near{};
                        float far{};
                        uint32_t light_scattering_tex{};
                        uint32_t integrated_light_scattering_tex{};
                    } pc;

                    pc.inverse_view_proj = scene_data.inverse_viewproj;
                    pc.froxel_dimensions = glm::uvec3(VOLUMETRIC_FROXEL_X, VOLUMETRIC_FROXEL_Y, VOLUMETRIC_FROXEL_Z);
                    pc.near = main_camera.near;
                    pc.far = cvar_system->get_float_cvar("volumetric.far_plane");
                    pc.light_scattering_tex = cvar_system->get_int_cvar("volumetric.spatial_filtering") ? bindless.scattering_extinction_srv + (frame_number % 2) : bindless.light_scattering_srv;
                    pc.integrated_light_scattering_tex = bindless.integrated_light_scattering_uav;

                    VkPushDataInfoEXT push_data_info{};
                    push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
                    push_data_info.data = { &pc, sizeof(PushConstants) };

                    vkCmdPushDataEXT(cmd, &push_data_info);

                    ShaderPass current_pass = *shader_passes["light_integration"];
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
                    // TODO: remove hardcoded froxel dim
                    auto groupcount_x = get_groupcount(VOLUMETRIC_FROXEL_X, 8);
                    auto groupcount_y = get_groupcount(VOLUMETRIC_FROXEL_Y, 8);

                    vkCmdDispatch(cmd, groupcount_x, groupcount_y, 1);
                }
            );

            graph.add_pass(
                "lighting_pass",
                Pass::PassType::ComputePass,
                [&](Pass& pass)
                {
                    pass.add_storage_buffer_read("light");
                    pass.add_storage_buffer_read("light_index");
                    pass.add_storage_buffer_read("light_grid");
                    pass.add_storage_buffer_read("meshlet_indices");
                    pass.add_storage_buffer_read("meshlet");
                    pass.add_storage_buffer_read("object");
                    pass.add_storage_buffer_read("material");
                    pass.add_storage_buffer_read("mesh");
                    pass.add_storage_buffer_read("sh");
                    pass.add_image_write("draw", draw_image.image);
                    pass.add_image_read("integrated_light_scattering", integrated_light_scattering_tex.image);
                },
                [&]()
                {
                    auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, "lighting_pass");
                    execute_shading(cmd);
                }
            );

            if (cvar_system->get_int_cvar("transparent"))
            {
                // TODO: make draw_image_mipmap conditional on whether KHR_materials_volume is used
                // TODO: consider doing this in compute and profile to see if its faster
                graph.add_pass(
                    "draw_image_mipmap",
                    Pass::PassType::ComputePass,
                    [&](Pass& pass)
                    {
                        pass.add_image_read("draw", draw_image.image);
                        pass.add_image_write("draw", draw_image.image);
                    },
                    [&]()
                    {
                        auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, "draw_image_mipmap");
                        vkutil::generate_mipmaps(cmd, draw_image.image, VkExtent2D{ swapchain.extent.width, swapchain.extent.height });
                    }
                );

                cull_and_render_transparent(graph, "transparent_late_", 0, true, 2);
            }

            if (cvar_system->get_int_cvar("transparent"))
            {
                graph.add_pass(
                    "composite_transparent",
                    Pass::PassType::ComputePass,
                    [&](Pass& pass)
                    {
                        pass.add_storage_buffer_read("oit");
                        pass.add_storage_buffer_write("oit");
                        pass.add_image_write("draw", draw_image.image);
                        pass.add_image_read("draw", draw_image.image);
                    },
                    [&]()
                    {
                        auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, "composite_transparent");

                        ShaderPass current_pass = *shader_passes["composite_transparent"];
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

                        struct PushData
                        {
                            VkDeviceAddress oit_buffer{};
                            glm::uvec2 screen_size{};
                            uint32_t draw_id{};
                        };

                        PushData pd{
                            bda_table.oit_buffer,
                            glm::uvec2(swapchain.extent.width, swapchain.extent.height),
                            bindless.draw_uav
                        };

                        VkPushDataInfoEXT push_data_info{};
                        push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
                        push_data_info.data = { &pd, sizeof(PushData) };

                        vkCmdPushDataEXT(cmd, &push_data_info);

                        auto groupcount_x = get_groupcount(swapchain.extent.width, 8);
                        auto groupcount_y = get_groupcount(swapchain.extent.height, 8);
                        vkCmdDispatch(cmd, groupcount_x, groupcount_y, 1);
                    }
                );
            }

            // TODO: fix autoexposure

            if (cvar_system->get_int_cvar("taa"))
            {
                graph.add_pass(
                    "taa",
                    Pass::PassType::ComputePass,
                    [&](Pass& pass)
                    {
                        pass.add_image_read("draw", draw_image.image);
                        pass.add_image_read("taa_history", accumulation_buffers[(frame_number + 1) % 2].image);
                        pass.add_image_write("taa_resolve", accumulation_buffers[frame_number % 2].image);
                    },
                    [&]()
                    {
                        auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, "taa");
                        resolve_taa(cmd);
                    }
                );
            }

            if (cvar_system->get_int_cvar("tonemapping") && cvar_system->get_int_cvar("debug.textures") == 0)
            {
                graph.add_pass(
                    "tonemapping",
                    Pass::PassType::ComputePass,
                    [&](Pass& pass)
                    {
                        pass.add_storage_buffer_read("luminance_avg");
                        if (cvar_system->get_int_cvar("taa"))
                            pass.add_image_read("taa_resolve", accumulation_buffers[frame_number % 2].image); // verify
                        else
                            pass.add_image_read("draw", draw_image.image);
                        pass.add_image_write("draw", draw_image.image);
                    },
                    [&]()
                    {
                        auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, "tonemapping");
                        ShaderPass current_pass = *shader_passes["tonemap"];
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

                        struct PushConstant
                        {
                            VkDeviceAddress luminance_avg_buffer{};
                            glm::vec2 screen_size{};
                            uint32_t src_id{};
                            uint32_t dst_id{};
                            uint32_t autoexposure{};
                            uint32_t tonemap_func{};
                        } pc;
                        pc.luminance_avg_buffer = bda_table.luminance_avg_buffer;
                        pc.screen_size = glm::vec2(swapchain.extent.width, swapchain.extent.height);
                        pc.src_id = cvar_system->get_int_cvar("taa") ? bindless.accum_uav + (frame_number % 2) : bindless.draw_uav;
                        pc.dst_id = bindless.draw_uav;
                        // pc.autoexposure = cvar_system->get_int_cvar("autoexposure");
                        pc.tonemap_func = cvar_system->get_int_cvar("tonemapping_func");

                        VkPushDataInfoEXT push_data_info{};
                        push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
                        push_data_info.data = { &pc, sizeof(PushConstant) };
                        vkCmdPushDataEXT(cmd, &push_data_info);
                        auto groupcount_x = get_groupcount(swapchain.extent.width, WARP_SIZE);
                        auto groupcount_y = get_groupcount(swapchain.extent.height, WARP_SIZE);
                        vkCmdDispatch(cmd, groupcount_x, groupcount_y, 1);
                    }
                );
            }

            if (cvar_system->get_int_cvar("debug.3d"))
            {
                graph.add_pass(
                    "debug_3d",
                    Pass::PassType::ComputePass,
                    [&](Pass& pass)
                    {
                        pass.add_image_write("draw", draw_image.image);
                    },
                    [&]()
                    {
                        struct PushConstants
                        {
                            glm::uvec2 swapchain_resolution{};
                            uint32_t debug_texture_id{};
                            uint32_t draw_id{};
                            uint32_t slice{};
                        } pc;

                        pc.swapchain_resolution = glm::uvec2(swapchain.extent.width, swapchain.extent.height);
                        pc.debug_texture_id = bindless.integrated_light_scattering_srv;
                        pc.draw_id = bindless.draw_uav;
                        pc.slice = cvar_system->get_int_cvar("z_slice");

                        VkPushDataInfoEXT push_data_info{};
                        push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
                        push_data_info.data = { &pc, sizeof(PushConstants) };
                        vkCmdPushDataEXT(cmd, &push_data_info);

                        ShaderPass current_pass = *shader_passes["debug_3d"];
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

                        auto groupcount_x = get_groupcount(swapchain.extent.width, WARP_SIZE);
                        auto groupcount_y = get_groupcount(swapchain.extent.height, WARP_SIZE);
                        vkCmdDispatch(cmd, groupcount_x, groupcount_y, 1);
                    }
                );
            }

            graph.add_pass(
                "copy_to_swapchain",
                Pass::PassType::ComputePass,
                [&](Pass& pass)
                {
                    pass.add_image_write("swapchain", swapchain.images[swapchain_image_idx]);
                    pass.add_image_read("draw", draw_image.image);
                },
                [&]()
                {
                    auto ts = ScopedTimestamp(&timestamp_manager, frame_number, cmd, get_current_frame().query_pool_timestamps, "copy_to_swapchain");
                    vkutil::copy_image(cmd, draw_image.image, swapchain.images[swapchain_image_idx], swapchain.extent, swapchain.extent);
                }
            );

            if (cvar_system->get_int_cvar("imgui"))
            {
                graph.add_pass(
                    "imgui",
                    Pass::PassType::GraphicsPass,
                    [&](Pass& pass)
                    {
                        pass.add_color_output("swapchain", swapchain.images[swapchain_image_idx]);
                    },
                    [&]()
                    {
                        draw_imgui(cmd, swapchain.image_views[swapchain_image_idx]);
                    }
                );
            }

            // graph.print();
            graph.bake();
            graph.execute(cmd);
        }

        stage_barrier(
            cmd,
            swapchain.images[swapchain_image_idx],
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_2_NONE,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            0
        );
    }
    // TracyVkCollect(tracy_ctx, get_current_frame().main_command_buffer);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmd_info{};
    cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cmd_info.commandBuffer = cmd;

    VkSemaphoreSubmitInfo wait_info{};
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    wait_info.semaphore = get_current_frame().image_acquired_semaphore;
    wait_info.value = 1;
    wait_info.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSemaphoreSubmitInfo signal_info{};
    signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signal_info.semaphore = render_done_semaphores[swapchain_image_idx];
    signal_info.value = 1;
    signal_info.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;

    VkSubmitInfo2 submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submit.pSignalSemaphoreInfos = &signal_info;
    submit.signalSemaphoreInfoCount = 1;
    submit.pWaitSemaphoreInfos = &wait_info;
    submit.waitSemaphoreInfoCount = 1;
    submit.pCommandBufferInfos = &cmd_info;
    submit.commandBufferInfoCount = 1;
    VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit, get_current_frame().render_fence));

    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &render_done_semaphores[swapchain_image_idx];
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swapchain.swapchain;
    present_info.pImageIndices = &swapchain_image_idx;

    VkResult present_e = vkQueuePresentKHR(graphics_queue, &present_info);
    // this is niri specific - implement a check to ensure size has indeed changed, if not, handle the false positive
    // see: https://github.com/zeux/niagara/commit/a9b85a2997772f15da82cb924871a2d51936bf71
    if (present_e == VK_ERROR_OUT_OF_DATE_KHR)
    {
        swapchain.dirty = true;
        return;
    }

    // FrameMark;
    frame_number++;
}

void VulkanEngine::run()
{
    SDL_Event e{};
    bool b_quit = false;

    auto last_frame = SDL_GetTicks();

    while (!b_quit)
    {
        auto start = SDL_GetTicks();
        auto deltatime = start - last_frame;
        stats.deltatime = static_cast<float>(deltatime / 1000.0f);
        last_frame = start;

        while (SDL_PollEvent(&e) != 0)
        {
            if (e.type == SDL_EVENT_QUIT)
                b_quit = true;

            key_callback(window, e);

            main_camera.process_sdl_event(e, SDL_GetWindowRelativeMouseMode(window));

            if (CVarSystem::get()->get_int_cvar("hot_reload") == 1)
            {
                CVarSystem::get()->set_int_cvar("hot_reload", 0);

                int recompile = std::system("ninja -C bin Shaders");

                if (recompile == 0)
                {
                    bool rebuild{};

                    for (auto& [name, program] : shader_cache.data)
                    {
                        std::string shader_path{ "shaders/compiled/" };
                        shader_path += name;
                        shader_path += ".spv";

                        auto time = std::filesystem::last_write_time(shader_path);

                        if (program->time != time)
                        {
                            program->time = time;
                            vkDestroyShaderModule(device, program->module, nullptr);
                            load_shader_module(shader_path.c_str(), device, &program->module);
                            rebuild = true;
                        }
                    }

                    if (rebuild)
                    {
                        VK_CHECK(vkDeviceWaitIdle(device));

                        for (const auto& [_, shader] : shader_passes)
                            vkDestroyPipeline(device, shader->pipeline, nullptr);

                        // instead of rebuilding everything, we could just update relevant pipelines, but full rebuild is almost instantaneous so we roll with this for now
                        shader_passes.clear();
                        init_pipelines();
                    }
                }
            }

            ImGui_ImplSDL3_ProcessEvent(&e);
        }

        bool update = update_swapchain(swapchain, window, physical_device, device, surface);

        // destroy and recreate textures
        if (update)
        {
            // destroy resources
            {
                destroy_image(device, allocator, draw_image);
                destroy_image(device, allocator, visibility_buffer);
                for (size_t i = 0; i < gbuffers.size(); ++i)
                    destroy_image(device, allocator, gbuffers[i]);
                destroy_image(device, allocator, depth_image);
                destroy_image(device, allocator, depth_pyramid);
                destroy_buffer(allocator, oit_buffer);
                destroy_image(device, allocator, accumulation_buffers[0]);
                destroy_image(device, allocator, accumulation_buffers[1]);
            }

            auto new_extent = VkExtent3D{ swapchain.extent.width, swapchain.extent.height, 1 };

            draw_image = create_image(
                device,
                allocator,
                new_extent,
                VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                0,
                true
            );
            resource_heap_manager.update_srv(bindless.draw_srv, draw_image);
            resource_heap_manager.update_uav(bindless.draw_uav, draw_image);
            // TODO: make below conditional on whether KHR_materials_volume is used
            auto draw_mip_levels = static_cast<uint32_t>(std::floor(std::log2(std::max(new_extent.width, new_extent.height)))) + 1;
            for (uint32_t mip = 1; mip < 11; mip++)
            {
                (mip < draw_mip_levels)
                    ? resource_heap_manager.update_uav(bindless.draw_uav + mip, draw_image, mip)
                    : resource_heap_manager.update_uav(bindless.draw_uav + mip, draw_image, draw_mip_levels - 1);
            }

            depth_image = create_image_with_view(device, allocator, new_extent, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
            resource_heap_manager.update_srv(bindless.depth_srv, depth_image);
            visibility_buffer = create_image_with_view(device, allocator, new_extent, VK_FORMAT_R32G32_UINT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
            resource_heap_manager.update_srv(bindless.vbuffer_srv, visibility_buffer);

            auto gbuffer_flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

            gbuffers.clear();
            gbuffers.emplace_back(create_image_with_view(device, allocator, new_extent, VK_FORMAT_R8G8B8A8_UNORM, gbuffer_flags, VK_IMAGE_ASPECT_COLOR_BIT));
            gbuffers.emplace_back(create_image_with_view(device, allocator, new_extent, VK_FORMAT_R16G16B16A16_SFLOAT, gbuffer_flags, VK_IMAGE_ASPECT_COLOR_BIT));
            gbuffers.emplace_back(create_image_with_view(device, allocator, new_extent, VK_FORMAT_R8G8B8A8_UNORM, gbuffer_flags, VK_IMAGE_ASPECT_COLOR_BIT));
            gbuffers.emplace_back(create_image_with_view(device, allocator, new_extent, VK_FORMAT_R8G8B8A8_UNORM, gbuffer_flags, VK_IMAGE_ASPECT_COLOR_BIT));
            resource_heap_manager.update_srv(bindless.gbuffer_srv + 0, gbuffers[0]);
            resource_heap_manager.update_srv(bindless.gbuffer_srv + 1, gbuffers[1]);
            resource_heap_manager.update_srv(bindless.gbuffer_srv + 2, gbuffers[2]);
            resource_heap_manager.update_srv(bindless.gbuffer_srv + 3, gbuffers[3]);

            VkExtent3D depth_pyramid_extent{};
            depth_pyramid_extent.width = nearest_pow2(swapchain.extent.width);
            depth_pyramid_extent.height = nearest_pow2(swapchain.extent.height);
            depth_pyramid_extent.depth = 1;
            depth_pyramid_level_count = static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(std::max(depth_pyramid_extent.width, depth_pyramid_extent.height))))) + 1;
            depth_pyramid = create_image(
                device,
                allocator,
                depth_pyramid_extent,
                VK_FORMAT_R32_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                0,
                true
            );
            resource_heap_manager.update_srv(bindless.depth_pyramid_srv, depth_pyramid);
            for (size_t mip = 0; mip < 11; mip++)
            {
                if (mip < depth_pyramid_level_count)
                    resource_heap_manager.update_uav(bindless.depth_pyramid_uav + mip, depth_pyramid, mip);
                else
                    resource_heap_manager.update_uav(bindless.depth_pyramid_uav + mip, depth_pyramid, depth_pyramid_level_count - 1);
            }

            // note: this is not the right way to handle TAA resize but for simplicity just destroy and recreate immediately
            accumulation_buffers[0] = create_image(device, allocator, new_extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
            accumulation_buffers[1] = create_image(device, allocator, new_extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
            resource_heap_manager.update_srv(bindless.accum_srv, accumulation_buffers[0]);
            resource_heap_manager.update_srv(bindless.accum_srv + 1, accumulation_buffers[1]);
            resource_heap_manager.update_uav(bindless.accum_uav, accumulation_buffers[0]);
            resource_heap_manager.update_uav(bindless.accum_uav + 1, accumulation_buffers[1]);

            {
                auto screen_pixels = new_extent.width * new_extent.height;
                oit_buffer = create_buffer(
                    allocator,
                    screen_pixels * sizeof(OITData),
                    0,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                );
                resource_heap_manager.update_buffer(bindless.oit, oit_buffer, oit_buffer.size);

                immediate_submit(
                    device,
                    graphics_queue,
                    imm_fence,
                    imm_command_pool,
                    imm_command_buffer,
                    [&](VkCommandBuffer cmd)
                    {
                        vkCmdFillBuffer(cmd, oit_buffer.buffer, 0, VK_WHOLE_SIZE, 0x3F800000);
                    }
                );
            }

            resource_heap_manager.write_resource_heap(device, resource_heap_buffer.info.pMappedData, true);
            register_bda_table();
        }

        freeze_camera = cvar_system->get_int_cvar("freeze_camera");

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        // this must be done before fence
        CVarSystem::get()->draw_imgui_editor();

        update_scene();

        draw();

        auto end = SDL_GetTicks();
        auto elapsed = end - start;
        stats.cpu_time = stats.cpu_time * 0.95 + elapsed * 0.05;
    }
}

void VulkanEngine::init_vulkan()
{
    vkb::InstanceBuilder builder{};

    // create vulkan instance, with basic debug features
    auto inst_ret = builder.set_app_name("Example Vulkan Application")
                        .request_validation_layers(USE_VALIDATION_LAYERS)
                        .set_debug_callback(custom_debug_callback)
                        .require_api_version(1, 4, 0)
                        .build();

    vkb::Instance vkb_inst = inst_ret.value();

    // grab the instance
    instance = vkb_inst.instance;
    debug_messenger = vkb_inst.debug_messenger;

    volkLoadInstanceOnly(instance);

    SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface);

    VkPhysicalDeviceVulkan14Features features14{};
    features14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;

    // vulkan 1.3 features
    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.dynamicRendering = true;
    features13.synchronization2 = true;
    features13.maintenance4 = true;
    features13.shaderDemoteToHelperInvocation = true;

    // vulkan 1.2 features
    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.bufferDeviceAddress = true;
    features12.descriptorIndexing = true;
    features12.descriptorBindingPartiallyBound = true;
    features12.descriptorBindingVariableDescriptorCount = true;
    features12.runtimeDescriptorArray = true;
    features12.shaderSampledImageArrayNonUniformIndexing = true;
    features12.drawIndirectCount = true;
    features12.samplerFilterMinmax = true;
    features12.hostQueryReset = true;
    features12.shaderFloat16 = true;
    features12.shaderInt8 = true;
    features12.storageBuffer8BitAccess = true;
    features12.shaderBufferInt64Atomics = true;
    features12.storagePushConstant8 = true; // note: possible slang capability bug, setting to true so val layer doesn't complain
    features12.uniformAndStorageBuffer8BitAccess = true;

    // vulkan 1.1 features
    VkPhysicalDeviceVulkan11Features features11{};
    features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    features11.storageBuffer16BitAccess = true;
    features11.storagePushConstant16 = true; // note: possible slang capability bug, setting to true so val layer doesn't complain
    features11.uniformAndStorageBuffer16BitAccess = true;

    // vulkan 1.0 features
    VkPhysicalDeviceFeatures features10{};
    features10.multiDrawIndirect = true;
    features10.pipelineStatisticsQuery = true;
    // features10.samplerAnisotropy = true;
    features10.depthClamp = true;
    features10.shaderInt16 = true;
    features10.shaderInt64 = true;
    features10.fragmentStoresAndAtomics = true;

    VkPhysicalDeviceMeshShaderFeaturesEXT mesh_shader_features{};
    mesh_shader_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
    mesh_shader_features.meshShader = true;
    mesh_shader_features.meshShaderQueries = true;

    VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT fragment_shader_interlock_features{};
    fragment_shader_interlock_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT;
    fragment_shader_interlock_features.fragmentShaderPixelInterlock = true;

    VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features{};
    ray_query_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    ray_query_features.rayQuery = true;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_features{};
    acceleration_structure_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    acceleration_structure_features.accelerationStructure = true;

    VkPhysicalDeviceDescriptorHeapFeaturesEXT desc_heap_features{};
    desc_heap_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;
    desc_heap_features.descriptorHeap = true;

    vkb::PhysicalDeviceSelector selector{ vkb_inst };
    vkb::PhysicalDevice physicalDevice =
        selector.set_minimum_version(1, 4)
            .set_required_features(features10)
            .set_required_features_14(features14)
            .set_required_features_13(features13)
            .set_required_features_12(features12)
            .set_required_features_11(features11)
            .add_required_extension("VK_KHR_calibrated_timestamps")
            .add_required_extension("VK_EXT_mesh_shader")
            .add_required_extension("VK_EXT_fragment_shader_interlock")
            .add_required_extension("VK_KHR_ray_query")
            .add_required_extension("VK_KHR_deferred_host_operations")
            .add_required_extension("VK_KHR_acceleration_structure")
            .add_required_extension("VK_EXT_descriptor_heap")
            .add_required_extension_features(mesh_shader_features)
            .add_required_extension_features(fragment_shader_interlock_features)
            .add_required_extension_features(ray_query_features)
            .add_required_extension_features(acceleration_structure_features)
            .add_required_extension_features(desc_heap_features)
            .set_surface(surface)
            .select()
            .value();

    // create the final vulkan device
    vkb::DeviceBuilder deviceBuilder{ physicalDevice };

    vkb::Device vkbDevice = deviceBuilder.build().value();

    // get the VkDevice handle used in the rest of a vulkan application
    device = vkbDevice.device;
    physical_device = physicalDevice.physical_device;

    volkLoadDevice(device);

    // use vkbootstrap to get a Graphics queue
    graphics_queue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    graphics_queue_family = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    VmaAllocatorCreateInfo allocator_info{};
    allocator_info.physicalDevice = physical_device;
    allocator_info.device = device;
    allocator_info.instance = instance;
    allocator_info.vulkanApiVersion = VK_API_VERSION_1_4;
    allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT; // allows usage of GPU pointers

    VmaVulkanFunctions vulkan_functions{};
    VK_CHECK(vmaImportVulkanFunctionsFromVolk(&allocator_info, &vulkan_functions));
    allocator_info.pVulkanFunctions = &vulkan_functions;
    vmaCreateAllocator(&allocator_info, &allocator);

    main_deletion_queue.push_function(
        [&]()
        {
            vmaDestroyAllocator(allocator);
        }
    );

    create_swapchain(swapchain, physical_device, device, surface, window_extent.width, window_extent.height);

    device_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    desc_heap_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT;
    as_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
    as_properties.pNext = &desc_heap_properties;
    device_properties.pNext = &as_properties;
    vkGetPhysicalDeviceProperties2(physical_device, &device_properties);
    assert(device_properties.properties.limits.timestampComputeAndGraphics);

    uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extension_count, extensions.data());

    struct ExtensionInfo
    {
        const char* name{};
        bool supported{};
    };

    std::vector<ExtensionInfo> extension_infos = {
        { VK_KHR_RAY_QUERY_EXTENSION_NAME, false },
        { VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false },
        { VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, false },
        { VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME, false },
        { VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME, false },
        { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, false }
    };

    for (uint32_t i = 0; i < extension_count; i++)
    {
        for (auto& ext : extension_infos)
        {
            if (strcmp(ext.name, extensions[i].extensionName) == 0)
                ext.supported = true;
        }
    }

    for (const auto& ext : extension_infos)
    {
        if (ext.supported == false)
            abort();
    }
}

void VulkanEngine::init_commands()
{
    VkCommandPoolCreateInfo command_pool_info{};
    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.queueFamilyIndex = graphics_queue_family;

    for (auto& frame : frames)
    {
        VK_CHECK(vkCreateCommandPool(device, &command_pool_info, nullptr, &frame.command_pool));

        VkCommandBufferAllocateInfo cmd_alloc_info{};
        cmd_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc_info.commandPool = frame.command_pool;
        cmd_alloc_info.commandBufferCount = 1;
        cmd_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        VK_CHECK(vkAllocateCommandBuffers(device, &cmd_alloc_info, &frame.main_command_buffer));
    }

    // tracy_ctx = TracyVkContextCalibrated(physical_device, device, graphics_queue, frames[0].main_command_buffer, vkGetPhysicalDeviceCalibrateableTimeDomainsKHR, vkGetCalibratedTimestampsKHR);

    VK_CHECK(vkCreateCommandPool(device, &command_pool_info, nullptr, &imm_command_pool));

    VkCommandBufferAllocateInfo cmd_alloc_info{};
    cmd_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc_info.commandPool = imm_command_pool;
    cmd_alloc_info.commandBufferCount = 1;
    cmd_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    VK_CHECK(vkAllocateCommandBuffers(device, &cmd_alloc_info, &imm_command_buffer));

    main_deletion_queue.push_function(
        [&]()
        {
            vkDestroyCommandPool(device, imm_command_pool, nullptr);
        }
    );
}

void VulkanEngine::init_sync_structures()
{
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    for (auto& frame : frames)
    {
        VK_CHECK(vkCreateFence(device, &fence_info, nullptr, &frame.render_fence));

        VK_CHECK(vkCreateSemaphore(device, &semaphore_info, nullptr, &frame.image_acquired_semaphore));
    }

    render_done_semaphores.resize(swapchain.images.size());
    for (auto& sem : render_done_semaphores)
    {
        VK_CHECK(vkCreateSemaphore(device, &semaphore_info, nullptr, &sem));
    }

    VK_CHECK(vkCreateFence(device, &fence_info, nullptr, &imm_fence));

    main_deletion_queue.push_function(
        [&]()
        {
            vkDestroyFence(device, imm_fence, nullptr);
        }
    );
}

void VulkanEngine::init_descriptors()
{
    // TODO: use proper size
    resource_heap_buffer = create_buffer(
        allocator,
        // TODO: compute this properly, accounting for image and buffer separately
        3000 * desc_heap_properties.imageDescriptorSize + desc_heap_properties.minResourceHeapReservedRange,
        VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
    );

    sampler_heap_buffer = create_buffer(
        allocator,
        7 * desc_heap_properties.samplerDescriptorSize + desc_heap_properties.minSamplerHeapReservedRange,
        VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
    );

    // TODO: use proper buffer descriptor size
    resource_heap_manager.buffer_descriptor_size = desc_heap_properties.imageDescriptorSize;
    resource_heap_manager.image_descriptor_size = desc_heap_properties.imageDescriptorSize;
    sampler_heap_manager.sampler_descriptor_size = desc_heap_properties.samplerDescriptorSize;

    resource_heap_manager.write_resource_heap(device, resource_heap_buffer.info.pMappedData);
    sampler_heap_manager.write_sampler_heap(device, sampler_heap_buffer.info.pMappedData);

    resource_heap_manager.build_desc_set_bindings(desc_mappings);
    sampler_heap_manager.build_desc_set_bindings(desc_mappings);
}

void VulkanEngine::init_shaders()
{
    shader_cache.add_shader(device, "cluster_grid.slang");
    shader_cache.add_shader(device, "light_culling.slang");
    shader_cache.add_shader(device, "hiz.slang");
    shader_cache.add_shader(device, "cull_mesh.slang");
    shader_cache.add_shader(device, "cull_meshlet.slang");
    shader_cache.add_shader(device, "cull_shadow.slang");
    shader_cache.add_shader(device, "resolve_taa.slang");
    shader_cache.add_shader(device, "equirectangular_to_cubemap.slang");
    shader_cache.add_shader(device, "spherical_harmonics.slang");
    shader_cache.add_shader(device, "irradiance.slang");
    shader_cache.add_shader(device, "prefiltered.slang");
    shader_cache.add_shader(device, "brdf.slang");
    shader_cache.add_shader(device, "luminance_histogram.slang");
    shader_cache.add_shader(device, "luminance_avg.slang");
    shader_cache.add_shader(device, "tonemap.slang");
    shader_cache.add_shader(device, "resolve_vbuffer.slang");
    shader_cache.add_shader(device, "resolve_gbuffer.slang");
    shader_cache.add_shader(device, "compact_dispatch.slang");
    shader_cache.add_shader(device, "hiz_spd.slang");
    shader_cache.add_shader(device, "depth.slang");
    shader_cache.add_shader(device, "rasterize_vbuffer.slang");
    shader_cache.add_shader(device, "rasterize_gbuffer.slang");
    shader_cache.add_shader(device, "mlab.slang");
    shader_cache.add_shader(device, "composite_transparent.slang");
    shader_cache.add_shader(device, "perlin.slang");
    shader_cache.add_shader(device, "debug_3d.slang");
    shader_cache.add_shader(device, "scattering_extinction.slang");
    shader_cache.add_shader(device, "light_scattering.slang");
    shader_cache.add_shader(device, "light_integration.slang");
    shader_cache.add_shader(device, "fog_spatial_filtering.slang");
    shader_cache.add_shader(device, "fog_temporal_filtering.slang");
    // shader_cache.add_shader(device, "rt.slang", sizeof(DeferredPushConstants));
}

void VulkanEngine::init_pipelines()
{
    VkShaderDescriptorSetAndBindingMappingInfoEXT desc_set_and_binding_mapping_info{};
    desc_set_and_binding_mapping_info.sType = VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT;
    desc_set_and_binding_mapping_info.mappingCount = desc_mappings.size();
    desc_set_and_binding_mapping_info.pMappings = desc_mappings.data();

    shader_passes["cluster_grid"] = create_compute_pipeline(device, shader_cache["cluster_grid.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["light_culling"] = create_compute_pipeline(device, shader_cache["light_culling.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["hiz"] = create_compute_pipeline(device, shader_cache["hiz.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["mesh_cull"] = create_compute_pipeline(device, shader_cache["cull_mesh.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["meshlet_cull"] = create_compute_pipeline(device, shader_cache["cull_meshlet.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["equirectangular_to_cubemap"] = create_compute_pipeline(device, shader_cache["equirectangular_to_cubemap.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["spherical_harmonics"] = create_compute_pipeline(device, shader_cache["spherical_harmonics.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["irradiance"] = create_compute_pipeline(device, shader_cache["irradiance.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["prefiltered"] = create_compute_pipeline(device, shader_cache["prefiltered.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["brdf"] = create_compute_pipeline(device, shader_cache["brdf.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["luminance_histogram"] = create_compute_pipeline(device, shader_cache["luminance_histogram.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["luminance_avg"] = create_compute_pipeline(device, shader_cache["luminance_avg.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["tonemap"] = create_compute_pipeline(device, shader_cache["tonemap.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["shadow_cull"] = create_compute_pipeline(device, shader_cache["cull_shadow.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["compact_dispatch"] = create_compute_pipeline(device, shader_cache["compact_dispatch.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["resolve_taa"] = create_compute_pipeline(device, shader_cache["resolve_taa.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["hiz_spd"] = create_compute_pipeline(device, shader_cache["hiz_spd.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["resolve_gbuffer"] = create_compute_pipeline(device, shader_cache["resolve_gbuffer.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["resolve_vbuffer"] = create_compute_pipeline(device, shader_cache["resolve_vbuffer.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["composite_transparent"] = create_compute_pipeline(device, shader_cache["composite_transparent.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["perlin"] = create_compute_pipeline(device, shader_cache["perlin.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["debug_3d"] = create_compute_pipeline(device, shader_cache["debug_3d.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["scattering_extinction"] = create_compute_pipeline(device, shader_cache["scattering_extinction.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["light_scattering"] = create_compute_pipeline(device, shader_cache["light_scattering.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["light_integration"] = create_compute_pipeline(device, shader_cache["light_integration.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["fog_spatial_filtering"] = create_compute_pipeline(device, shader_cache["fog_spatial_filtering.slang"], &desc_set_and_binding_mapping_info);
    shader_passes["fog_temporal_filtering"] = create_compute_pipeline(device, shader_cache["fog_temporal_filtering.slang"], &desc_set_and_binding_mapping_info);

    // shader_passes["ray_tracing"] = create_compute_pipeline(device, shader_cache["rt.slang"], &desc_set_and_binding_mapping_info);

    shader_passes["gbuffer_vert"] = create_graphics_pipeline(
        device,
        shader_cache["rasterize_gbuffer.slang"],
        { VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT },
        &desc_set_and_binding_mapping_info,
        { gbuffers[0].format, gbuffers[1].format, gbuffers[2].format, gbuffers[3].format },
        { 1 },
        [&](VkPipelineRasterizationStateCreateInfo& r, VkPipelineDepthStencilStateCreateInfo& d)
        {
        }
    );
    shader_passes["gbuffer_vert_alphaclip"] = create_graphics_pipeline(
        device,
        shader_cache["rasterize_gbuffer.slang"],
        { VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT },
        &desc_set_and_binding_mapping_info,
        { gbuffers[0].format, gbuffers[1].format, gbuffers[2].format, gbuffers[3].format },
        { 0 },
        [&](VkPipelineRasterizationStateCreateInfo& r, VkPipelineDepthStencilStateCreateInfo& d)
        {
            r.cullMode = VK_CULL_MODE_NONE;
        }
    );
    shader_passes["gbuffer_mesh"] = create_graphics_pipeline(
        device,
        shader_cache["rasterize_gbuffer.slang"],
        { VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_STAGE_FRAGMENT_BIT },
        &desc_set_and_binding_mapping_info,
        { gbuffers[0].format, gbuffers[1].format, gbuffers[2].format, gbuffers[3].format },
        { 1 },
        [&](VkPipelineRasterizationStateCreateInfo& r, VkPipelineDepthStencilStateCreateInfo& d)
        {
        }
    );
    shader_passes["gbuffer_mesh_alphaclip"] = create_graphics_pipeline(
        device,
        shader_cache["rasterize_gbuffer.slang"],
        { VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_STAGE_FRAGMENT_BIT },
        &desc_set_and_binding_mapping_info,
        { gbuffers[0].format, gbuffers[1].format, gbuffers[2].format, gbuffers[3].format },
        { 0 },
        [&](VkPipelineRasterizationStateCreateInfo& r, VkPipelineDepthStencilStateCreateInfo& d)
        {
            r.cullMode = VK_CULL_MODE_NONE;
        }
    );

    shader_passes["vbuffer"] = create_graphics_pipeline(
        device,
        shader_cache["rasterize_vbuffer.slang"],
        { VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_STAGE_FRAGMENT_BIT },
        &desc_set_and_binding_mapping_info,
        { visibility_buffer.format },
        { 1 },
        [&](VkPipelineRasterizationStateCreateInfo& r, VkPipelineDepthStencilStateCreateInfo& d)
        {
        }
    );
    shader_passes["vbuffer_alphaclip"] = create_graphics_pipeline(
        device,
        shader_cache["rasterize_vbuffer.slang"],
        { VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_STAGE_FRAGMENT_BIT },
        &desc_set_and_binding_mapping_info,
        { visibility_buffer.format },
        { 0 },
        [&](VkPipelineRasterizationStateCreateInfo& r, VkPipelineDepthStencilStateCreateInfo& d)
        {
            r.cullMode = VK_CULL_MODE_NONE;
        }
    );

    shader_passes["depth"] = create_graphics_pipeline(
        device,
        shader_cache["depth.slang"],
        { VK_SHADER_STAGE_VERTEX_BIT },
        &desc_set_and_binding_mapping_info,
        {},
        { 1 },
        [&](VkPipelineRasterizationStateCreateInfo& r, VkPipelineDepthStencilStateCreateInfo& d)
        {
            r.cullMode = VK_CULL_MODE_FRONT_BIT;
            r.depthClampEnable = VK_TRUE;
        }
    );
    shader_passes["depth_alphaclip"] = create_graphics_pipeline(
        device,
        shader_cache["depth.slang"],
        { VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT },
        &desc_set_and_binding_mapping_info,
        {},
        { 0 },
        [&](VkPipelineRasterizationStateCreateInfo& r, VkPipelineDepthStencilStateCreateInfo& d)
        {
            r.cullMode = VK_CULL_MODE_NONE;
            r.depthClampEnable = VK_TRUE;
        }
    );

    // TODO: alphacoverage and transmission currently implemented as an ubershader, but the default for transmission
    // should not be cull_mode==none with depth write disabled
    shader_passes["mlab_vert"] = create_graphics_pipeline(
        device,
        shader_cache["mlab.slang"],
        { VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT },
        &desc_set_and_binding_mapping_info,
        {},
        {},
        [&](VkPipelineRasterizationStateCreateInfo& r, VkPipelineDepthStencilStateCreateInfo& d)
        {
            d.depthWriteEnable = VK_FALSE;
            r.cullMode = VK_CULL_MODE_NONE;
        }
    );
    shader_passes["mlab_mesh"] = create_graphics_pipeline(
        device,
        shader_cache["mlab.slang"],
        { VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_STAGE_FRAGMENT_BIT },
        &desc_set_and_binding_mapping_info,
        {},
        {},
        [&](VkPipelineRasterizationStateCreateInfo& r, VkPipelineDepthStencilStateCreateInfo& d)
        {
            d.depthWriteEnable = VK_FALSE;
            r.cullMode = VK_CULL_MODE_NONE;
        }
    );
}

void VulkanEngine::init_resources()
{
    for (auto& frame : frames)
    {
        frame.scene_buffer = create_buffer(
            allocator,
            sizeof(SceneData),
            VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
        );
    }
    resource_heap_manager.add_buffer(frames[0].scene_buffer, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

    auto image_extent = VkExtent3D{ swapchain.extent.width, swapchain.extent.height, 1 };

    // if reverting to VK_FORMAT_R16G16B16A16_SFLOAT, need to preexpose lights
    draw_image = create_image(
        device,
        allocator,
        image_extent,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        0,
        true
    );
    bindless.draw_srv = resource_heap_manager.add_srv(draw_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    bindless.draw_uav = resource_heap_manager.add_uav(draw_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);

    // TODO: make below conditional on whether KHR_materials_volume is used
    auto draw_mip_levels = static_cast<uint32_t>(std::floor(std::log2(std::max(swapchain.extent.width, swapchain.extent.height)))) + 1;
    for (uint32_t mip = 1; mip < 11; mip++)
    {
        (mip < draw_mip_levels)
            ? resource_heap_manager.add_uav(draw_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, mip)
            : resource_heap_manager.add_uav(draw_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, draw_mip_levels - 1);
    }

    VkImageUsageFlags gbuffer_flags{
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
    };

    visibility_buffer = create_image_with_view(device, allocator, image_extent, VK_FORMAT_R32G32_UINT, gbuffer_flags, VK_IMAGE_ASPECT_COLOR_BIT);
    bindless.vbuffer_srv = resource_heap_manager.add_srv(visibility_buffer, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);

    accumulation_buffers[0] = create_image(device, allocator, image_extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    accumulation_buffers[1] = create_image(device, allocator, image_extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    bindless.accum_srv = resource_heap_manager.add_srv(accumulation_buffers[0], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    resource_heap_manager.add_srv(accumulation_buffers[1], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    bindless.accum_uav = resource_heap_manager.add_uav(accumulation_buffers[0], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    resource_heap_manager.add_uav(accumulation_buffers[1], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);

    // albedo, normal, metal-roughness-occlusion, emissive
    gbuffers.emplace_back(create_image_with_view(device, allocator, image_extent, VK_FORMAT_R8G8B8A8_UNORM, gbuffer_flags, VK_IMAGE_ASPECT_COLOR_BIT));
    gbuffers.emplace_back(create_image_with_view(device, allocator, image_extent, VK_FORMAT_R16G16B16A16_SFLOAT, gbuffer_flags, VK_IMAGE_ASPECT_COLOR_BIT));
    gbuffers.emplace_back(create_image_with_view(device, allocator, image_extent, VK_FORMAT_R8G8B8A8_UNORM, gbuffer_flags, VK_IMAGE_ASPECT_COLOR_BIT));
    gbuffers.emplace_back(create_image_with_view(device, allocator, image_extent, VK_FORMAT_R8G8B8A8_UNORM, gbuffer_flags, VK_IMAGE_ASPECT_COLOR_BIT));
    bindless.gbuffer_srv = resource_heap_manager.add_srv(gbuffers[0], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    resource_heap_manager.add_srv(gbuffers[1], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    resource_heap_manager.add_srv(gbuffers[2], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    resource_heap_manager.add_srv(gbuffers[3], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);

    depth_image.format = VK_FORMAT_D32_SFLOAT;
    depth_image = create_image_with_view(device, allocator, image_extent, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
    bindless.depth_srv = resource_heap_manager.add_srv(depth_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT);

    for (size_t idx = 0; idx < cascade_data.size(); idx++)
    {
        cascade_data[idx].shadow_map = create_image_with_view(
            device,
            allocator,
            VkExtent3D{ SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1 },
            VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT
        );
        auto shadowmap_id = resource_heap_manager.add_srv(cascade_data[idx].shadow_map, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT);
        if (idx == 0)
            bindless.shadowmap_srv = shadowmap_id;
    }

    VkExtent3D depth_pyramid_extent{};
    depth_pyramid_extent.width = nearest_pow2(swapchain.extent.width);
    depth_pyramid_extent.height = nearest_pow2(swapchain.extent.height);
    depth_pyramid_extent.depth = 1;

    depth_pyramid = create_image(
        device,
        allocator,
        depth_pyramid_extent,
        VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        0,
        true
    );
    depth_pyramid_level_count = static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(std::max(depth_pyramid_extent.width, depth_pyramid_extent.height))))) + 1;
    bindless.depth_pyramid_srv = resource_heap_manager.add_srv(depth_pyramid, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_ASPECT_COLOR_BIT);
    bindless.depth_pyramid_uav = resource_heap_manager.add_uav(depth_pyramid, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    // we reserve enough slots in image_cache to handle 1920x1080
    // this makes it easier updating descriptors if window is resized
    for (size_t mip = 1; mip < 11; mip++)
    {
        if (mip < depth_pyramid_level_count)
            resource_heap_manager.add_uav(depth_pyramid, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_ASPECT_COLOR_BIT, mip);
        else
            resource_heap_manager.add_uav(depth_pyramid, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_ASPECT_COLOR_BIT, depth_pyramid_level_count - 1);
    }

    // global light list
    std::mt19937 mt(42);
    std::uniform_real_distribution<float> pos(-1.0f, 1.0f);
    std::uniform_real_distribution<float> color(0.f, 1.0f);

    std::vector<PointLight> light_data(MAX_POINT_LIGHTS);

    float light_area = 10.f; // in radius
    float light_radius = 3.f;

    for (size_t i = 0; i < MAX_POINT_LIGHTS; i++)
    {
        light_data[i].pos = glm::vec4(pos(mt) * light_area, std::abs(pos(mt) * light_area), pos(mt) * light_area, light_radius); // pos & radius
        glm::vec3 adjusted_color = glm::vec3(color(mt), color(mt), color(mt));
        light_data[i].color = glm::vec4(adjusted_color, 1.0);
    }

    light_buffer = create_buffer_with_data(device, graphics_queue, imm_fence, imm_command_pool, imm_command_buffer, allocator, light_data.data(), MAX_POINT_LIGHTS * sizeof(PointLight));

    const uint32_t total_clusters = CLUSTER_X * CLUSTER_Y * CLUSTER_DEPTH_SLICES;

    light_cluster_buffer = create_buffer(allocator, total_clusters * sizeof(ClusterAABB), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    // TODO: could use smaller more conservative size
    light_index_buffer = create_buffer(allocator, total_clusters * MAX_POINT_LIGHTS * sizeof(uint32_t), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    light_grid_buffer = create_buffer(allocator, total_clusters * sizeof(LightGrid), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    light_count_buffer = create_buffer(allocator, sizeof(uint32_t), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    const char* hdri_path = { "assets/pisa.hdr" };
    float* data{};
    int width{};
    int height{};
    int channels{};

    data = stbi_loadf(hdri_path, &width, &height, &channels, STBI_rgb_alpha);
    auto extent = VkExtent3D(static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1);

    hdri = upload_image(
        device,
        graphics_queue,
        imm_fence,
        imm_command_pool,
        imm_command_buffer,
        allocator,
        (void*)data,
        4,
        extent,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    bindless.hdri_srv = resource_heap_manager.add_srv(hdri, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);

    extent.width /= 4;
    extent.height = extent.width;

    skybox_cubemap = create_cubemap(
        device,
        allocator,
        extent,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        0,
        true
    );
    bindless.skybox_srv = resource_heap_manager.add_srv(skybox_cubemap, VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_ASPECT_COLOR_BIT);
    scene_data.skybox_id = bindless.skybox_srv;
    bindless.skybox_uav = resource_heap_manager.add_uav(skybox_cubemap, VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_ASPECT_COLOR_BIT);

    irradiance_cubemap = create_cubemap(
        device,
        allocator,
        VkExtent3D{ 64, 64, 1 },
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    // TODO: irradiance map is never used but leaving it in here for future debugging purposes
    bindless.irradiance_srv = resource_heap_manager.add_srv(irradiance_cubemap, VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_ASPECT_COLOR_BIT);
    scene_data.irradiance_id = bindless.irradiance_srv;
    bindless.irradiance_uav = resource_heap_manager.add_uav(irradiance_cubemap, VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_ASPECT_COLOR_BIT);

    prefiltered_envmap = create_cubemap(
        device,
        allocator,
        VkExtent3D{ 512, 512, 1 },
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        0,
        true
    );
    bindless.prefiltered_srv = resource_heap_manager.add_srv(prefiltered_envmap, VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_ASPECT_COLOR_BIT);
    scene_data.prefiltered_id = bindless.prefiltered_srv;
    auto prefiltered_mips = static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(std::max(prefiltered_envmap.extent.width, prefiltered_envmap.extent.height))))) + 1;
    bindless.prefiltered_uav = resource_heap_manager.add_uav(prefiltered_envmap, VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    for (auto mip = 1; mip < prefiltered_mips; mip++)
        resource_heap_manager.add_uav(prefiltered_envmap, VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_ASPECT_COLOR_BIT, mip);

    brdf_lut = create_image(
        device,
        allocator,
        VkExtent3D{ 128, 128, 1 },
        VK_FORMAT_R16G16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    bindless.brdf_srv = resource_heap_manager.add_srv(brdf_lut, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    scene_data.brdf_id = bindless.brdf_srv;
    bindless.brdf_uav = resource_heap_manager.add_uav(brdf_lut, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);

    perlin_noise = create_3d_image(
        device,
        allocator,
        VkExtent3D{ 128, 128, 128 },
        VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    bindless.perlin_srv = resource_heap_manager.add_srv(perlin_noise, VK_IMAGE_VIEW_TYPE_3D, VK_IMAGE_ASPECT_COLOR_BIT);
    bindless.perlin_uav = resource_heap_manager.add_uav(perlin_noise, VK_IMAGE_VIEW_TYPE_3D, VK_IMAGE_ASPECT_COLOR_BIT);

    for (auto& tex : scattering_extinction_tex)
    {
        tex = create_3d_image(
            device,
            allocator,
            VkExtent3D{ VOLUMETRIC_FROXEL_X, VOLUMETRIC_FROXEL_Y, VOLUMETRIC_FROXEL_Z },
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
    }
    bindless.scattering_extinction_srv = resource_heap_manager.add_srv(scattering_extinction_tex[0], VK_IMAGE_VIEW_TYPE_3D, VK_IMAGE_ASPECT_COLOR_BIT);
    resource_heap_manager.add_srv(scattering_extinction_tex[1], VK_IMAGE_VIEW_TYPE_3D, VK_IMAGE_ASPECT_COLOR_BIT);
    bindless.scattering_extinction_uav = resource_heap_manager.add_uav(scattering_extinction_tex[0], VK_IMAGE_VIEW_TYPE_3D, VK_IMAGE_ASPECT_COLOR_BIT);
    resource_heap_manager.add_uav(scattering_extinction_tex[1], VK_IMAGE_VIEW_TYPE_3D, VK_IMAGE_ASPECT_COLOR_BIT);

    light_scattering_tex = create_3d_image(
        device,
        allocator,
        VkExtent3D{ VOLUMETRIC_FROXEL_X, VOLUMETRIC_FROXEL_Y, VOLUMETRIC_FROXEL_Z },
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    bindless.light_scattering_srv = resource_heap_manager.add_srv(light_scattering_tex, VK_IMAGE_VIEW_TYPE_3D, VK_IMAGE_ASPECT_COLOR_BIT);
    bindless.light_scattering_uav = resource_heap_manager.add_uav(light_scattering_tex, VK_IMAGE_VIEW_TYPE_3D, VK_IMAGE_ASPECT_COLOR_BIT);

    integrated_light_scattering_tex = create_3d_image(
        device,
        allocator,
        VkExtent3D{ VOLUMETRIC_FROXEL_X, VOLUMETRIC_FROXEL_Y, VOLUMETRIC_FROXEL_Z },
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    bindless.integrated_light_scattering_srv = resource_heap_manager.add_srv(integrated_light_scattering_tex, VK_IMAGE_VIEW_TYPE_3D, VK_IMAGE_ASPECT_COLOR_BIT);
    bindless.integrated_light_scattering_uav = resource_heap_manager.add_uav(integrated_light_scattering_tex, VK_IMAGE_VIEW_TYPE_3D, VK_IMAGE_ASPECT_COLOR_BIT);

    {
        // noise from: https://github.com/electronicarts/fastnoise
        const char* blue_noise_path = { "assets/bluenoise32.png" };
        int width{};
        int height{};
        int channels{};
        int desired_channels = 4;
        stbi_uc* data = stbi_load(blue_noise_path, &width, &height, &channels, desired_channels);
        auto extent = VkExtent3D(static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1);
        blue_noise_tex = upload_image(
            device,
            graphics_queue,
            imm_fence,
            imm_command_pool,
            imm_command_buffer,
            allocator,
            data,
            static_cast<uint32_t>(desired_channels),
            extent,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_STORAGE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
    }
    bindless.blue_noise_uav = resource_heap_manager.add_uav(blue_noise_tex, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);

    main_deletion_queue.push_function(
        [&]()
        {
            destroy_buffer(allocator, light_buffer);
            destroy_buffer(allocator, light_cluster_buffer);
            destroy_buffer(allocator, light_index_buffer);
            destroy_buffer(allocator, light_grid_buffer);
            destroy_buffer(allocator, light_count_buffer);
            destroy_image(device, allocator, hdri);
            destroy_image(device, allocator, skybox_cubemap);
            destroy_image(device, allocator, irradiance_cubemap);
            destroy_image(device, allocator, prefiltered_envmap);
            destroy_image(device, allocator, brdf_lut);

            for (auto& cascade : cascade_data)
            {
                destroy_image(device, allocator, cascade.shadow_map);
            }

            destroy_image(device, allocator, perlin_noise);
            destroy_image(device, allocator, scattering_extinction_tex[0]);
            destroy_image(device, allocator, scattering_extinction_tex[1]);
            destroy_image(device, allocator, light_scattering_tex);
            destroy_image(device, allocator, integrated_light_scattering_tex);
            destroy_image(device, allocator, blue_noise_tex);
        }
    );

    {
        auto screen_pixels = swapchain.extent.width * swapchain.extent.height;
        oit_buffer = create_buffer(
            allocator,
            screen_pixels * sizeof(OITData),
            0,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
        );
        bindless.oit = resource_heap_manager.add_buffer(oit_buffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

        immediate_submit(
            device,
            graphics_queue,
            imm_fence,
            imm_command_pool,
            imm_command_buffer,
            [&](VkCommandBuffer cmd)
            {
                vkCmdFillBuffer(cmd, oit_buffer.buffer, 0, VK_WHOLE_SIZE, 0x3F800000);
            }
        );
    }

    sh_buffer = create_buffer(allocator, 27 * sizeof(float), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    luminance_buffer = create_buffer(allocator, 256 * sizeof(uint32_t), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    luminance_avg_buffer = create_buffer(allocator, sizeof(float), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    // limit of ~16.7 meshlets, ~64mb
    // TODO: implement error handling/limit check in shader; just drop the meshlets?
    cluster_indices = create_buffer(allocator, MESHLET_LIMIT * sizeof(uint32_t), 0, VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT);

    dispatch_buffer = create_buffer(
        allocator,
        3 * sizeof(uint32_t),
        0,
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
    );
    meshlet_dispatch_buffer = create_buffer(
        allocator,
        3 * sizeof(uint32_t),
        0,
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
    );

    auto count_size = 2 * sizeof(uint32_t);
    auto draw_commands_size = (MAX_OPAQUE_DRAWS + MAX_ALPHACLIP_DRAWS) * sizeof(VkDrawIndexedIndirectCommand);
    draw_indirect_buffer = create_buffer(
        allocator,
        (count_size + draw_commands_size) * NUMBER_OF_CASCADES,
        0,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
    );

    {
        spd_counter_buffer = create_buffer(allocator, sizeof(uint32_t), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        immediate_submit(
            device,
            graphics_queue,
            imm_fence,
            imm_command_pool,
            imm_command_buffer,
            [&](VkCommandBuffer cmd)
            {
                vkCmdFillBuffer(cmd, spd_counter_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
            }
        );
    }
}

void VulkanEngine::init_renderables(int file_count, char** file_paths)
{
    render_scene.mask_pass.type = RenderScene::MeshPassType::Mask;
    render_scene.opaque_pass.type = RenderScene::MeshPassType::Opaque;
    render_scene.transparent_pass.type = RenderScene::MeshPassType::Transparent;
    render_scene.transmission_pass.type = RenderScene::MeshPassType::Transparent;

    auto start = SDL_GetTicks();

    {
        std::vector<std::string> asset_paths(file_count - 1);
        for (int i = 1; i < file_count; i++)
        {
            asset_paths[i - 1] = file_paths[i];
        }
        auto asset_file = load_gltfs(device, graphics_queue, imm_fence, imm_command_pool, imm_command_buffer, allocator, &resource_heap_manager, asset_paths);
        assert(asset_file.has_value());
        loaded_scene = std::move(*asset_file);
    }

    auto end = SDL_GetTicks();
    auto elapsed = end - start;
    fmt::println("load gltf: {}ms", elapsed);

    for (const auto& n : loaded_scene->top_nodes)
    {
        register_object(n.get(), glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, 2)));
    }

#ifdef STRESS_TEST
    std::mt19937 mt(42);
    auto draw_radius = 400.0f;
    auto draw_count = 500'000;

    std::vector<glm::mat4> transforms(draw_count);
    for (size_t i = 0; i < draw_count; i++)
    {
        const float x = static_cast<float>(mt()) / static_cast<float>(mt.max()) * draw_radius - draw_radius * 0.5f;
        const float y = static_cast<float>(mt()) / static_cast<float>(mt.max()) * draw_radius - draw_radius * 0.5f;
        const float z = static_cast<float>(mt()) / static_cast<float>(mt.max()) * -draw_radius;

        glm::mat4 t = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
        glm::vec3 axis = glm::normalize(
            glm::vec3(
                static_cast<float>(mt()) / static_cast<float>(mt.max()),
                static_cast<float>(mt()) / static_cast<float>(mt.max()),
                static_cast<float>(mt()) / static_cast<float>(mt.max())
            )
        );
        glm::mat4 r = glm::rotate(glm::mat4(1.0f), glm::radians(static_cast<float>(mt()) / static_cast<float>(mt.max()) * 360.0f), axis);
        glm::mat4 s = glm::scale(glm::mat4(1.0f), glm::vec3(static_cast<float>(mt()) / static_cast<float>(mt.max())) + 1.0f);
        transforms[i] = t * r * s;
    }

    for (auto& transform : transforms)
    {
        for (const auto& n : loaded_scene->top_nodes)
        {
            register_object(n.get(), transform);
        }
    }
#endif

    upload_scene_data_to_buffers();

    loaded_scene->indices.clear();
    loaded_scene->vertices.clear();
    loaded_scene->meshlet_indices.clear();
    loaded_scene->meshlets.clear();
    loaded_scene->materials.clear();
    // TODO: don't clear nodes if we have animation
    loaded_scene->top_nodes.clear();
}

void VulkanEngine::register_object(const Node* node, const glm::mat4& top_matrix)
{
    glm::mat4 node_matrix = top_matrix * node->world_transform;
    if (node->mesh_asset != nullptr)
    {
        auto it = render_scene.mesh_cache.find(node->mesh_asset.get());
        uint32_t handle = -1;
        bool found = it != render_scene.mesh_cache.end();
        if (found)
            handle = it->second;
        else
            render_scene.mesh_cache[node->mesh_asset.get()] = static_cast<uint32_t>(render_scene.meshes.size());

        for (size_t i = 0; i < node->mesh_asset->mesh.size(); i++)
        {
            ObjectData obj{};

            glm::vec3 translation{};
            glm::vec3 scale{};
            glm::vec4 rotation{};

            decompose_transform(node_matrix, translation, scale, rotation);

            // TODO: handle non uniform scaling
            obj.translation = translation;
            obj.scale = std::max(std::max(scale.x, scale.y), scale.z);
            obj.orientation = glm::quat(rotation.w, rotation.x, rotation.y, rotation.z);

            const MeshData& mesh = node->mesh_asset->mesh[i];
            obj.material_id = mesh.material_id;
            // note: storing # of meshlets for lod 0; we apply offset after registering objects
            obj.meshlet_bit_offset = mesh.meshlet_bits;

            if (found)
            {
                obj.mesh_id = static_cast<uint32_t>(handle + i);
            }
            else
            {
                obj.mesh_id = static_cast<uint32_t>(render_scene.meshes.size());

                render_scene.meshes.emplace_back(
                    Mesh{
                        .center = mesh.center,
                        .radius = mesh.radius,
                        .lod_count = mesh.lod_count,
                        .vertex_offset = mesh.vertex_offset,
                        .mesh_lods = mesh.mesh_lods }
                );
            }

            switch (mesh.pass)
            {
            case MaterialPass::Mask:
                obj.post_pass = 1;
                break;
            case MaterialPass::Blend:
                obj.post_pass = 2;
                break;
            case MaterialPass::Transmission:
                obj.post_pass = 3;
                break;
            default: // Opaque
                obj.post_pass = 0;
                break;
            }

            auto render_id = static_cast<uint32_t>(render_scene.renderables.size());
            render_scene.renderables.push_back(obj);

            switch (mesh.pass)
            {
            case MaterialPass::Opaque:
                render_scene.opaque_pass.unbatched_objects.push_back(render_id);
                break;
            case MaterialPass::Mask:
                render_scene.mask_pass.unbatched_objects.push_back(render_id);
                break;
            case MaterialPass::Blend:
            case MaterialPass::Transmission:
                render_scene.transparent_pass.unbatched_objects.push_back(render_id);
                break;
            }
        }
    }

    for (const auto& c : node->children)
        register_object(c.get(), top_matrix);
}

void VulkanEngine::update_scene()
{
    main_camera.far = static_cast<float>(cvar_system->get_int_cvar("draw_distance"));
    main_camera.update(static_cast<float>(stats.deltatime));

    scene_data.view = main_camera.get_view_matrix();
    scene_data.proj = main_camera.perspective;

    if (cvar_system->get_int_cvar("taa"))
    {
        auto idx = frame_number % 8;
        auto offset_projection = glm::translate(glm::mat4(1.0f), glm::vec3(jitter_offset[idx].x, jitter_offset[idx].y, 0.0));
        scene_data.proj = offset_projection * scene_data.proj;
    }

    scene_data.previous_viewproj = scene_data.viewproj;
    scene_data.viewproj = scene_data.proj * scene_data.view;
    scene_data.inverse_viewproj = glm::inverse(scene_data.viewproj);
    last_view = freeze_camera ? last_view : scene_data.view;
    last_proj = freeze_camera ? last_proj : scene_data.proj;

    // scene_data.sunlight_dir = glm::vec4(7.75, 12.5, 12.5, 1.);
    // scene_data.sunlight_dir = glm::vec4(0.0, 0.0, -12.5, 1.);
    scene_data.sunlight_dir = glm::vec4(0.001, 12.0, 3.0, 1.);
    // scene_data.sunlight_dir = glm::vec4(0.001, 12.0, 0.0, 1.);
    scene_data.sunlight_color = glm::vec4(1.0, 1.0, 1.0, 1.0);

    if (cvar_system->get_int_cvar("shadows") && !cvar_system->get_int_cvar("shadows_rt"))
    {
        update_cascade();
        for (size_t i = 0; i < cascade_data.size(); i++)
        {
            scene_data.shadow_transforms[i] = cascade_data[i].viewproj;
        }
    }

    scene_data.camera_pos = glm::vec4(main_camera.position, 1.0);

    auto elapsed_ms = SDL_GetTicks();
    auto ms_per_orbit = 10000;
    float rot_angle = static_cast<float>(elapsed_ms % ms_per_orbit) / static_cast<float>(ms_per_orbit) * 360.0f;
    scene_data.light_rot = glm::rotate(glm::mat4(1.0f), glm::radians(rot_angle), glm::vec3(0, 1, 0));
    scene_data.light_rot = glm::mat4(1.0f);
}

void VulkanEngine::resolve_taa(VkCommandBuffer cmd)
{
    ShaderPass current_pass = *shader_passes["resolve_taa"];
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

    struct PushConstant
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
    } pc;

    auto jitter_count = jitter_offset.size();
    auto current_jitter = jitter_offset[frame_number % jitter_count];
    auto previous_jitter = jitter_offset[(frame_number - 1) % jitter_count];
    pc.jitter_offset = glm::vec4(current_jitter, previous_jitter);
    pc.screen_size = glm::vec2(static_cast<float>(swapchain.extent.width), static_cast<float>(swapchain.extent.height));
    pc.current_id = bindless.draw_srv;
    pc.history_id = bindless.accum_srv + ((frame_number + 1) % 2);
    pc.resolve_id = bindless.accum_uav + (frame_number % 2);
    pc.depth_id = bindless.depth_srv;
    pc.velocity_id = 0; // unused
    pc.variance_clipping = cvar_system->get_int_cvar("taa.variance_clip");
    pc.history_filter = cvar_system->get_int_cvar("taa.catmull_rom");
    pc.local_filter = cvar_system->get_int_cvar("taa.mitchell");
    pc.ycocg = cvar_system->get_int_cvar("taa.ycocg");
    pc.valid_history = first_frame ? 0 : 1;
    pc.dynamic = cvar_system->get_int_cvar("taa.dynamic");
    first_frame = false; // set this elsewhere?

    VkPushDataInfoEXT push_data_info{};
    push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
    push_data_info.data = { &pc, sizeof(PushConstant) };
    vkCmdPushDataEXT(cmd, &push_data_info);
    auto groupcount_x = get_groupcount(swapchain.extent.width, WARP_SIZE);
    auto groupcount_y = get_groupcount(swapchain.extent.height, WARP_SIZE);
    vkCmdDispatch(cmd, groupcount_x, groupcount_y, 1);
}

void VulkanEngine::update_cascade()
{
    // https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-10-parallel-split-shadow-maps-programmable-gpus
    float near = static_cast<float>(cvar_system->get_int_cvar("shadows.distance"));
    float far = main_camera.far;
    float m = static_cast<float>(NUMBER_OF_CASCADES);
    float range = far - near;
    float ratio = far / near;
    float lambda = cvar_system->get_float_cvar("shadows.cascade_split");

    for (int idx = 0; idx < NUMBER_OF_CASCADES; idx++)
    {
        auto i = static_cast<float>(idx + 1); // i = [1, m]
        float log = near * std::pow(ratio, i / m);
        float uniform = near + range * i / m;
        float d = lambda * (log - uniform) + uniform; // world space
        cascade_data[idx].split_ratio = (d - near) / range;
        scene_data.cascade_splits[idx] = d * -1.0f;
    }

    auto light_dir = glm::normalize(glm::vec3(scene_data.sunlight_dir));

    glm::mat4 view = main_camera.get_view_matrix();

    glm::mat4 proj = glm::perspective(
        glm::radians(main_camera.fov),
        static_cast<float>(swapchain.extent.width) / static_cast<float>(swapchain.extent.height),
        static_cast<float>(cvar_system->get_int_cvar("shadows.distance")),
        main_camera.near
    );
    glm::mat4 inv_viewproj = glm::inverse(proj * view);

    std::array<glm::vec3, 8> frustum_corners{
        glm::vec3(-1, 1, 1),
        glm::vec3(1, 1, 1),
        glm::vec3(-1, -1, 1),
        glm::vec3(1, -1, 1),
        glm::vec3(-1, 1, 0),
        glm::vec3(1, 1, 0),
        glm::vec3(-1, -1, 0),
        glm::vec3(1, -1, 0), // reverse depth order, flipping z values will be incorrect
    };

    for (auto& frustum_corner : frustum_corners)
    {
        glm::vec4 corner = inv_viewproj * glm::vec4(frustum_corner, 1.0);
        frustum_corner = glm::vec3(corner / corner.w);
    }

    float last_split = 0.0;
    for (size_t i = 0; i < cascade_data.size(); i++)
    {
        float current_split = cascade_data[i].split_ratio;

        std::array<glm::vec3, 8> transformed_corners = frustum_corners;

        for (size_t corner_i = 0; corner_i < 4; corner_i++)
        {
            glm::vec3 distance = transformed_corners[corner_i + 4] - transformed_corners[corner_i];
            transformed_corners[corner_i + 4] = transformed_corners[corner_i] + distance * current_split; // far plane
            transformed_corners[corner_i] = transformed_corners[corner_i] + distance * last_split; // near plane
        }
        last_split = current_split;

        // note: try ritter's for tighter stable cascades?
        glm::vec3 center{};
        for (auto transformed_corner : transformed_corners)
        {
            center += transformed_corner;
        }
        center /= 8.0f;

        float radius{};
        for (auto transformed_corner : transformed_corners)
        {
            float dist = glm::length(transformed_corner - center);
            radius = std::max(dist, radius);
        }

        // for shadow culling
        {
            scene_data.shadow_views[i] = glm::lookAt(center + radius * light_dir, center, glm::vec3(0, 1, 0));
            scene_data.shadow_widths[i] = radius;
        }

        // stable csm via snapping projection matrix - https://github.com/TheRealMJP/Shadows/blob/master/Shadows/SetupShadows.hlsl
        // stable csm via snapping frustum center, less robust - https://alextardif.com/shadowmapping.html
        glm::mat4 shadow_view = glm::lookAt(center + radius * light_dir, center, glm::vec3(0, 1, 0));
        glm::mat4 shadow_proj = glm::ortho(-radius, radius, -radius, radius, radius * 2.0f, 0.0f);

        /*
		    glm::vec2 shadow_origin = glm::vec2(0.0);
		    shadow_origin = shadow_proj * shadow_view * glm::vec4(shadow_origin, 0.0, 1.0);
		    shadow_origin *= (SHADOW_MAP_SIZE / 2.0f);

		    glm::vec2 rounded_origin = glm::round(shadow_origin);
		    glm::vec2 offset = rounded_origin - shadow_origin;
		    offset *= (2.0f / SHADOW_MAP_SIZE);

		    shadow_proj[3][0] += offset.x;
		    shadow_proj[3][1] += offset.y;
		*/

        cascade_data[i].viewproj = shadow_proj * shadow_view;
    }
}

void VulkanEngine::init_imgui()
{
    VkDescriptorPoolSize pool_size{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE };

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;

    VkDescriptorPool imgui_pool{};
    vkCreateDescriptorPool(device, &pool_info, nullptr, &imgui_pool);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls

    // Setup Platform/Renderer backends
    ImGui_ImplSDL3_InitForVulkan(window);
    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.ApiVersion = VK_API_VERSION_1_4;
    init_info.Instance = instance;
    init_info.PhysicalDevice = physical_device;
    init_info.Device = device;
    init_info.QueueFamily = graphics_queue_family;
    init_info.Queue = graphics_queue;
    init_info.DescriptorPool = imgui_pool;
    init_info.MinImageCount = 2; // TODO: are these related to FIF or swapchain count?
    init_info.ImageCount = 2; // TODO: are these related to FIF or swapchain count?
    init_info.UseDynamicRendering = true;
    VkPipelineRenderingCreateInfo render_info{};
    render_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    render_info.colorAttachmentCount = 1;
    auto swapchain_image_format = VK_FORMAT_B8G8R8A8_UNORM; // TODO: do not hardcore in case of mismatch
    render_info.pColorAttachmentFormats = &swapchain_image_format;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo = render_info;

    // https://github.com/ocornut/imgui/issues/4854#issuecomment-1362380609 FOR VOLK COMPATIBILITY
    ImGui_ImplVulkan_LoadFunctions(
        0,
        [](const char* function_name, void* vulkan_instance)
        {
            return vkGetInstanceProcAddr(*(static_cast<VkInstance*>(vulkan_instance)), function_name);
        },
        &instance
    );

    ImGui_ImplVulkan_Init(&init_info);

    main_deletion_queue.push_function(
        [&, imgui_pool]()
        {
            ImGui_ImplVulkan_Shutdown();
            ImGui_ImplSDL3_Shutdown();
            ImGui::DestroyContext();
            vkDestroyDescriptorPool(device, imgui_pool, nullptr);
        }
    );
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView swapchain_view)
{
    VkRenderingAttachmentInfo color_attachment{};
    color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachment.imageView = swapchain_view;
    color_attachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo render_info{};
    render_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    render_info.renderArea = VkRect2D{ VkOffset2D{ 0, 0 }, swapchain.extent };
    render_info.layerCount = 1;
    render_info.colorAttachmentCount = 1;
    render_info.pColorAttachments = &color_attachment;

    vkCmdBeginRendering(cmd, &render_info);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRendering(cmd);
}

void VulkanEngine::upload_scene_data_to_buffers()
{
    VkBufferUsageFlags ray_tracing_flags = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    vertex_buffer = create_buffer_with_data(
        device,
        graphics_queue,
        imm_fence,
        imm_command_pool,
        imm_command_buffer,
        allocator,
        loaded_scene->vertices.data(),
        loaded_scene->vertices.size() * sizeof(Vertex),
        ray_tracing_flags
    );

    index_buffer = create_buffer_with_data(
        device,
        graphics_queue,
        imm_fence,
        imm_command_pool,
        imm_command_buffer,
        allocator,
        loaded_scene->indices.data(),
        loaded_scene->indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | ray_tracing_flags
    );
    meshlet_indices = create_buffer_with_data(
        device,
        graphics_queue,
        imm_fence,
        imm_command_pool,
        imm_command_buffer,
        allocator,
        loaded_scene->meshlet_indices.data(),
        loaded_scene->meshlet_indices.size() * sizeof(uint32_t)
    );

    meshlet_buffer = create_buffer_with_data(
        device,
        graphics_queue,
        imm_fence,
        imm_command_pool,
        imm_command_buffer,
        allocator,
        loaded_scene->meshlets.data(),
        loaded_scene->meshlets.size() * sizeof(Meshlet)
    );

    material_buffer = create_buffer_with_data(
        device,
        graphics_queue,
        imm_fence,
        imm_command_pool,
        imm_command_buffer,
        allocator,
        loaded_scene->materials.data(),
        loaded_scene->materials.size() * sizeof(MaterialData)
    );

    {
        uint32_t meshlet_visibility_offset{};
        for (auto& renderable : render_scene.renderables)
        {
            // meshlet count for LOD 0 only
            uint32_t meshlet_count = renderable.meshlet_bit_offset;
            renderable.meshlet_bit_offset = meshlet_visibility_offset;
            meshlet_visibility_offset += meshlet_count;
        }
        render_scene.total_meshlets_bits = meshlet_visibility_offset;
    }

    object_buffer = create_buffer_with_data(
        device,
        graphics_queue,
        imm_fence,
        imm_command_pool,
        imm_command_buffer,
        allocator,
        render_scene.renderables.data(),
        render_scene.renderables.size() * sizeof(ObjectData),
        0
    );

    mesh_buffer = create_buffer_with_data(
        device,
        graphics_queue,
        imm_fence,
        imm_command_pool,
        imm_command_buffer,
        allocator,
        render_scene.meshes.data(),
        render_scene.meshes.size() * sizeof(Mesh),
        0
    );

    {
        std::array<RenderScene::MeshPass*, 3> passes = { &render_scene.opaque_pass, &render_scene.mask_pass, &render_scene.transparent_pass };
        unsigned int total = 0;
        std::vector<uint32_t> staging{};
        for (RenderScene::MeshPass* pass : passes)
        {
            pass->indices_offset = total;
            total += static_cast<unsigned int>(pass->unbatched_objects.size());

            for (unsigned int unbatched_object : pass->unbatched_objects)
            {
                staging.push_back(unbatched_object);
            }
        }

        indices_buffer = create_buffer_with_data(
            device,
            graphics_queue,
            imm_fence,
            imm_command_pool,
            imm_command_buffer,
            allocator,
            staging.data(),
            total * sizeof(uint32_t)
        );
    }

    // allocating for worst case
    vis_buffer = create_buffer(
        allocator,
        render_scene.renderables.size() * sizeof(uint32_t),
        0,
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
    );

    // TODO: implement limit, currently shader side has 1000000 hardcoded
    // TODO: modify with shadows in mind
    prefix_sum_buffer = create_buffer(
        allocator,
        sizeof(uint64_t) + render_scene.renderables.size() * sizeof(PrefixSumData),
        0,
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
    );

    immediate_submit(
        device,
        graphics_queue,
        imm_fence,
        imm_command_pool,
        imm_command_buffer,
        [&](VkCommandBuffer cmd)
        {
            vkCmdFillBuffer(cmd, vis_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(cmd, prefix_sum_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
        }
    );

    {
        size_t meshlet_visibility_size = (render_scene.total_meshlets_bits + 31) / 32;
        meshlet_vis_buffer = create_buffer(allocator, meshlet_visibility_size * sizeof(uint32_t), 0, VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

        immediate_submit(
            device,
            graphics_queue,
            imm_fence,
            imm_command_pool,
            imm_command_buffer,
            [&](VkCommandBuffer cmd)
            {
                vkCmdFillBuffer(cmd, meshlet_vis_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
            }
        );
    }
}

// indices address, count, late & post_pass set in executecomputecull
void VulkanEngine::ready_cull_mesh(RenderScene::MeshPass& pass, CullData& cull_data, glm::mat4& proj)
{
    auto projT = glm::transpose(proj);

    auto m0 = projT[0];
    auto m1 = projT[1];
    // auto m2 = projT[2];
    auto m3 = projT[3];

    auto left_plane = m3 + m0;
    auto bottom_plane = m3 + m1;

    auto normalize_plane = [&](glm::vec4& plane)
    {
        float length = glm::length(glm::vec3(plane));
        plane /= length;
    };

    normalize_plane(left_plane);
    normalize_plane(bottom_plane);

    cull_data.view = freeze_camera ? last_view : scene_data.view;
    cull_data.frustum_planes = glm::vec4(left_plane.x, left_plane.z, bottom_plane.y, bottom_plane.z);

    // cull_data.indices_buffer_address; // set during execute
    cull_data.object_buffer_address = bda_table.object_buffer;
    cull_data.mesh_buffer_address = bda_table.mesh_buffer;
    cull_data.draw_indirect_address = bda_table.draw_indirect_buffer;
    cull_data.dispatch_buffer_address = bda_table.dispatch_buffer;
    cull_data.vis_buffer_address = bda_table.vis_buffer;
    cull_data.prefix_sum_buffer = bda_table.prefix_sum_buffer;

    // cull_data.count = static_cast<uint32_t>(pass.unbatched_objects.size()); // set during execute
    cull_data.texture_id = bindless.depth_pyramid_srv;
    cull_data.occlusion_enabled = cvar_system->get_int_cvar("occlusion_culling");

    cull_data.p00 = proj[0][0];
    cull_data.p11 = proj[1][1]; // equivalent to 1 / tan(fovy/2)
    cull_data.near = main_camera.near;
    cull_data.far = main_camera.far;

    cull_data.resolution = glm::vec2(depth_pyramid.extent.width, depth_pyramid.extent.height);
    cull_data.texture_lod = static_cast<float>(std::floor(std::log2(static_cast<float>(std::max(depth_pyramid.extent.width, depth_pyramid.extent.height)))) + 1);
    cull_data.lod_distance_factor = 2.0f / (cull_data.p11 * static_cast<float>(swapchain.extent.height));
    cull_data.lod_enabled = cvar_system->get_int_cvar("lod");
    cull_data.task_submit = cvar_system->get_int_cvar("mesh_shaders");
}

// count, late & post_pass set in executecomputecull
void VulkanEngine::ready_cull_meshlet(RenderScene::MeshPass& pass, ClusterCullData& cull_data, glm::mat4& proj)
{
    auto projT = glm::transpose(proj);

    auto m0 = projT[0];
    auto m1 = projT[1];
    // auto m2 = projT[2];
    auto m3 = projT[3];

    auto left_plane = m3 + m0;
    auto bottom_plane = m3 + m1;

    auto normalize_plane = [&](glm::vec4& plane)
    {
        float length = glm::length(glm::vec3(plane));
        plane /= length;
    };

    normalize_plane(left_plane);
    normalize_plane(bottom_plane);

    cull_data.view = freeze_camera ? last_view : scene_data.view;
    cull_data.frustum_planes = glm::vec4(left_plane.x, left_plane.z, bottom_plane.y, bottom_plane.z);

    cull_data.object_buffer_address = bda_table.object_buffer;
    cull_data.meshlet_buffer_address = bda_table.meshlet_buffer;
    cull_data.cluster_indices_address = bda_table.cluster_indices;
    cull_data.meshlet_dispatch_address = bda_table.meshlet_dispatch_buffer;
    cull_data.cluster_vis_address = bda_table.meshlet_vis_buffer;
    cull_data.prefix_sum_buffer = bda_table.prefix_sum_buffer;

    // cull_data.count = static_cast<uint32_t>(pass.unbatched_objects.size()); // unused
    cull_data.texture_id = bindless.depth_pyramid_srv;
    cull_data.occlusion_enabled = cvar_system->get_int_cvar("occlusion_culling");

    cull_data.p00 = proj[0][0];
    cull_data.p11 = proj[1][1];
    cull_data.near = main_camera.near;
    cull_data.far = main_camera.far;

    cull_data.resolution = glm::vec2(depth_pyramid.extent.width, depth_pyramid.extent.height);
    cull_data.texture_lod = static_cast<float>(std::floor(std::log2(static_cast<float>(std::max(depth_pyramid.extent.width, depth_pyramid.extent.height)))) + 1);
    cull_data.lod_distance_factor = 2.0f / (cull_data.p11 * static_cast<float>(swapchain.extent.height));
    cull_data.lod_enabled = cvar_system->get_int_cvar("lod");
    cull_data.task_submit = cvar_system->get_int_cvar("mesh_shaders");
}

void VulkanEngine::execute_compact_dispatch(VkCommandBuffer cmd)
{
    ShaderPass current_pass = *shader_passes["compact_dispatch"];

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

    struct PushConstant
    {
        VkDeviceAddress prefix_sum_buffer{};
        VkDeviceAddress dispatch_buffer{};
    } pc;

    pc.prefix_sum_buffer = bda_table.prefix_sum_buffer;
    pc.dispatch_buffer = bda_table.dispatch_buffer;

    VkPushDataInfoEXT push_data_info{};
    push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
    push_data_info.data = { &pc, sizeof(PushConstant) };
    vkCmdPushDataEXT(cmd, &push_data_info);
    vkCmdDispatch(cmd, 1, 1, 1);
}

void VulkanEngine::execute_cull_mesh(VkCommandBuffer cmd, const RenderScene::MeshPass& pass, CullData& cull_data, bool late, uint32_t post_pass)
{
    ShaderPass current_pass = *shader_passes["mesh_cull"];
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

    cull_data.indices_buffer_address = bda_table.indices_buffer;
    cull_data.indices_buffer_address += pass.indices_offset * sizeof(uint32_t);

    cull_data.count = static_cast<uint32_t>(pass.unbatched_objects.size());
    cull_data.late = late ? 1 : 0;
    cull_data.post_pass = post_pass;

    VkPushDataInfoEXT push_data_info{};
    push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
    push_data_info.data = { &cull_data, sizeof(CullData) };
    vkCmdPushDataEXT(cmd, &push_data_info);
    auto groupcount_x = get_groupcount(static_cast<uint32_t>(pass.unbatched_objects.size()), CULL_WGSIZE);
    vkCmdDispatch(cmd, groupcount_x, 1, 1);
}

void VulkanEngine::execute_cull_meshlet(VkCommandBuffer cmd, ClusterCullData& cull_data, VkBuffer dispatch_buffer, uint32_t offset, bool late, uint32_t post_pass)
{
    ShaderPass current_pass = *shader_passes["meshlet_cull"];
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

    // cull_data.count; // unused
    cull_data.late = late ? 1 : 0;
    cull_data.post_pass = post_pass;

    VkPushDataInfoEXT push_data_info{};
    push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
    push_data_info.data = { &cull_data, sizeof(ClusterCullData) };
    vkCmdPushDataEXT(cmd, &push_data_info);

    // TODO: offset is always 0 as alphaclip and transparent (latter probably leaving it as is in the future) are not culled with opaque,
    // hence we write over opaque's space
    vkCmdDispatchIndirect(cmd, dispatch_buffer, offset);
}

void VulkanEngine::execute_shadow_cull(VkCommandBuffer cmd)
{
    ShaderPass current_pass = *shader_passes["shadow_cull"];

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

    struct PushConstant
    {
        VkDeviceAddress object_buffer_address{};
        VkDeviceAddress mesh_buffer_address{};
        VkDeviceAddress indices_buffer_address{};
        VkDeviceAddress draw_buffer_address{};
        uint32_t count{};
        uint32_t lod_enabled{};
    } pc;

    pc.object_buffer_address = bda_table.object_buffer;
    pc.mesh_buffer_address = bda_table.mesh_buffer;
    pc.indices_buffer_address = bda_table.indices_buffer;
    pc.draw_buffer_address = bda_table.draw_indirect_buffer;

    std::vector<RenderScene::MeshPass*> passes = { &render_scene.opaque_pass, &render_scene.mask_pass };
    uint32_t cull_count{};
    for (const auto& pass : passes)
    {
        cull_count += static_cast<uint32_t>(pass->unbatched_objects.size());
    }
    pc.count = cull_count;
    pc.lod_enabled = cvar_system->get_int_cvar("lod");

    VkPushDataInfoEXT push_data_info{};
    push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
    push_data_info.data = { &pc, sizeof(PushConstant) };
    vkCmdPushDataEXT(cmd, &push_data_info);
    auto groupcount_x = get_groupcount(cull_count, CULL_WGSIZE);
    vkCmdDispatch(cmd, groupcount_x, 1, 1);
}

void VulkanEngine::render(VkCommandBuffer cmd, bool late, uint32_t post_pass)
{
    // deferred
    VkClearColorValue clear_color_value{ { 0.f, 0.f, 0.f, 1.0f } };
    VkClearValue clear_value{ .color = clear_color_value };

    std::vector<VkRenderingAttachmentInfo> rendering_attachment_infos{};
    bool visibility_rendering = cvar_system->get_int_cvar("vbuffer") && cvar_system->get_int_cvar("mesh_shaders");
    if (visibility_rendering)
    {
        VkRenderingAttachmentInfo info{};
        info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        info.imageView = visibility_buffer.view;
        info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        info.loadOp = late ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
        info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        if (late)
            info.clearValue = clear_value;

        rendering_attachment_infos.push_back(info);
    }
    else
    {
        for (int i = 0; i < gbuffers.size(); i++)
        {
            VkRenderingAttachmentInfo info{};
            info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            info.imageView = gbuffers[i].view;
            info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            info.loadOp = late ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
            info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            if (late)
                info.clearValue = clear_value;

            rendering_attachment_infos.push_back(info);
        }
    }

    VkRenderingAttachmentInfo depth_attachment{};
    depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_attachment.imageView = depth_image.view;
    depth_attachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    depth_attachment.loadOp = late ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.clearValue.depthStencil.depth = 0.f;

    VkRenderingInfo render_info{};
    render_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    render_info.renderArea = VkRect2D{ VkOffset2D{ 0, 0 }, swapchain.extent };
    render_info.layerCount = 1;
    render_info.colorAttachmentCount = static_cast<uint32_t>(rendering_attachment_infos.size());
    render_info.pColorAttachments = rendering_attachment_infos.data();
    render_info.pDepthAttachment = &depth_attachment;

    vkCmdBeginRendering(cmd, &render_info);

    VkViewport viewport{};
    viewport.x = 0;
    viewport.y = static_cast<float>(swapchain.extent.height);
    viewport.width = static_cast<float>(swapchain.extent.width);
    viewport.height = -static_cast<float>(swapchain.extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = swapchain.extent.width;
    scissor.extent.height = swapchain.extent.height;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // auto depth_bias = 0.f;
    // auto slope_scaled_depth_bias = 0.f;
    //  TODO: refactor to account for different geometry (double-sided or back face culled)
    // vkCmdSetDepthBias(cmd, -depth_bias, 0.0f, -slope_scaled_depth_bias);

    struct PushConstant
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
    } pc;

    pc.object_buffer_address = bda_table.object_buffer;
    pc.vertex_buffer_address = bda_table.vertex_buffer;
    pc.meshlet_buffer_address = bda_table.meshlet_buffer;
    pc.meshlet_indices_buffer_address = bda_table.meshlet_indices;
    pc.cluster_indices_address = bda_table.cluster_indices;
    pc.material_buffer_address = bda_table.material_buffer;
    pc.prefix_sum_buffer = bda_table.prefix_sum_buffer;
    auto jitter_count = jitter_offset.size();
    auto current_jitter = jitter_offset[frame_number % jitter_count];
    auto previous_jitter = jitter_offset[(frame_number - 1) % jitter_count];
    pc.screen_size = glm::uvec2(swapchain.extent.width, swapchain.extent.height);
    pc.jitter_offset = glm::vec4(current_jitter, previous_jitter);

    if (!cvar_system->get_int_cvar("mesh_shaders"))
    {
        auto pq = ScopedPipelineQuery(&query_manager, frame_number, cmd, get_current_frame().query_pool_pipelines, PipelineQueryType::Vertex);

        ShaderPass current_pass = post_pass == 0 ? *shader_passes["gbuffer_vert"] : *shader_passes["gbuffer_vert_alphaclip"];

        VkPushDataInfoEXT push_data_info{};
        push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
        push_data_info.data = { &pc, sizeof(PushConstant) };
        vkCmdPushDataEXT(cmd, &push_data_info);

        vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);

        // reuse opaque section for rendering alphaClipped geometry; alphaClipped reserved for shadows
        vkCmdDrawIndexedIndirectCount(
            cmd,
            draw_indirect_buffer.buffer,
            2 * sizeof(uint32_t),
            draw_indirect_buffer.buffer,
            0,
            MAX_MESH_DRAWS,
            sizeof(VkDrawIndexedIndirectCommand)
        );
    }
    else // mesh shading path
    {
        auto pq = ScopedPipelineQuery(&query_manager, frame_number, cmd, get_current_frame().query_pool_mesh_primitives, PipelineQueryType::Mesh);

        ShaderPass current_pass{};
        if (visibility_rendering)
        {
            current_pass = post_pass == 0 ? *shader_passes["vbuffer"] : *shader_passes["vbuffer_alphaclip"];
        }
        else // deferred rendering
        {
            current_pass = post_pass == 0 ? *shader_passes["gbuffer_mesh"] : *shader_passes["gbuffer_mesh_alphaclip"];
        }

        VkPushDataInfoEXT push_data_info{};
        push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
        push_data_info.data = { &pc, sizeof(PushConstant) };
        vkCmdPushDataEXT(cmd, &push_data_info);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
        vkCmdDrawMeshTasksIndirectEXT(cmd, meshlet_dispatch_buffer.buffer, 0, 1, 0);
    }

    vkCmdEndRendering(cmd);
}

void VulkanEngine::render_transparent(VkCommandBuffer cmd)
{
    VkRenderingAttachmentInfo depth_attachment{};
    depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_attachment.imageView = depth_image.view;
    depth_attachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.clearValue.depthStencil.depth = 0.f;

    VkRenderingInfo render_info{};
    render_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    render_info.renderArea = VkRect2D{ VkOffset2D{ 0, 0 }, swapchain.extent };
    render_info.layerCount = 1;
    render_info.pDepthAttachment = &depth_attachment;
    render_info.pStencilAttachment = nullptr;

    vkCmdBeginRendering(cmd, &render_info);

    VkViewport viewport{};
    viewport.x = 0;
    viewport.y = static_cast<float>(swapchain.extent.height);
    viewport.width = static_cast<float>(swapchain.extent.width);
    viewport.height = -static_cast<float>(swapchain.extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = swapchain.extent.width;
    scissor.extent.height = swapchain.extent.height;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    struct PushConstant
    {
        VkDeviceAddress object_buffer_address{};
        VkDeviceAddress vertex_buffer_address{};
        VkDeviceAddress meshlet_buffer_address{};
        VkDeviceAddress meshlet_indices_buffer_address{};
        VkDeviceAddress cluster_indices_address{};
        VkDeviceAddress material_buffer_address{};
        VkDeviceAddress prefix_sum_buffer{};
        VkDeviceAddress sh_buffer{};
        VkDeviceAddress light_buffer{};
        VkDeviceAddress light_index_buffer{};
        VkDeviceAddress light_grid_buffer{};
        glm::vec2 cluster_size{};
        glm::uvec2 screen_size{};
        float max_prefiltered_lod{};
        uint32_t framebuffer_id{};
        uint32_t volume{};
        uint32_t point_lights{};
        uint32_t scale{};
        uint32_t bias{};
    } pc;

    pc.object_buffer_address = bda_table.object_buffer;
    pc.vertex_buffer_address = bda_table.vertex_buffer;
    pc.meshlet_buffer_address = bda_table.meshlet_buffer;
    pc.meshlet_indices_buffer_address = bda_table.meshlet_indices;
    pc.cluster_indices_address = bda_table.cluster_indices;
    pc.material_buffer_address = bda_table.material_buffer;
    pc.prefix_sum_buffer = bda_table.prefix_sum_buffer;
    pc.sh_buffer = bda_table.sh_buffer;
    pc.light_buffer = bda_table.light_buffer;
    pc.light_index_buffer = bda_table.light_index_buffer;
    pc.light_grid_buffer = bda_table.light_grid_buffer;
    auto cluster_x = ceil(static_cast<float>(swapchain.extent.width) / CLUSTER_X); // # cluster dim
    auto cluster_y = ceil(static_cast<float>(swapchain.extent.height) / CLUSTER_Y); // # cluster dim
    pc.cluster_size = glm::vec2(cluster_x, cluster_y);
    pc.screen_size = glm::uvec2(swapchain.extent.width, swapchain.extent.height);
    pc.max_prefiltered_lod = static_cast<float>(std::floor(std::log2(static_cast<float>(std::max(prefiltered_envmap.extent.width, prefiltered_envmap.extent.height))))) + 1;
    pc.framebuffer_id = bindless.draw_srv;
    pc.volume = cvar_system->get_int_cvar("volume");
    pc.point_lights = cvar_system->get_int_cvar("point_lights");
    const float ratio = main_camera.far / main_camera.near;
    pc.scale = static_cast<float>(CLUSTER_DEPTH_SLICES) / std::log(ratio);
    pc.bias = static_cast<float>(CLUSTER_DEPTH_SLICES) * std::log(main_camera.near) / std::log(ratio);

    if (!cvar_system->get_int_cvar("mesh_shaders"))
    {
        auto pq = ScopedPipelineQuery(&query_manager, frame_number, cmd, get_current_frame().query_pool_pipelines, PipelineQueryType::Vertex);

        ShaderPass current_pass = *shader_passes["mlab_vert"];

        VkPushDataInfoEXT push_data_info{};
        push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
        push_data_info.data = { &pc, sizeof(PushConstant) };
        vkCmdPushDataEXT(cmd, &push_data_info);

        vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);

        vkCmdDrawIndexedIndirectCount(
            cmd,
            draw_indirect_buffer.buffer,
            2 * sizeof(uint32_t),
            draw_indirect_buffer.buffer,
            0,
            MAX_MESH_DRAWS,
            sizeof(VkDrawIndexedIndirectCommand)
        );
    }
    else // mesh shading path
    {
        auto pq = ScopedPipelineQuery(&query_manager, frame_number, cmd, get_current_frame().query_pool_mesh_primitives, PipelineQueryType::Mesh);

        ShaderPass current_pass = *shader_passes["mlab_mesh"];

        VkPushDataInfoEXT push_data_info{};
        push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
        push_data_info.data = { &pc, sizeof(PushConstant) };
        vkCmdPushDataEXT(cmd, &push_data_info);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
        vkCmdDrawMeshTasksIndirectEXT(cmd, meshlet_dispatch_buffer.buffer, 0, 1, 0);
    }

    vkCmdEndRendering(cmd);
}

void VulkanEngine::render_shadows(VkCommandBuffer cmd, uint32_t cascade_idx)
{
    auto pq = ScopedPipelineQuery(&query_manager, frame_number, cmd, get_current_frame().query_pool_pipelines, PipelineQueryType::Vertex);
    VkRenderingAttachmentInfo depth_attachment{};
    depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_attachment.imageView = cascade_data[cascade_idx].shadow_map.view;
    depth_attachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // depth pyramid?
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.clearValue.depthStencil.depth = 0.f;

    auto shadow_extent = VkExtent2D{ SHADOW_MAP_SIZE, SHADOW_MAP_SIZE };
    VkRenderingInfo render_info{};
    render_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    render_info.renderArea = VkRect2D{ VkOffset2D{ 0, 0 }, shadow_extent };
    render_info.layerCount = 1;
    render_info.pDepthAttachment = &depth_attachment;

    vkCmdBeginRendering(cmd, &render_info);

    VkViewport viewport{};
    viewport.x = 0;
    viewport.y = static_cast<float>(shadow_extent.height);
    viewport.width = static_cast<float>(shadow_extent.width);
    viewport.height = -static_cast<float>(shadow_extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = shadow_extent.width;
    scissor.extent.height = shadow_extent.height;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    auto depth_bias = 0.f;
    auto slope_scaled_depth_bias = 0.f;
    vkCmdSetDepthBias(cmd, -depth_bias, 0.0f, -slope_scaled_depth_bias);

    struct PushConstant
    {
        glm::mat4 viewproj{};
        VkDeviceAddress material_buffer_address{};
        VkDeviceAddress object_buffer_address{};
        VkDeviceAddress vertex_buffer_address{};
    } pc;

    pc.viewproj = cascade_data[cascade_idx].viewproj;
    pc.material_buffer_address = bda_table.material_buffer;
    pc.object_buffer_address = bda_table.object_buffer;
    pc.vertex_buffer_address = bda_table.vertex_buffer;

    {
        ShaderPass current_pass = *shader_passes["depth"];

        VkPushDataInfoEXT push_data_info{};
        push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
        push_data_info.data = { &pc, sizeof(PushConstant) };
        vkCmdPushDataEXT(cmd, &push_data_info);

        vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

        auto cascade_offset = cascade_idx * (2 * sizeof(uint32_t) + (MAX_OPAQUE_DRAWS + MAX_ALPHACLIP_DRAWS) * sizeof(VkDrawIndexedIndirectCommand));

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
        vkCmdDrawIndexedIndirectCount(
            cmd,
            draw_indirect_buffer.buffer,
            2 * sizeof(uint32_t) + cascade_offset,
            draw_indirect_buffer.buffer,
            0 + cascade_offset,
            MAX_OPAQUE_DRAWS,
            sizeof(VkDrawIndexedIndirectCommand)
        );

        if (cvar_system->get_int_cvar("alphaclip"))
        {
            current_pass = *shader_passes["depth_alphaclip"];
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);

            vkCmdDrawIndexedIndirectCount(
                cmd,
                draw_indirect_buffer.buffer,
                2 * sizeof(uint32_t) + MAX_OPAQUE_DRAWS * sizeof(VkDrawIndexedIndirectCommand) + cascade_offset,
                draw_indirect_buffer.buffer,
                sizeof(uint32_t) + cascade_offset,
                MAX_ALPHACLIP_DRAWS,
                sizeof(VkDrawIndexedIndirectCommand)
            );
        }
    }

    vkCmdEndRendering(cmd);
}

void VulkanEngine::execute_hiz_spd(VkCommandBuffer cmd)
{
    ShaderPass current_pass = *shader_passes["hiz_spd"];
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

    auto width = next_pow2(swapchain.extent.width);
    auto height = next_pow2(swapchain.extent.height);
    auto groupcount_x = get_groupcount(width, 64);
    auto groupcount_y = get_groupcount(height, 64);

    struct PushConstant
    {
        VkDeviceAddress spd_counter_buffer{};
        glm::vec2 rcp_resolution{};
        uint32_t mips{};
        uint32_t num_wgs{};
        uint32_t src_id{}; // texture to sample
        uint32_t dst_id{}; // image to write to, offset accordingly!
        uint32_t sampler_id{};
        // uint32_t wg_offset; // note: for subregion downsampling, not implemented for now
    } pc;

    pc.spd_counter_buffer = bda_table.spd_counter_buffer;
    pc.rcp_resolution = glm::vec2(1.0) / glm::vec2(width, height);
    pc.mips = depth_pyramid_level_count;
    pc.num_wgs = groupcount_x * groupcount_y;
    pc.src_id = bindless.depth_srv;
    pc.dst_id = bindless.depth_pyramid_uav;
    pc.sampler_id = DEPTH_REDUCTION_SAMPLER;

    VkPushDataInfoEXT push_data_info{};
    push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
    push_data_info.data = { &pc, sizeof(PushConstant) };
    vkCmdPushDataEXT(cmd, &push_data_info);
    vkCmdDispatch(cmd, groupcount_x, groupcount_y, 1);
}

void VulkanEngine::execute_hiz(VkCommandBuffer cmd)
{
    ShaderPass current_pass = *shader_passes["hiz"];
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

    struct PushConstant
    {
        std::array<int32_t, 2> image_size{};
        uint32_t texture_id{};
        uint32_t image_id{};
        uint32_t lod{};
    } pc;

    uint32_t mip_levels = depth_pyramid_level_count;

    for (uint32_t i = 0; i < mip_levels; i++)
    {
        int32_t width = std::max(static_cast<int32_t>(depth_pyramid.extent.width) >> i, 1);
        int32_t height = std::max(static_cast<int32_t>(depth_pyramid.extent.height) >> i, 1);
        pc.image_size = { width, height };
        pc.texture_id = i == 0 ? bindless.depth_srv : bindless.depth_pyramid_srv;
        pc.image_id = bindless.depth_pyramid_uav + i;
        pc.lod = i == 0 ? 0 : i - 1;

        VkPushDataInfoEXT push_data_info{};
        push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
        push_data_info.data = { &pc, sizeof(PushConstant) };
        vkCmdPushDataEXT(cmd, &push_data_info);

        auto groupcount_x = get_groupcount(width, WARP_SIZE);
        auto groupcount_y = get_groupcount(height, WARP_SIZE);
        vkCmdDispatch(cmd, groupcount_x, groupcount_y, 1);

        if (i < mip_levels - 1)
        {
            VkImageMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkImageSubresourceRange subresourcerange{};

            subresourcerange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            subresourcerange.baseMipLevel = i;
            subresourcerange.levelCount = 1;
            subresourcerange.layerCount = 1;
            barrier.subresourceRange = subresourcerange;
            barrier.image = depth_pyramid.image;

            VkDependencyInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            info.imageMemoryBarrierCount = 1;
            info.pImageMemoryBarriers = &barrier;

            vkCmdPipelineBarrier2(cmd, &info);
        }
    }
}

void VulkanEngine::build_cluster_grid()
{
    //> draw
    VkCommandBuffer cmd = imm_command_buffer;
    VK_CHECK(vkResetFences(device, 1, &imm_fence));

    VK_CHECK(vkResetCommandPool(device, imm_command_pool, 0));

    VkCommandBufferBeginInfo cmd_begin_info{};
    cmd_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmd_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));

    ShaderPass current_pass = *shader_passes["cluster_grid"];
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

    struct PushConstant
    {
        glm::mat4 inverse_proj{};
        VkDeviceAddress light_cluster_buffer_address{};
        glm::vec2 screen_size{};
        glm::vec2 cluster_dim{};
        float near{};
        float far{};
        uint32_t depth_slices{};
    } pc;

    pc.inverse_proj = glm::inverse(main_camera.perspective);
    pc.light_cluster_buffer_address = bda_table.light_cluster_buffer;
    pc.screen_size = glm::vec2(swapchain.extent.width, swapchain.extent.height);
    auto cluster_x = ceil(static_cast<float>(swapchain.extent.width) / CLUSTER_X); // # cluster dim
    auto cluster_y = ceil(static_cast<float>(swapchain.extent.height) / CLUSTER_Y); // # cluster dim
    pc.cluster_dim = glm::vec2(cluster_x, cluster_y);
    pc.near = main_camera.near; // reverse-z
    pc.far = main_camera.far;
    pc.depth_slices = CLUSTER_DEPTH_SLICES;

    VkPushDataInfoEXT push_data_info{};
    push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
    push_data_info.data = { &pc, sizeof(PushConstant) };
    vkCmdPushDataEXT(cmd, &push_data_info);

    vkCmdDispatch(cmd, 1, 1, CLUSTER_DEPTH_SLICES);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmd_info{};
    cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cmd_info.commandBuffer = cmd;

    VkSubmitInfo2 submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submit_info.pCommandBufferInfos = &cmd_info;
    submit_info.commandBufferInfoCount = 1;

    VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit_info, imm_fence));
    VK_CHECK(vkWaitForFences(device, 1, &imm_fence, true, 9999999999));
}

void VulkanEngine::execute_light_culling(VkCommandBuffer cmd)
{
    ShaderPass current_pass = *shader_passes["light_culling"];
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

    struct PushConstant
    {
        glm::mat4 view{};
        glm::mat4 light_rot{};
        VkDeviceAddress light_cluster_buffer_address{};
        VkDeviceAddress light_buffer_address{};
        VkDeviceAddress light_index_buffer_address{};
        VkDeviceAddress light_grid_buffer_address{};
        VkDeviceAddress light_count_buffer_address{};
    } pc;

    pc.view = scene_data.view;
    pc.light_rot = scene_data.light_rot;

    pc.light_cluster_buffer_address = bda_table.light_cluster_buffer;
    pc.light_buffer_address = bda_table.light_buffer;
    pc.light_index_buffer_address = bda_table.light_index_buffer;
    pc.light_grid_buffer_address = bda_table.light_grid_buffer;
    pc.light_count_buffer_address = bda_table.light_count_buffer;

    VkPushDataInfoEXT push_data_info{};
    push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
    push_data_info.data = { &pc, sizeof(PushConstant) };
    vkCmdPushDataEXT(cmd, &push_data_info);

    vkCmdDispatch(cmd, 1, 1, CLUSTER_DEPTH_SLICES / CLUSTER_Z);
}

void VulkanEngine::execute_shading(VkCommandBuffer cmd)
{
    ShaderPass current_pass{};
    bool visibility_rendering = cvar_system->get_int_cvar("vbuffer") && cvar_system->get_int_cvar("mesh_shaders");
    if (visibility_rendering)
    {
        current_pass = *shader_passes["resolve_vbuffer"];
        // if (cvar_system->get_int_cvar("rt"))
        //     current_pass = *shader_passes["ray_tracing"];
    }
    else
    {
        current_pass = *shader_passes["resolve_gbuffer"];
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

    struct PushConstant
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
        uint32_t volumetrics{};
        uint32_t volumetrics_tex{};
        float volumetrics_scale{};
        float volumetrics_bias{};
        glm::uvec3 volumetrics_froxel_dim{};
        uint32_t blue_noise_tex{};
    } pc;

    auto cluster_x = ceil(static_cast<float>(swapchain.extent.width) / CLUSTER_X); // # cluster dim
    auto cluster_y = ceil(static_cast<float>(swapchain.extent.height) / CLUSTER_Y); // # cluster dim
    // TODO: BA channels potentially unused, if so remove
    pc.cluster_size = glm::vec4(cluster_x, cluster_y, CLUSTER_DEPTH_SLICES, 0.0);
    pc.screen_size = glm::vec2(swapchain.extent.width, swapchain.extent.height);

    pc.light_buffer_address = bda_table.light_buffer;
    pc.light_index_buffer_address = bda_table.light_index_buffer;
    pc.light_grid_buffer_address = bda_table.light_grid_buffer;
    pc.meshlet_indices_address = bda_table.meshlet_indices;
    pc.meshlet_buffer_address = bda_table.meshlet_buffer;
    pc.vertex_buffer_address = bda_table.vertex_buffer;
    pc.object_buffer_address = bda_table.object_buffer;
    pc.material_buffer_address = bda_table.material_buffer;
    pc.index_buffer_address = bda_table.index_buffer;
    pc.mesh_buffer_address = bda_table.mesh_buffer;
    pc.sh_buffer_address = bda_table.sh_buffer;

    pc.draw_id = bindless.draw_uav;
    pc.depth_id = bindless.depth_srv;
    pc.gbuffer_id = visibility_rendering ? bindless.vbuffer_srv : bindless.gbuffer_srv;
    pc.shadow_id = bindless.shadowmap_srv;
    pc.light_culling = cvar_system->get_int_cvar("point_lights");
    pc.near = main_camera.near;

    const float light_frustum_ratio = main_camera.far / main_camera.near;
    pc.scale = static_cast<float>(CLUSTER_DEPTH_SLICES) / std::log(light_frustum_ratio);
    pc.bias = static_cast<float>(CLUSTER_DEPTH_SLICES) * std::log(main_camera.near) / std::log(light_frustum_ratio);
    pc.shadows = cvar_system->get_int_cvar("shadows");
    pc.shadows_rt = cvar_system->get_int_cvar("shadows_rt");
    pc.max_prefiltered_lod = static_cast<float>(std::floor(std::log2(static_cast<float>(std::max(prefiltered_envmap.extent.width, prefiltered_envmap.extent.height))))) + 1;
    pc.debug = cvar_system->get_int_cvar("debug.textures");
    pc.volumetrics = cvar_system->get_int_cvar("volumetric_fog") && pc.shadows && !pc.shadows_rt;
    pc.volumetrics_tex = bindless.integrated_light_scattering_srv;
    float volumetrics_slices = static_cast<float>(VOLUMETRIC_FROXEL_Z);
    const float fog_frustum_ratio = cvar_system->get_float_cvar("volumetric.far_plane") / main_camera.near;
    pc.volumetrics_scale = volumetrics_slices / std::log(fog_frustum_ratio);
    pc.volumetrics_bias = volumetrics_slices * std::log(main_camera.near) / std::log(fog_frustum_ratio);
    pc.volumetrics_froxel_dim = glm::uvec3(VOLUMETRIC_FROXEL_X, VOLUMETRIC_FROXEL_Y, VOLUMETRIC_FROXEL_Z);
    pc.blue_noise_tex = bindless.blue_noise_uav;

    VkPushDataInfoEXT push_data_info{};
    push_data_info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
    push_data_info.data = { &pc, sizeof(PushConstant) };
    vkCmdPushDataEXT(cmd, &push_data_info);

    auto groupcount_x = get_groupcount(swapchain.extent.width, 8);
    auto groupcount_y = get_groupcount(swapchain.extent.height, 8);
    vkCmdDispatch(cmd, groupcount_x, groupcount_y, 1);
}

// TODO:
// - separate into buildBLAS and buildTLAS
// - build flags
// - opaque/non opaque flags
// - figure out alignment for scratch and BLAS
// - update bit for rebuild
// - lifetime of buffers
void VulkanEngine::create_acceleration_structures()
{
    VkDeviceAddress vb_address = get_buffer_address(device, vertex_buffer.buffer);
    VkDeviceAddress ib_address = get_buffer_address(device, index_buffer.buffer);

    std::vector<Mesh>& meshes = render_scene.meshes;

    AllocatedBuffer scratch_buffer{};
    std::vector<VkAccelerationStructureKHR> blas_handles(meshes.size());

    std::vector<uint32_t> primitive_counts(meshes.size());
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> build_infos(meshes.size());
    std::vector<VkAccelerationStructureGeometryKHR> geometries(meshes.size());

    std::vector<size_t> as_offsets(meshes.size());
    std::vector<size_t> as_sizes(meshes.size());
    std::vector<size_t> scratch_offsets(meshes.size());

    size_t total_as_size = 0;
    size_t total_scratch_size = 0;

    const size_t alignment = 256;
    VkBuildAccelerationStructureFlagsKHR blas_flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    VkBuildAccelerationStructureFlagsKHR tlas_flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

    for (size_t i = 0; i < meshes.size(); ++i)
    {
        const auto& mesh = meshes[i];

        auto& build_info = build_infos[i];
        auto& geometry = geometries[i];
        auto& primitive_count = primitive_counts[i];

        uint32_t lod_index = 0;

        primitive_count = mesh.mesh_lods[lod_index].count / 3;

        VkAccelerationStructureGeometryTrianglesDataKHR triangle_data{};
        triangle_data.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        triangle_data.vertexFormat = VK_FORMAT_R16G16B16_SFLOAT;
        triangle_data.vertexData.deviceAddress = vb_address + sizeof(Vertex) * mesh.vertex_offset;
        triangle_data.vertexStride = sizeof(Vertex);
        triangle_data.maxVertex = mesh.mesh_lods[lod_index].count - 1; // max index accessed hence -1
        triangle_data.indexType = VK_INDEX_TYPE_UINT32;
        triangle_data.indexData.deviceAddress = ib_address + sizeof(uint32_t) * mesh.mesh_lods[lod_index].first_index;

        geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.geometry.triangles = triangle_data;
        // geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

        build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        build_info.geometryCount = 1;
        build_info.pGeometries = &geometry;
        build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build_info.flags = blas_flags;

        VkAccelerationStructureBuildSizesInfoKHR build_sizes{ .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };

        vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info, &primitive_count, &build_sizes);

        as_offsets[i] = total_as_size;
        as_sizes[i] = build_sizes.accelerationStructureSize;
        scratch_offsets[i] = total_scratch_size;

        total_as_size = (total_as_size + build_sizes.accelerationStructureSize + alignment - 1) & ~(alignment - 1);
        total_scratch_size = (total_scratch_size + build_sizes.buildScratchSize + alignment - 1) & ~(alignment - 1);
    }

    blas_buffer = create_buffer(allocator, total_as_size, 0, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    auto scratch_buffer_address_alignment = as_properties.minAccelerationStructureScratchOffsetAlignment;
    scratch_buffer = create_buffer(allocator, total_scratch_size, 0, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, scratch_buffer_address_alignment);

    // fmt::println("blas_buffer: {}", size_in_bytes(total_as_size));
    // fmt::println("scratch_buffer: {}", size_in_bytes(total_scratch_size));

    VkDeviceAddress scratch_address = get_buffer_address(device, scratch_buffer.buffer);
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> build_ranges(meshes.size());
    std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> build_ranges_ptrs(meshes.size());
    for (size_t i = 0; i < meshes.size(); i++)
    {
        VkAccelerationStructureCreateInfoKHR as_info{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            .buffer = blas_buffer.buffer,
            .offset = as_offsets[i],
            .size = as_sizes[i],
            .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR
        };

        vkCreateAccelerationStructureKHR(device, &as_info, nullptr, &blas_handles[i]);

        build_infos[i].scratchData.deviceAddress = scratch_address + scratch_offsets[i];
        build_infos[i].dstAccelerationStructure = blas_handles[i];

        build_ranges[i].primitiveCount = primitive_counts[i];
        build_ranges_ptrs[i] = &build_ranges[i];
    }

    immediate_submit(
        device,
        graphics_queue,
        imm_fence,
        imm_command_pool,
        imm_command_buffer,
        [&](VkCommandBuffer cmd)
        {
            vkCmdBuildAccelerationStructuresKHR(cmd, static_cast<uint32_t>(build_infos.size()), build_infos.data(), build_ranges_ptrs.data());
        }
    );

    destroy_buffer(allocator, scratch_buffer);
    std::vector<VkDeviceAddress> blas_addresses(meshes.size());

    for (size_t i = 0; i < meshes.size(); i++)
    {
        VkAccelerationStructureDeviceAddressInfoKHR address_info{};
        address_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        address_info.accelerationStructure = blas_handles[i];
        blas_addresses[i] = vkGetAccelerationStructureDeviceAddressKHR(device, &address_info);
    }

    std::vector<VkAccelerationStructureInstanceKHR> instances(render_scene.renderables.size());
    for (size_t i = 0; i < render_scene.renderables.size(); i++)
    {
        ObjectData obj = render_scene.renderables[i];

        glm::mat3 transform = glm::mat3_cast(obj.orientation) * obj.scale;
        transform = glm::transpose(transform);

        memcpy(instances[i].transform.matrix[0], &transform[0], sizeof(float) * 3);
        memcpy(instances[i].transform.matrix[1], &transform[1], sizeof(float) * 3);
        memcpy(instances[i].transform.matrix[2], &transform[2], sizeof(float) * 3);
        instances[i].transform.matrix[0][3] = obj.translation.x; // row-major
        instances[i].transform.matrix[1][3] = obj.translation.y;
        instances[i].transform.matrix[2][3] = obj.translation.z;
        instances[i].instanceCustomIndex = i; // note: instanceCustomIndex 24 bits only
        instances[i].mask = 0xFF; // lets us programatically select set of instances to trace, without rebuilding TLAS
        instances[i].flags = obj.post_pass == 0 ? VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR : VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
        instances[i].accelerationStructureReference = blas_addresses[obj.mesh_id];
    }

    tlas_instance_buffer = create_buffer_with_data(
        device,
        graphics_queue,
        imm_fence,
        imm_command_pool,
        imm_command_buffer,
        allocator,
        instances.data(),
        instances.size() * sizeof(VkAccelerationStructureInstanceKHR),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
    );

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.data.deviceAddress = get_buffer_address(device, tlas_instance_buffer.buffer);

    VkAccelerationStructureBuildGeometryInfoKHR build_info{};
    build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.flags = tlas_flags; // allow update bit for dynamic scene
    build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.geometryCount = 1;
    build_info.pGeometries = &geometry;

    uint32_t draw_count = static_cast<uint32_t>(render_scene.renderables.size());

    VkAccelerationStructureBuildSizesInfoKHR build_size{ .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info, &draw_count, &build_size);

    tlas_buffer = create_buffer(allocator, build_size.accelerationStructureSize, 0, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    scratch_buffer = create_buffer(allocator, build_size.buildScratchSize, 0, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    scratch_address = get_buffer_address(device, scratch_buffer.buffer);
    // fmt::println("tlas scratch_buffer: {}", size_in_bytes(build_size.buildScratchSize));
    // fmt::println("tlas instance buffer: {}", size_in_bytes(tlas_instance_buffer.info.size));
    // fmt::println("tlas buffer: {}", size_in_bytes(build_size.accelerationStructureSize));

    VkAccelerationStructureCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    create_info.buffer = tlas_buffer.buffer;
    create_info.size = build_size.accelerationStructureSize;
    create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    vkCreateAccelerationStructureKHR(device, &create_info, nullptr, &tlas_as);

    build_info.scratchData.deviceAddress = scratch_address;
    build_info.dstAccelerationStructure = tlas_as;

    VkAccelerationStructureBuildRangeInfoKHR build_range{};
    build_range.primitiveCount = draw_count;

    const VkAccelerationStructureBuildRangeInfoKHR* build_range_ptr = &build_range;

    immediate_submit(
        device,
        graphics_queue,
        imm_fence,
        imm_command_pool,
        imm_command_buffer,
        [&](VkCommandBuffer cmd)
        {
            vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &build_range_ptr);
        }
    );

    destroy_buffer(allocator, scratch_buffer);
    destroy_buffer(allocator, tlas_instance_buffer);

    main_deletion_queue.push_function(
        [&, blas_handles]()
        {
            destroy_buffer(allocator, blas_buffer);
            destroy_buffer(allocator, tlas_buffer);

            for (auto& blas : blas_handles)
            {
                vkDestroyAccelerationStructureKHR(device, blas, nullptr);
            }

            vkDestroyAccelerationStructureKHR(device, tlas_as, nullptr);
        }
    );

    resource_heap_manager.add_acceleration_structure(tlas_as, 0);
}

void VulkanEngine::register_queries_with_imgui()
{
    ImGui::Begin("Stats");

    ImGui::Text("Total render time:    %.3f ms", stats.cpu_time);
    ImGui::NewLine();
    timestamp_manager.add_imgui_text(frame_number);
    ImGui::NewLine();
    query_manager.add_imgui_text(frame_number);

    ImGui::End();
    ImGui::Render();
}

void VulkanEngine::register_bda_table()
{
    bda_table.light_buffer = get_buffer_address(device, light_buffer.buffer);
    bda_table.light_cluster_buffer = get_buffer_address(device, light_cluster_buffer.buffer);
    bda_table.light_index_buffer = get_buffer_address(device, light_index_buffer.buffer);
    bda_table.light_grid_buffer = get_buffer_address(device, light_grid_buffer.buffer);
    bda_table.light_count_buffer = get_buffer_address(device, light_count_buffer.buffer);
    bda_table.vertex_buffer = get_buffer_address(device, vertex_buffer.buffer);
    bda_table.index_buffer = get_buffer_address(device, index_buffer.buffer);
    bda_table.indices_buffer = get_buffer_address(device, indices_buffer.buffer);
    bda_table.object_buffer = get_buffer_address(device, object_buffer.buffer);
    bda_table.mesh_buffer = get_buffer_address(device, mesh_buffer.buffer);
    bda_table.meshlet_buffer = get_buffer_address(device, meshlet_buffer.buffer);
    bda_table.meshlet_indices = get_buffer_address(device, meshlet_indices.buffer);
    bda_table.material_buffer = get_buffer_address(device, material_buffer.buffer);
    bda_table.draw_indirect_buffer = get_buffer_address(device, draw_indirect_buffer.buffer);
    bda_table.dispatch_buffer = get_buffer_address(device, dispatch_buffer.buffer);
    bda_table.vis_buffer = get_buffer_address(device, vis_buffer.buffer);
    bda_table.meshlet_vis_buffer = get_buffer_address(device, meshlet_vis_buffer.buffer);
    bda_table.meshlet_dispatch_buffer = get_buffer_address(device, meshlet_dispatch_buffer.buffer);
    bda_table.cluster_indices = get_buffer_address(device, cluster_indices.buffer);
    bda_table.oit_buffer = get_buffer_address(device, oit_buffer.buffer);
    bda_table.sh_buffer = get_buffer_address(device, sh_buffer.buffer);
    bda_table.luminance_buffer = get_buffer_address(device, luminance_buffer.buffer);
    bda_table.luminance_avg_buffer = get_buffer_address(device, luminance_avg_buffer.buffer);
    bda_table.prefix_sum_buffer = get_buffer_address(device, prefix_sum_buffer.buffer);
    bda_table.spd_counter_buffer = get_buffer_address(device, spd_counter_buffer.buffer);
}

int main(int argc, char** argv)
{
    VulkanEngine engine{};

    if (argc < 2)
        return 1;

    engine.init(argc, argv);
    engine.run();
    engine.cleanup();

    return 0;
}
