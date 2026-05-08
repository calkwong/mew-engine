#include "common.h"
#include "config.h"
#include "vk_math.h"
#include "vk_engine.h"
#include "cvars.h"
#include "inputs.h"
#include "vk_descriptors.h"
#include "resources.h"
#include "vk_initializers.h"
#include "vk_loader.h"
#include "vk_pipelines.h"
#include "vk_scene.h"
#include "cache.h"
#include "push_constants.h"
#include "rendergraph.h"

#include <filesystem>
#include <vk_mem_alloc.h>
#include <tracy/Tracy.hpp>
// #include <tracy/TracyVulkan.hpp>
#include "stb_image.h"

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_timer.h>
#include <SDL3/SDL_vulkan.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <span>
#include <thread>
#include <utility>
#include <cstdlib>

VulkanEngine* loaded_engine{};

VulkanEngine& VulkanEngine::get() { return *loaded_engine; }

#ifdef NDEBUG
constexpr bool USE_VALIDATION_LAYERS = false;
#else
constexpr bool USE_VALIDATION_LAYERS = true;
#endif

#define SINGLE // uncomment if loading a proper scene

AutoCVar_Int CVAR_RENDER_IMGUI{ "render.imgui", "Imgui", 1, CVarFlags::EditCheckbox | CVarFlags::EditHide };
AutoCVar_Int CVAR_DISABLE_CAMERA{ "render.disable_camera", "Disable camera", 0, CVarFlags::EditCheckbox | CVarFlags::EditHide };
AutoCVar_Int CVAR_HOT_RELOAD{ "render.hot_reload", "Hot reload shaders", 0, CVarFlags::EditCheckbox | CVarFlags::EditHide };

AutoCVar_Int CVAR_RENDER_VBUFFER{ "render.vbuffer", "Vbuffer path", 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_RENDER_MESH_SHADERS{ "render.mesh_shaders", "Mesh shaders path", 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_RENDER_ALPHACLIP{ "render.alphaclip", "Alphaclip", 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_RENDER_TRANSPARENT{ "render.transparent", "Transparent", 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_RENDER_POINT_LIGHTS{ "render.point_lights", "Point lights", 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_RENDER_OCCLUSION_CULL{ "render.occlusion_cull", "Occlusion culling", 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_RENDER_LOD{ "render.lod", "LODs", 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_RENDER_SHADOWS{ "render.shadows", "Shadows", 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_RENDER_SHADOWS_RT{ "render.shadows_rt", "Ray traced shadows", 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_RENDER_TAA{ "render.taa", "TAA", 0, CVarFlags::EditCheckbox }; // | CVarFlags::EditHide };
AutoCVar_Int CVAR_RENDER_RT{ "render.ray_tracing", "RT", 0, CVarFlags::EditCheckbox };

AutoCVar_Float CVAR_SHADOWS_CASCADE_SPLIT{ "shadows.cascade_split", "Cascades log factor", 0.95f, CVarFlags::EditDragFloat, 0.f, 1.f, 0.005f };
AutoCVar_Int CVAR_SHADOWS_DISTANCE{ "shadows.distance", "Shadow draw distance", 48, CVarFlags::EditSliderInt, 20, 200, 5 };

AutoCVar_Int CVAR_DEBUG_TEXTURES{ "debug.textures", "Debug textures", 0, CVarFlags::EditSliderInt, 0, DEBUG_COUNT, 1 };

AutoCVar_Int CVAR_MISC_DRAW_DISTANCE{ "misc.draw_distance", "Draw distance", 1000, CVarFlags::EditSliderInt, 100, 1000, 100 };
AutoCVar_Int CVAR_MISC_AUTOEXPOSURE{ "misc.autoexposure", "Autoexposure", 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_MISC_TONEMAP{ "misc.tonemap", "Tonemapping", 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_MISC_TONEMAP_FUNC{ "misc.tonemap_func", "Tonemapping function", 0, CVarFlags::EditSliderInt, 0, 3, 1 };
AutoCVar_Int CVAR_MISC_FREEZE_CAMERA{ "misc.freeze_camera", "Freeze camera", 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_MISC_HIZ_SPD{ "misc.hiz_spd", "HiZ SPD", 1, CVarFlags::EditCheckbox };

AutoCVar_Int CVAR_TAA_VARIANCE_CLIP{ "taa.variance_clip", "Variance clipping", 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_TAA_CATMULL_ROM{ "taa.catmull_rom", "Catmull filter", 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_TAA_MITCHELL{ "taa.mitchell", "Mitchell filter", 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_TAA_YCOCG{ "taa.ycogy", "YCoCg", 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_TAA_DYNAMIC{ "taa.dynamic", "Dynamic luma weights", 0, CVarFlags::EditCheckbox };

AutoCVar_Float CVAR_PBR_METALLIC{ "pbr.metallic", "Metallic", 0.0f, CVarFlags::EditDragFloat, 0.f, 1.f, 0.05f };
AutoCVar_Float CVAR_PBR_ROUGHNESS{ "pbr.roughness", "Roughness", 0.5f, CVarFlags::EditDragFloat, 0.f, 1.f, 0.05f };

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

void VulkanEngine::init(int argc, char** argv)
{
#ifdef NDEBUG
    fmt::println("Release mode");
#else
    fmt::println("Debug mode");
#endif

	assert(loaded_engine == nullptr);
	loaded_engine = this;

	VK_CHECK(volkInitialize());

	SDL_Init(SDL_INIT_VIDEO);

	auto window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

	window = SDL_CreateWindow(
	    "Vulkan Engine",
	    static_cast<int>(window_extent.width),
	    static_cast<int>(window_extent.height),
	    window_flags
	);

	// SDL_SetRelativeMouseMode(true);
	SDL_SetWindowRelativeMouseMode(window, true);

	init_vulkan();

	init_swapchain();

	init_commands();

	init_sync_structures();

	init_descriptors();

	init_shaders();

	init_pipelines();

	main_camera.position = glm::vec3(0, 0, 5);
	main_camera.far = static_cast<float>(CVAR_MISC_DRAW_DISTANCE.get());
	main_camera.near = 0.01f;
	main_camera.fov = 70.0f;
	// TODO: refactor if window resize
	main_camera.set_perspective_matrix(glm::radians(main_camera.fov), static_cast<float>(draw_extent.width) / static_cast<float>(draw_extent.height), main_camera.near);

	init_resources();

	init_renderables(argc, argv);

	upload_buffers();

	create_acceleration_structures();

	update_descriptors();

	init_imgui();

	build_cluster_grid(); // TODO: support draw distance change and rebuilding
	execute_baked_gi();

	// first_frame transitions to avoid validation errors
	{
		immediate_submit([&](VkCommandBuffer cmd)
		{
			// frame 0 history buffer
			stage_barrier(cmd, accumulation_buffers[(frame_number + 1) % 2].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, 0, 0, 0);

			vkCmdFillBuffer(cmd, render_scene.luminance_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
			// TODO: figure out a good default for luminance avg
			vkCmdFillBuffer(cmd, render_scene.luminance_avg_buffer.buffer, 0, VK_WHOLE_SIZE, 0x40000000); // 0x40000000 = 2.0f

			stage_barrier(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
		});
	}

	VkQueryPoolCreateInfo query_pool_info{};
	query_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
	query_pool_info.queryCount = QUERY_COUNT;
	for (auto& frame : frames)
	{
		VK_CHECK(vkCreateQueryPool(device, &query_pool_info, nullptr, &frame.query_pool_timestamps));
		vkResetQueryPool(device, frame.query_pool_timestamps, 0, QUERY_COUNT);
	}
	query_pool_info.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
	query_pool_info.queryCount = QUERY_COUNT;
	query_pool_info.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT;
	for (auto& frame : frames)
	{
		VK_CHECK(vkCreateQueryPool(device, &query_pool_info, nullptr, &frame.query_pool_pipelines));
		vkResetQueryPool(device, frame.query_pool_pipelines, 0, QUERY_COUNT);
	}

	// initialize jitter offsets
    for (int i = 0; i < jitter_offset.size(); i++)
    {
        float halton_x = 2.0f * Halton(i + 1, 2) - 1.0f;
        float halton_y = 2.0f * Halton(i + 1, 3) - 1.0f;
        float x = halton_x / static_cast<float>(draw_extent.width); // TODO: image resize
        float y = halton_y / static_cast<float>(draw_extent.height);
        jitter_offset[i] = glm::vec2(x, y);
    }

	is_initialized = true;
}

void VulkanEngine::cleanup()
{
	if (is_initialized)
	{

		vkDeviceWaitIdle(device);

		// TracyVkDestroy(tracy_ctx);

		loaded_scene.reset();

		for (const auto& info : sampler_cache.image_infos)
		{
			vkDestroySampler(device, info.sampler, nullptr);
		}

		for (auto& frame : frames)
		{
			vkDestroyCommandPool(device, frame.command_pool, nullptr);

			vkDestroyFence(device, frame.render_fence, nullptr);
			vkDestroySemaphore(device, frame.image_acquired_semaphore, nullptr);

			destroy_buffer(allocator, frame.scene_buffer);

			frame.deletion_queue.flush();

			vkDestroyQueryPool(device, frame.query_pool_timestamps, nullptr);
			vkDestroyQueryPool(device, frame.query_pool_pipelines, nullptr);
		}

		for (auto& sem : render_done_semaphores)
    	{
        	vkDestroySemaphore(device, sem, nullptr);
    	}

		for (const auto& [_, shader_program] : shader_cache.data)
		{
			vkDestroyShaderModule(device, shader_program.get()->module, nullptr);
		}

		destroy_buffer(allocator, render_scene.object_buffer);
		destroy_buffer(allocator, render_scene.mesh_buffer);
		destroy_buffer(allocator, render_scene.meshlet_buffer);
		destroy_buffer(allocator, render_scene.meshlet_indices);
		destroy_buffer(allocator, render_scene.material_buffer);
		destroy_buffer(allocator, render_scene.vertex_buffer);
		destroy_buffer(allocator, render_scene.index_buffer);

		destroy_buffer(allocator, render_scene.draw_indirect_buffer);
		destroy_buffer(allocator, render_scene.dispatch_buffer);
		destroy_buffer(allocator, render_scene.vis_buffer);
		destroy_buffer(allocator, render_scene.meshlet_vis_buffer);
		destroy_buffer(allocator, render_scene.meshlet_dispatch_buffer);
		destroy_buffer(allocator, render_scene.cluster_indices);

		destroy_buffer(allocator, render_scene.oit_buffer);
		destroy_buffer(allocator, render_scene.indices_buffer);

		destroy_buffer(allocator, render_scene.sh_buffer);
		destroy_buffer(allocator, render_scene.luminance_buffer);
		destroy_buffer(allocator, render_scene.luminance_avg_buffer);

		destroy_buffer(allocator, render_scene.prefix_sum_buffer);
		destroy_buffer(allocator, render_scene.spd_counter_buffer);

		for (const auto& [_, shader] : shader_passes)
		{
			vkDestroyPipeline(device, shader->pipeline, nullptr);
			vkDestroyPipelineLayout(device, shader->layout, nullptr);
		}

		main_deletion_queue.flush();

		destroy_swapchain();

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyDevice(device, nullptr);

		vkb::destroy_debug_utils_messenger(instance, debug_messenger);
		vkDestroyInstance(instance, nullptr);

		SDL_DestroyWindow(window);

		volkFinalize();
	}

	loaded_engine = nullptr;
}

void VulkanEngine::execute_baked_gi()
{
	VK_CHECK(vkResetFences(device, 1, &imm_fence));
	VK_CHECK(vkResetCommandPool(device, imm_command_pool, 0));
	VkCommandBufferBeginInfo cmd_begin_info = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	VK_CHECK(vkBeginCommandBuffer(imm_command_buffer, &cmd_begin_info));

	RenderGraph graph{};

	graph.add_pass("skybox", Pass::PassType::ComputePass,
        [&](Pass& pass) {
            pass.add_image_write("skybox", hdri_cubemap.image);
        },
        [&]() {
           	ShaderPass current_pass = *shader_passes["equirectangular_to_cubemap"];
    		IBLPushConstants pc{};
    		pc.image_size = glm::vec2(hdri_cubemap.extent.width, hdri_cubemap.extent.height);
    		pc.texture_id = texture_cache.get_hdri();
    		pc.image_id = image_cache.get_hdri();

    		vkCmdBindPipeline(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
    		vkCmdBindDescriptorSets(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
    		vkCmdBindDescriptorSets(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
    		vkCmdBindDescriptorSets(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
    		vkCmdBindDescriptorSets(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);
    		vkCmdPushConstants(imm_command_buffer, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(IBLPushConstants), &pc);
    		auto groupcount_x = get_groupcount(hdri_cubemap.extent.width, WARP_SIZE);
    		auto groupcount_y = get_groupcount(hdri_cubemap.extent.height, WARP_SIZE);
    		vkCmdDispatch(imm_command_buffer, groupcount_x, groupcount_y, 6);

    		// set to cubemap/skybo
    		auto updated_hdri_id = texture_cache.get_hdri() + 1;
    		texture_cache.set_hdri(updated_hdri_id);
        }
	);

	graph.add_pass("skybox_mipmap", Pass::PassType::ComputePass,
        [&](Pass& pass) {
            pass.add_image_read("skybox", hdri_cubemap.image);
            pass.add_image_write("skybox", hdri_cubemap.image);
        },
        [&]() {
            vkutil::generate_mipmaps(imm_command_buffer, hdri_cubemap.image, VkExtent2D(hdri_cubemap.extent.width, hdri_cubemap.extent.height), 6);
        }
	);

	graph.add_pass("spherical_harmonics", Pass::PassType::ComputePass,
        [&](Pass& pass) {
            pass.add_image_read("skybox", hdri_cubemap.image);
            pass.add_storage_buffer_write("sh");
        },
        [&]() {
            ShaderPass current_pass = *shader_passes["spherical_harmonics"];
    		SHPushConstants pc{};
    		pc.sh_buffer_address = get_buffer_address(device, render_scene.sh_buffer.buffer);
    		pc.cubemap_id = static_cast<uint32_t>(scene_data.textures[0]);

    		vkCmdBindPipeline(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
    		vkCmdBindDescriptorSets(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
    		vkCmdBindDescriptorSets(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
    		vkCmdBindDescriptorSets(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
    		vkCmdBindDescriptorSets(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);
    		vkCmdPushConstants(imm_command_buffer, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SHPushConstants), &pc);

    		vkCmdDispatch(imm_command_buffer, 1, 1, 1);
        }
	);

	graph.add_pass("irradiance", Pass::PassType::ComputePass,
        [&](Pass& pass) {
            pass.add_image_read("skybox", hdri_cubemap.image);
            pass.add_image_write("irradiance", irradiance_cubemap.image);
        },
        [&]() {
            ShaderPass current_pass = *shader_passes["irradiance"];
    		IBLPushConstants pc{};
    		pc.image_size = glm::vec2(irradiance_cubemap.extent.width, irradiance_cubemap.extent.height);
    		pc.texture_id = static_cast<uint32_t>(scene_data.textures[0]);
    		pc.image_id = image_cache.get_hdri() + 1;

    		vkCmdBindPipeline(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
    		vkCmdBindDescriptorSets(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
    		vkCmdBindDescriptorSets(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
    		vkCmdBindDescriptorSets(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
    		vkCmdBindDescriptorSets(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);
    		vkCmdPushConstants(imm_command_buffer, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(IBLPushConstants), &pc);
    		auto groupcount_x = get_groupcount(irradiance_cubemap.extent.width, WARP_SIZE);
    		auto groupcount_y = get_groupcount(irradiance_cubemap.extent.height, WARP_SIZE);
    		vkCmdDispatch(imm_command_buffer, groupcount_x, groupcount_y, 6);
        }
	);

	// TODO: fix this, potentially problematic
	uint32_t brdf_id{};

	graph.add_pass("prefiltered", Pass::PassType::ComputePass,
        [&](Pass& pass) {
            pass.add_image_read("skybox", hdri_cubemap.image);
            pass.add_image_write("prefiltered", prefiltered_envmap.image);
        },
        // TODO: fix - we are dispatching wg_size that is more than necessary here
        [&]() {
            ShaderPass current_pass = *shader_passes["prefiltered"];
    		vkCmdBindPipeline(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
    		vkCmdBindDescriptorSets(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
    		vkCmdBindDescriptorSets(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
    		vkCmdBindDescriptorSets(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
    		vkCmdBindDescriptorSets(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);
    		IBLPushConstants pc{};
    		pc.texture_id = static_cast<uint32_t>(scene_data.textures[0]);

    		auto mips = static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(std::max(prefiltered_envmap.extent.width, prefiltered_envmap.extent.height))))) + 1;
    		for (uint32_t i = 0; i < mips; i++)
    		{
    			pc.image_id = image_cache.get_hdri() + 2 + i;
    			pc.roughness = static_cast<float>(i) / static_cast<float>(mips);
    			vkCmdPushConstants(imm_command_buffer, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(IBLPushConstants), &pc);
    			auto groupcount_x = get_groupcount(prefiltered_envmap.extent.width, WARP_SIZE);
    			auto groupcount_y = get_groupcount(prefiltered_envmap.extent.height, WARP_SIZE);
    			vkCmdDispatch(imm_command_buffer, groupcount_x, groupcount_y, 6);
    		}
    		brdf_id = image_cache.get_hdri() + 2 + mips;
        }
	);

	graph.add_pass("brdf_lut", Pass::PassType::ComputePass,
        [&](Pass& pass) {
            pass.add_image_write("brdf_lut", brdf_lut.image);
        },
        [&]() {
            ShaderPass current_pass = *shader_passes["brdf"];
    		IBLPushConstants pc{};
    		pc.image_size = glm::vec2(brdf_lut.extent.width, brdf_lut.extent.height);
    		pc.image_id = brdf_id;

    		vkCmdBindPipeline(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
    		vkCmdBindDescriptorSets(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
    		vkCmdBindDescriptorSets(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
    		vkCmdBindDescriptorSets(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
    		vkCmdBindDescriptorSets(imm_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);
    		vkCmdPushConstants(imm_command_buffer, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(IBLPushConstants), &pc);
    		auto groupcount_x = get_groupcount(brdf_lut.extent.width, WARP_SIZE);
    		auto groupcount_y = get_groupcount(brdf_lut.extent.height, WARP_SIZE);
    		vkCmdDispatch(imm_command_buffer, groupcount_x, groupcount_y, 1);
        }
	);

	graph.bake();
	graph.execute(imm_command_buffer);

	VK_CHECK(vkEndCommandBuffer(imm_command_buffer));
	VkCommandBufferSubmitInfo cmd_info = vkinit::command_buffer_submit_info(imm_command_buffer);
	VkSubmitInfo2 submit = vkinit::submit_info(&cmd_info, nullptr, nullptr);
	VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit, imm_fence));
	VK_CHECK(vkWaitForFences(device, 1, &imm_fence, true, 9999999999));
}

void VulkanEngine::draw()
{
	{
		VK_CHECK(vkWaitForFences(device, 1, &get_current_frame().render_fence, true, 1000000000));
	}
	VK_CHECK(vkResetFences(device, 1, &get_current_frame().render_fence));

	get_current_frame().deletion_queue.flush();

	auto* scene_uniform_data = static_cast<SceneData*>(get_current_frame().scene_buffer.info.pMappedData);
	*scene_uniform_data = scene_data;

	CullData forward_mesh_cull_data{};
	ClusterCullData forward_cluster_cull_data{};

	{
		auto proj = freeze_camera ? last_proj : scene_data.proj;

		ready_mesh_cull(render_scene.opaque_pass, forward_mesh_cull_data, proj);
		ready_meshlet_cull(render_scene.opaque_pass, forward_cluster_cull_data, proj);
	}

	uint32_t swapchain_image_idx{};
	{
		VK_CHECK(vkAcquireNextImageKHR(device, swapchain, 1000000000, get_current_frame().image_acquired_semaphore, nullptr, &swapchain_image_idx));
	}

	// record currentFrame-2's timestamps
	{
		std::array<uint64_t, TIMESTAMP_QUERIES> timestamp_results{};
		std::array<uint64_t, PIPELINE_QUERIES> pipeline_results{};

		vkGetQueryPoolResults(
		    device,
		    get_current_frame().query_pool_timestamps,
		    0,
		    static_cast<uint32_t>(timestamp_results.size()),
		    timestamp_results.size() * sizeof(uint64_t),
		    timestamp_results.data(),
		    sizeof(uint64_t),
		    VK_QUERY_RESULT_64_BIT
		);

		vkGetQueryPoolResults(
		    device,
		    get_current_frame().query_pool_pipelines,
		    0,
		    static_cast<uint32_t>(pipeline_results.size()),
		    pipeline_results.size() * sizeof(uint64_t),
		    pipeline_results.data(),
		    sizeof(uint64_t),
		    VK_QUERY_RESULT_64_BIT
		);

		double timestamp_period = device_properties.limits.timestampPeriod;
		auto get_time = [&](size_t start, size_t end) -> double
		{
			return static_cast<double>(timestamp_results[end] - timestamp_results[start]) * timestamp_period * 1e-6;
		};

		stats.early_cull = get_time(0, 1);
		stats.early_indirect = get_time(2, 3);
		stats.late_cull = get_time(4, 5);
		stats.late_indirect = get_time(6, 7);
		stats.mask_cull = get_time(8, 9);
		stats.mask_indirect = get_time(10, 11);
		stats.light_culling = get_time(12, 13);
		stats.deferred_shading = get_time(14, 15);
		stats.transparent_cull = get_time(16, 17);
		stats.transparent_render = get_time(18, 19);
		stats.shadow_cull = get_time(20, 21);
		stats.shadow_render = get_time(22, 23);
		stats.taa_resolve = get_time(24, 25);
		auto gpu_time = get_time(26, 27);
		stats.gpu_time = gpu_time + 0.95 * (stats.gpu_time - gpu_time);
		stats.hiz = get_time(28, 29);

		stats.triangle_count = 0;
		// for (size_t i = 0; i < pipeline_results.size() - 1; i++)
		for (size_t i = 0; i < 4 - 1; i++)
		{
			stats.triangle_count += static_cast<unsigned int>(pipeline_results[i]);
		}
		stats.cascade0 = static_cast<unsigned int>(pipeline_results[4]);
		stats.cascade1 = static_cast<unsigned int>(pipeline_results[5]);
		stats.cascade2 = static_cast<unsigned int>(pipeline_results[6]);
		stats.cascade3 = static_cast<unsigned int>(pipeline_results[7]);
	}

	auto& frame_query_pool_timestamps = get_current_frame().query_pool_timestamps;
	auto& frame_query_pool_pipelines = get_current_frame().query_pool_pipelines;

	vkResetQueryPool(device, frame_query_pool_timestamps, 0, QUERY_COUNT);
	vkResetQueryPool(device, frame_query_pool_pipelines, 0, QUERY_COUNT);

	VkCommandBuffer cmd = get_current_frame().main_command_buffer;

	VK_CHECK(vkResetCommandPool(device, get_current_frame().command_pool, 0));

	VkCommandBufferBeginInfo cmd_begin_info = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));

	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 26);

	auto zero_buffers = [&]() {
    	vkCmdFillBuffer(cmd, render_scene.dispatch_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
    	vkCmdFillBuffer(cmd, render_scene.meshlet_dispatch_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
    	vkCmdFillBuffer(cmd, render_scene.draw_indirect_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
    	vkCmdFillBuffer(cmd, render_scene.prefix_sum_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
	};

	auto two_pass_occlusion_culling = [&](RenderGraph& graph, const std::string& prefix, RenderScene::MeshPass& mesh_pass, uint32_t offset, bool late, uint32_t post_pass, uint32_t query, uint32_t timestamp, bool clear = false) {
        graph.add_pass(prefix+"zero_buffers", Pass::PassType::ComputePass,
               [&](Pass& pass) {
                   pass.add_storage_buffer_write("dispatch");
                   pass.add_storage_buffer_write("meshlet_dispatch");
                   pass.add_storage_buffer_write("draw_indirect");
                   pass.add_storage_buffer_write("prefix_sum");
               },
               [&]() {
                   zero_buffers();
               }
        );

        graph.add_pass(prefix+"cull_meshes", Pass::PassType::ComputePass,
               [&](Pass& pass) {
                   pass.add_storage_buffer_write("object");
                   pass.add_storage_buffer_write("mesh");
                   pass.add_storage_buffer_write("indices");
                   pass.add_storage_buffer_write("draw_indirect");
                   pass.add_storage_buffer_write("dispatch");
                   pass.add_storage_buffer_write("vis");
                   pass.add_storage_buffer_write("prefix_sum");
                   if (late)
                   {
                       pass.add_image_read("depth", depth_image.image);
                       pass.add_image_read("hiz", depth_pyramid.image);
                   }
               },
               [&, late, post_pass, timestamp]() {
                   vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, timestamp + 0);
                   execute_compute_cull(cmd, mesh_pass, forward_mesh_cull_data, late, post_pass);
                   if (!CVAR_RENDER_MESH_SHADERS.get())
                       vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, timestamp + 1);
               }
        );

        if (CVAR_RENDER_MESH_SHADERS.get())
        {
            graph.add_pass(prefix+"compact_dispatch", Pass::PassType::ComputePass,
                   [&](Pass& pass) {
                       pass.add_storage_buffer_read("dispatch");
                       pass.add_storage_buffer_write("dispatch");
                       pass.add_storage_buffer_write("prefix_sum");
                   },
                   [&]() {
                       execute_compact_dispatch(cmd);
                   }
            );

            graph.add_pass(prefix+"cull_meshlets", Pass::PassType::ComputePass,
                   [&](Pass& pass) {
                       pass.add_storage_buffer_write("object");
                       pass.add_storage_buffer_write("meshlet");
                       pass.add_storage_buffer_write("cluster_indices");
                       pass.add_storage_buffer_write("meshlet_dispatch");
                       pass.add_storage_buffer_write("cluster_vis");
                       pass.add_storage_buffer_write("prefix_sum");
                       if (late)
                       {
                           pass.add_image_read("depth", depth_image.image);
                           pass.add_image_read("hiz", depth_pyramid.image);
                       }
                   },
                   [&, mesh_pass, offset, late, post_pass, timestamp]() {
                       execute_compute_cull(cmd, forward_cluster_cull_data, render_scene.dispatch_buffer.buffer, offset, late, post_pass);
                       vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, timestamp + 1);
                   }
            );
        }

        graph.add_pass(prefix+"rasterization", Pass::PassType::GraphicsPass,
            [&, clear](Pass& pass) {
                pass.add_storage_buffer_write("object");
                pass.add_storage_buffer_write("meshlet");
                pass.add_storage_buffer_write("meshlet_indices");
                pass.add_storage_buffer_write("cluster_indices");
                pass.add_storage_buffer_write("material");
                pass.add_storage_buffer_write("prefix_sum");
                pass.add_depth_stencil_output("depth", depth_image.image);
                bool visibility_rendering = CVAR_RENDER_VBUFFER.get() && CVAR_RENDER_MESH_SHADERS.get();
                if (visibility_rendering)
                {
                    pass.add_color_output("vis_buffer", visibility_buffer.image);
                    pass.add_color_output("velocity", velocity_buffer.image);
                    if (!clear)
                    {
                        pass.add_image_read("vis_buffer", visibility_buffer.image);
                        pass.add_image_read("velocity", velocity_buffer.image);
                    }
                }
                else
                {
                    pass.add_color_output("gbuffer0", gbuffers[0].image);
                    pass.add_color_output("gbuffer1", gbuffers[1].image);
                    pass.add_color_output("gbuffer2", gbuffers[2].image);
                    pass.add_color_output("gbuffer3", gbuffers[3].image);
                    if (!clear)
                    {
                        pass.add_image_read("gbuffer0", gbuffers[0].image);
                        pass.add_image_read("gbuffer1", gbuffers[1].image);
                        pass.add_image_read("gbuffer2", gbuffers[2].image);
                        pass.add_image_read("gbuffer3", gbuffers[3].image);
                    }
                }
            },
            [&, late, post_pass, query, timestamp]() {
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, timestamp + 2);
                render(cmd, late, post_pass, query);
               	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, timestamp + 3);
            }
        );
	};

	auto transparent_pass = [&](RenderGraph& graph, const std::string& prefix, uint32_t offset, bool late, uint32_t post_pass, uint32_t query, uint32_t timestamp) {
  		graph.add_pass(prefix+"zero_buffers", Pass::PassType::ComputePass,
            [&](Pass& pass) {
                pass.add_storage_buffer_write("dispatch");
                pass.add_storage_buffer_write("meshlet_dispatch");
                pass.add_storage_buffer_write("draw_indirect");
                pass.add_storage_buffer_write("prefix_sum");
            },
            [&]() {
                zero_buffers();
            }
        );

        graph.add_pass(prefix+"cull_meshes", Pass::PassType::ComputePass,
            [&](Pass& pass) {
                pass.add_storage_buffer_write("object");
                pass.add_storage_buffer_write("mesh");
                pass.add_storage_buffer_write("indices");
                pass.add_storage_buffer_write("draw_indirect");
                pass.add_storage_buffer_write("dispatch");
                pass.add_storage_buffer_write("vis");
                pass.add_storage_buffer_write("prefix_sum");
                if (late)
                {
                    pass.add_image_read("depth", depth_image.image);
                    pass.add_image_read("hiz", depth_pyramid.image);
                }
            },
            [&, late, post_pass, timestamp]() {
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, timestamp + 0);
                execute_compute_cull(cmd, render_scene.transparent_pass, forward_mesh_cull_data, late, post_pass);
                if (!CVAR_RENDER_MESH_SHADERS.get())
                    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, timestamp + 1);
            }
        );

        if (CVAR_RENDER_MESH_SHADERS.get())
        {
            graph.add_pass(prefix+"compact_dispatch", Pass::PassType::ComputePass,
                [&](Pass& pass) {
                    pass.add_storage_buffer_read("dispatch");
                    pass.add_storage_buffer_write("dispatch");
                    pass.add_storage_buffer_write("prefix_sum");
                },
                [&]() {
                    execute_compact_dispatch(cmd);
                }
            );

            graph.add_pass(prefix+"cull_meshlets", Pass::PassType::ComputePass,
                [&](Pass& pass) {
                    pass.add_storage_buffer_write("object");
                    pass.add_storage_buffer_write("meshlet");
                    pass.add_storage_buffer_write("cluster_indices");
                    pass.add_storage_buffer_write("meshlet_dispatch");
                    pass.add_storage_buffer_write("cluster_vis");
                    pass.add_storage_buffer_write("prefix_sum");
                    if (late)
                    {
                        pass.add_image_read("depth", depth_image.image);
                        pass.add_image_read("hiz", depth_pyramid.image);
                    }
                },
                [&, offset, late, post_pass, timestamp]() {
                    execute_compute_cull(cmd, forward_cluster_cull_data, render_scene.dispatch_buffer.buffer, offset, late, post_pass);
                    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, timestamp + 1);
                }
            );
        }

        graph.add_pass("transparent_forward", Pass::PassType::GraphicsPass,
            [&](Pass& pass) {
                pass.add_storage_buffer_write("oit");
                pass.add_storage_buffer_write("object");
                pass.add_storage_buffer_write("meshlet");
                pass.add_storage_buffer_write("meshlet_indices");
                pass.add_storage_buffer_write("cluster_indices");
                pass.add_storage_buffer_write("material");
                pass.add_storage_buffer_write("prefix_sum");
                pass.add_image_read("depth", depth_image.image);
            },
            [&, query, timestamp]() {
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, timestamp + 2);
                render_transparent(cmd, query);
               	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, timestamp + 3);
            }
        );
    };

    RenderGraph graph{};
	{
		two_pass_occlusion_culling(graph, "opaque_early_", render_scene.opaque_pass, 0, false, 0, 0, 0, true);

		if (!freeze_camera)
		{
            graph.add_pass("hiz", Pass::PassType::ComputePass,
                [&](Pass& pass) {
                    if (CVAR_MISC_HIZ_SPD.get())
                    {
                        pass.add_storage_buffer_write("spd_counter");
                        pass.add_storage_buffer_read("spd_counter");
                    }
                    pass.add_image_read("depth", depth_image.image);
                    pass.add_image_write("hiz", depth_pyramid.image);
                },
                [&]() {
                    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 28);
                    if (CVAR_MISC_HIZ_SPD.get())
                        execute_spd(cmd);
                    else
                        build_depth_pyramid(cmd);
                    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 29);
                }
            );
		}
		else
		{
    		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 28);
      		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 29);
		}

		two_pass_occlusion_culling(graph, "opaque_late_", render_scene.opaque_pass, 0, true, 0, 1, 4);

		// alphaclip postpass only, this is using early pass hiz for culling
		if (CVAR_RENDER_ALPHACLIP.get())
		    two_pass_occlusion_culling(graph, "alphaclip_late_", render_scene.mask_pass, 0, true, 1, 2, 8);
		else
		{
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 8);
    		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 9);
    		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 10);
    		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 11);
    		vkCmdBeginQuery(cmd, get_current_frame().query_pool_pipelines, 2, 0);
    		vkCmdEndQuery(cmd, get_current_frame().query_pool_pipelines, 2);
		}

		if (CVAR_RENDER_TRANSPARENT.get())
		    transparent_pass(graph, "transparent_late_", 0, true, 2, 3, 16);
		else
		{
    		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 16);
    		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 17);
    		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 18);
    		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 19);
    		vkCmdBeginQuery(cmd, get_current_frame().query_pool_pipelines, 3, 0);
    		vkCmdEndQuery(cmd, get_current_frame().query_pool_pipelines, 3);
		};

		if (CVAR_RENDER_POINT_LIGHTS.get())
		{
    		// TODO: combine this somewhere
    		// TODO: handle as Transfer instead of setting to Compute?
    		graph.add_pass("zero light buffers", Pass::PassType::ComputePass,
                [&](Pass& pass) {
                    pass.add_storage_buffer_write("light_count");
                },
                [&]() {
                    vkCmdFillBuffer(cmd, light_count_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
                }
            );

    		graph.add_pass("light_culling", Pass::PassType::ComputePass,
                [&](Pass& pass) {
                    pass.add_storage_buffer_read("light_cluster");
                    pass.add_storage_buffer_read("light");
                    pass.add_storage_buffer_write("light_index");
                    pass.add_storage_buffer_write("light_grid");
                    pass.add_storage_buffer_write("light_count");
                },
                [&]() {
                    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 12);
              		execute_light_culling(cmd);
              		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 13);
                }
            );
		}
		else
		{
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 12);
		    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 13);
		}

		if (CVAR_RENDER_SHADOWS.get() && !CVAR_RENDER_SHADOWS_RT.get())
		{
            graph.add_pass("zero shadow buffers", Pass::PassType::ComputePass,
                [&](Pass& pass) {
                    pass.add_storage_buffer_write("draw_indirect");
                    pass.add_storage_buffer_write("dispatch");
                },
                [&]() {
                    vkCmdFillBuffer(cmd, render_scene.dispatch_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
                    vkCmdFillBuffer(cmd, render_scene.draw_indirect_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
                }
            );

            graph.add_pass("cull shadow casters", Pass::PassType::ComputePass,
                [&](Pass& pass) {
                    pass.add_storage_buffer_read("object");
                    pass.add_storage_buffer_read("mesh");
                    pass.add_storage_buffer_write("indices");
                    pass.add_storage_buffer_write("draw_indirect");
                },
                [&]() {
                    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 20);
                    execute_shadow_cull(cmd);
		            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 21);
                }
            );

            graph.add_pass("render shadows", Pass::PassType::GraphicsPass,
                [&](Pass& pass) {
                    pass.add_storage_buffer_write("material");
                    pass.add_storage_buffer_write("object");
                    for (size_t i = 0; i < cascade_data.size(); i++)
                        pass.add_depth_stencil_output("shadowmap_"+std::to_string(i), cascade_data[i].shadow_map.image);
                },
                [&]() {
                    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 22);
              		uint32_t query_index = 4;
              		for (size_t i = 0; i < cascade_data.size(); i++, query_index++)
                        render_shadows(cmd, static_cast<uint32_t>(i), query_index);
              		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 23);
                }
            );
		}
		else
		{
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 20);
    		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 21);
    		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 22);
    		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 23);
    		vkCmdBeginQuery(cmd, get_current_frame().query_pool_pipelines, 4, 0);
    		vkCmdEndQuery(cmd, get_current_frame().query_pool_pipelines, 4);
		}

		graph.add_pass("lighting pass", Pass::PassType::ComputePass,
            [&](Pass& pass) {
                pass.add_storage_buffer_read("light");
                pass.add_storage_buffer_read("light_index");
                pass.add_storage_buffer_read("light_grid");
                pass.add_storage_buffer_read("oit");
                pass.add_storage_buffer_write("oit");
                pass.add_storage_buffer_read("meshlet_indices");
                pass.add_storage_buffer_read("meshlet");
                pass.add_storage_buffer_read("object");
                pass.add_storage_buffer_read("material");
                pass.add_storage_buffer_read("mesh");
                pass.add_storage_buffer_read("sh");
                // TODO: technically not a color attachment here
                pass.add_color_output("draw", draw_image.image);
            },
            [&]() {
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 14);
          		execute_shading(cmd);
          		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 15);
            }
        );

        // TODO: fix autoexposure

        if (CVAR_RENDER_TAA.get())
        {
            graph.add_pass("taa", Pass::PassType::ComputePass,
                [&](Pass& pass) {
                    pass.add_image_read("velocity", velocity_buffer.image);
                    pass.add_image_read("draw", draw_image.image);
                    pass.add_image_read("taa_history", accumulation_buffers[(frame_number + 1) % 2].image);
                    pass.add_image_write("taa_resolve", accumulation_buffers[frame_number % 2].image);
                },
                [&]() {
                    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 24);
              		resolve_taa(cmd);
              		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 25);
                }
            );
        }
        else
        {
           	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 24);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 25);
        }

        if (CVAR_MISC_TONEMAP.get() && CVAR_DEBUG_TEXTURES.get() == 0)
        {
            graph.add_pass("tonemapping", Pass::PassType::ComputePass,
                [&](Pass& pass) {
                    pass.add_storage_buffer_read("luminance_avg");
                    pass.add_storage_buffer_read("draw");
                    pass.add_storage_buffer_write("draw");
                },
                [&]() {
                    ShaderPass current_pass = *shader_passes["tonemap"];
              		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
              		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
              		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
              		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
              		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);

              		TonemapPushConstants pc{};
              		pc.luminance_avg_buffer = get_buffer_address(device, render_scene.luminance_avg_buffer.buffer);
              		pc.screen_size = glm::vec2(draw_image.extent.width, draw_image.extent.height);
              		pc.src_id = CVAR_RENDER_TAA.get() ? image_cache.get_accumulation_buffer(frame_number % 2) : image_cache.get_draw_image();
              		pc.dst_id = image_cache.get_draw_image();
              		// pc.autoexposure = CVAR_MISC_AUTOEXPOSURE.get();
              		pc.tonemap_func = CVAR_MISC_TONEMAP_FUNC.get();

              		vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TonemapPushConstants), &pc);
              		auto groupcount_x = get_groupcount(draw_image.extent.width, WARP_SIZE);
              		auto groupcount_y = get_groupcount(draw_image.extent.height, WARP_SIZE);
              		vkCmdDispatch(cmd, groupcount_x, groupcount_y, 1);
                }
            );

        }

        // graph.print();
    	graph.bake();
    	graph.execute(cmd);
	}

	// TODO: move this into rendergraph
	stage_barrier(cmd, swapchain_images[swapchain_image_idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT
	);

	vkutil::copy_image(cmd, draw_image.image, swapchain_images[swapchain_image_idx], draw_extent, swapchain_extent);

	stage_barrier(cmd, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);

	{
		if (CVAR_RENDER_IMGUI.get())
			draw_imgui(cmd, swapchain_image_views[swapchain_image_idx]);
	}

	stage_barrier(
	    cmd,
	    swapchain_images[swapchain_image_idx],
	    VK_IMAGE_LAYOUT_GENERAL,
	    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
	    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
	    VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
	    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
	    0
	);

	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 27);

	// TracyVkCollect(tracy_ctx, get_current_frame().main_command_buffer);
	VK_CHECK(vkEndCommandBuffer(cmd));

	VkCommandBufferSubmitInfo cmd_info = vkinit::command_buffer_submit_info(cmd);

	VkSemaphoreSubmitInfo wait_info = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, get_current_frame().image_acquired_semaphore);
	VkSemaphoreSubmitInfo submit_info = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, render_done_semaphores[swapchain_image_idx]); // all graphics bit?

	VkSubmitInfo2 submit = vkinit::submit_info(&cmd_info, &submit_info, &wait_info);

	{
		VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit, get_current_frame().render_fence));
	}

	VkPresentInfoKHR present_info{};
	present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	present_info.waitSemaphoreCount = 1;
	present_info.pWaitSemaphores = &render_done_semaphores[swapchain_image_idx];
	present_info.swapchainCount = 1;
	present_info.pSwapchains = &swapchain;
	present_info.pImageIndices = &swapchain_image_idx;

	VK_CHECK(vkQueuePresentKHR(graphics_queue, &present_info));
	// FrameMark;
	frame_number++;
}

void VulkanEngine::run()
{
	SDL_Event e;
	bool b_quit = false;

	auto last_frame = std::chrono::system_clock::now();

	while (!b_quit)
	{
		auto start = std::chrono::system_clock::now();
		auto deltatime = std::chrono::duration_cast<std::chrono::microseconds>(start - last_frame);
		stats.deltatime = static_cast<float>(deltatime.count()) / 1000000.0f; // microseconds to seconds
		last_frame = start;

		while (SDL_PollEvent(&e) != 0)
		{
			if (e.type == SDL_EVENT_QUIT)
				b_quit = true;

			if (e.type == SDL_EVENT_WINDOW_MINIMIZED)
				stop_rendering = true;
			if (e.type == SDL_EVENT_WINDOW_RESTORED)
				stop_rendering = false;

			key_callback(window, e);

			if (SDL_GetWindowRelativeMouseMode(window))
				main_camera.process_sdl_event(e);

			if (get_int_cvars("render.hot_reload") == 1)
			{
			    set_int_cvars("render.hot_reload", 0);

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
							vkutil::load_shader_module(shader_path.c_str(), device, &program->module);
							rebuild = true;
						}
					}

					if (rebuild)
					{
						VK_CHECK(vkDeviceWaitIdle(device));

						for (const auto& [_, shader] : shader_passes)
						{
							vkDestroyPipeline(device, shader->pipeline, nullptr);
							vkDestroyPipelineLayout(device, shader->layout, nullptr);
						}

						// TODO: instead of rebuilding everything, we could just update relevant pipelines, but full rebuild is almost instantaneous so we roll with this for now
						shader_passes.clear();
						init_pipelines();
					}
				}
			}

			ImGui_ImplSDL3_ProcessEvent(&e);
		}

		if (stop_rendering)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			continue;
		}

		freeze_camera = CVAR_MISC_FREEZE_CAMERA.get();

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplSDL3_NewFrame();
		ImGui::NewFrame();

		// ImGui::ShowDemoWindow();
		CVarSystem::get()->draw_imgui_editor();

		{
			ImGui::Begin("Stats");
			ImGui::Text("Total render time:    %.3f ms", stats.cpu_time);
			ImGui::Text("Gpu render time:      %.3f ms", stats.gpu_time);
			// ImGui::Text("scene update time %f ms", stats.scene_update_time);
			ImGui::Text("Early cull:           %.3f ms", stats.early_cull);
			ImGui::Text("Late  cull:           %.3f ms", stats.late_cull);
			ImGui::Text("Early render:         %.3f ms", stats.early_indirect);
			ImGui::Text("Late render:          %.3f ms", stats.late_indirect);
			ImGui::Text("Light culling:        %.3f ms", stats.light_culling);
			ImGui::Text("Deferred shading:     %.3f ms", stats.deferred_shading);
			ImGui::Text("Build hiz:            %.3f ms", stats.hiz);
			ImGui::Text("Mask cull:            %.3f ms", stats.mask_cull);
			ImGui::Text("Mask render:          %.3f ms", stats.mask_indirect);
			ImGui::Text("Transparent cull:     %.3f ms", stats.transparent_cull);
			ImGui::Text("Transparent render:   %.3f ms", stats.transparent_render);
			ImGui::Text("Shadow cull:          %.3f ms", stats.shadow_cull);
			ImGui::Text("Shadow render:        %.3f ms", stats.shadow_render);
			ImGui::Text("TAA resolve:          %.3f ms", stats.taa_resolve);
			ImGui::Text("Triangles:            %u", stats.triangle_count);
			ImGui::Text("Cascade 0:            %u", stats.cascade0);
			ImGui::Text("Cascade 1:            %u", stats.cascade1);
			ImGui::Text("Cascade 2:            %u", stats.cascade2);
			ImGui::Text("Cascade 3:            %u", stats.cascade3);
			ImGui::Text("Clipping invocations: %.1fM", static_cast<double>(stats.triangle_count) * 1e-6);

			ImGui::End();
		}

		ImGui::Render();

		update_scene();

		draw();

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f;
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

	// vulkan 1.1 features
	VkPhysicalDeviceVulkan11Features features11{};
	features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
	features11.storageBuffer16BitAccess = true;
	features11.storagePushConstant16 = true; // note: possible slang capability bug, setting to true so val layer doesn't complain

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

	VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT fragment_shader_interlock_features{};
	fragment_shader_interlock_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT;
	fragment_shader_interlock_features.fragmentShaderPixelInterlock = true;

	VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features{};
	ray_query_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
	ray_query_features.rayQuery = true;

	VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_features{};
	acceleration_structure_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
	acceleration_structure_features.accelerationStructure = true;

	// use vkbootstrap to select a gpu.
	// we want a gpu that can write to the SDL surface and supports vulkan 1.3 with the correct features
	vkb::PhysicalDeviceSelector selector{ vkb_inst };
	vkb::PhysicalDevice physicalDevice = selector
	                                         .set_minimum_version(1, 4)
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
											 .add_required_extension_features(mesh_shader_features)
	                                         .add_required_extension_features(fragment_shader_interlock_features)
										     .add_required_extension_features(ray_query_features)
											 .add_required_extension_features(acceleration_structure_features)
	                                         .set_surface(surface)
	                                         .select()
	                                         .value();

	// create the final vulkan device
	vkb::DeviceBuilder deviceBuilder{ physicalDevice };

	vkb::Device vkbDevice = deviceBuilder.build().value();

	// get the VkDevice handle used in the rest of a vulkan application
	device = vkbDevice.device;
	chosen_gpu = physicalDevice.physical_device;

	volkLoadDevice(device);

	// use vkbootstrap to get a Graphics queue
	graphics_queue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
	graphics_queue_family = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

	VmaAllocatorCreateInfo allocator_info{};
	allocator_info.physicalDevice = chosen_gpu;
	allocator_info.device = device;
	allocator_info.instance = instance;
	allocator_info.vulkanApiVersion = VK_API_VERSION_1_4;
	allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT; // allows usage of GPU pointers

	VmaVulkanFunctions vulkan_functions{};
	VK_CHECK(vmaImportVulkanFunctionsFromVolk(&allocator_info, &vulkan_functions));
	allocator_info.pVulkanFunctions = &vulkan_functions;
	vmaCreateAllocator(&allocator_info, &allocator);

	main_deletion_queue.push_function([&]()
	{
		vmaDestroyAllocator(allocator);
	});

	vkGetPhysicalDeviceProperties(chosen_gpu, &device_properties);
	assert(device_properties.limits.timestampComputeAndGraphics);

	uint32_t count = 0;
	vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr);
	std::vector<VkExtensionProperties> extensions(count);
	vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, extensions.data());
	std::vector<const char*> extension_names = {
	    VK_KHR_RAY_QUERY_EXTENSION_NAME,
		VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
		VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
		VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME,
		VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME,
	};

	// check for extension support
	for (uint32_t i = 0; i < count; i++)
	{
		for (const auto* extension : extension_names)
    	{
           	if (strcmp(extension, extensions[i].extensionName) == 0)
                fmt::println("{} supported", extension);
    	}
	}
}

void VulkanEngine::init_swapchain()
{
	create_swapchain(window_extent.width, window_extent.height);

	VkExtent3D draw_image_extent{ window_extent.width, window_extent.height, 1 };

	// TODO: after deferred - transfer_src & general only?
	VkImageUsageFlags draw_image_flags{
		VK_IMAGE_USAGE_TRANSFER_SRC_BIT | // copy to swapchain
		VK_IMAGE_USAGE_SAMPLED_BIT | // for post FX sampling
		VK_IMAGE_USAGE_STORAGE_BIT // write in compute
	};

	// if reverting to VK_FORMAT_R16G16B16A16_SFLOAT, need to preexpose lights
	draw_image = create_image(device, allocator, draw_image_extent, VK_FORMAT_R32G32B32A32_SFLOAT, draw_image_flags, VK_IMAGE_ASPECT_COLOR_BIT);

	auto id = texture_cache.add_texture(draw_image.view);
	assert(id == 0); // hardcode to id 0
	texture_cache.set_draw_image(id);

	id = image_cache.add_texture(draw_image.view);
	assert(id == 0); // hardcode to id 0
	image_cache.set_draw_image(id);

	// TODO: refactor prob necessary after implementing window/swapchain resize
	draw_extent.width = draw_image.extent.width;
	draw_extent.height = draw_image.extent.height;

	VkImageUsageFlags gbuffer_flags{
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
		VK_IMAGE_USAGE_SAMPLED_BIT
	};

	// visibility path - visibility, velocity
	{
		visibility_buffer = create_image(device, allocator, draw_image_extent, VK_FORMAT_R32G32_UINT, gbuffer_flags, VK_IMAGE_ASPECT_COLOR_BIT);
		auto vis_id = texture_cache.add_texture(visibility_buffer.view);
		texture_cache.set_visibility_buffer(vis_id);
		velocity_buffer = create_image(device, allocator, draw_image_extent, VK_FORMAT_R16G16_SFLOAT, gbuffer_flags, VK_IMAGE_ASPECT_COLOR_BIT);
		texture_cache.add_texture(velocity_buffer.view);

		for (int i = 0; i < 2; ++i) // ping pong
		{
			accumulation_buffers[i] = create_image(device, allocator, draw_image_extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
			auto texture_accum_id = texture_cache.add_texture(accumulation_buffers[i].view);
			auto image_accum_id = image_cache.add_texture(accumulation_buffers[i].view);

			if (i == 0)
			{
				texture_cache.set_accumulation_buffer(texture_accum_id);
				image_cache.set_accumulation_buffer(image_accum_id);
			}
		}
	}

	// deferred path - albedo, normal, metalroughness, velocity
	{
		gbuffers.emplace_back(create_image(device, allocator, draw_image_extent, VK_FORMAT_R8G8B8A8_UNORM, gbuffer_flags, VK_IMAGE_ASPECT_COLOR_BIT));
		auto gbuffer_id = texture_cache.add_texture(gbuffers[0].view);
		texture_cache.set_gbuffers(gbuffer_id);
		gbuffers.emplace_back(create_image(device, allocator, draw_image_extent, VK_FORMAT_R16G16B16A16_SFLOAT, gbuffer_flags, VK_IMAGE_ASPECT_COLOR_BIT));
		texture_cache.add_texture(gbuffers[1].view);
		gbuffers.emplace_back(create_image(device, allocator, draw_image_extent, VK_FORMAT_R8G8_SNORM, gbuffer_flags, VK_IMAGE_ASPECT_COLOR_BIT));
		texture_cache.add_texture(gbuffers[2].view);
		gbuffers.emplace_back(create_image(device, allocator, draw_image_extent, VK_FORMAT_R16G16_SFLOAT, gbuffer_flags, VK_IMAGE_ASPECT_COLOR_BIT));
		texture_cache.add_texture(gbuffers[3].view);
	}

	depth_image.format = VK_FORMAT_D32_SFLOAT;

	depth_image = create_image(device, allocator, draw_image_extent, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);

	id = texture_cache.add_texture(depth_image.view);
	texture_cache.set_depth_image(id);

	main_deletion_queue.push_function([&]()
	{
		vkDestroyImageView(device, draw_image.view, nullptr);
		vmaDestroyImage(allocator, draw_image.image, draw_image.allocation);
		vkDestroyImageView(device, visibility_buffer.view, nullptr);
		vmaDestroyImage(allocator, visibility_buffer.image, visibility_buffer.allocation);
		vkDestroyImageView(device, velocity_buffer.view, nullptr);
		vmaDestroyImage(allocator, velocity_buffer.image, velocity_buffer.allocation);
		vkDestroyImageView(device, accumulation_buffers[0].view, nullptr);
		vmaDestroyImage(allocator, accumulation_buffers[0].image, accumulation_buffers[0].allocation);
		vkDestroyImageView(device, accumulation_buffers[1].view, nullptr);
		vmaDestroyImage(allocator, accumulation_buffers[1].image, accumulation_buffers[1].allocation);
		vkDestroyImageView(device, depth_image.view, nullptr);
		vmaDestroyImage(allocator, depth_image.image, depth_image.allocation);

		for (int i = 0; i < GBUFFER_COUNT; i++)
		{
			vkDestroyImageView(device, gbuffers[i].view, nullptr);
			vmaDestroyImage(allocator, gbuffers[i].image, gbuffers[i].allocation);
		}
	});
}

void VulkanEngine::init_commands()
{
	VkCommandPoolCreateInfo command_pool_info = vkinit::command_pool_create_info(
	    graphics_queue_family
	);

	for (auto& frame : frames)
	{
		VK_CHECK(vkCreateCommandPool(device, &command_pool_info, nullptr, &frame.command_pool));

		VkCommandBufferAllocateInfo cmd_alloc_info = vkinit::command_buffer_allocate_info(
		    frame.command_pool
		);

		VK_CHECK(vkAllocateCommandBuffers(device, &cmd_alloc_info, &frame.main_command_buffer));
	}

	// tracy_ctx = TracyVkContextCalibrated(chosen_gpu, device, graphics_queue, frames[0].main_command_buffer, vkGetPhysicalDeviceCalibrateableTimeDomainsKHR, vkGetCalibratedTimestampsKHR);

	VK_CHECK(vkCreateCommandPool(device, &command_pool_info, nullptr, &imm_command_pool));
	VkCommandBufferAllocateInfo cmd_alloc_info = vkinit::command_buffer_allocate_info(
	    imm_command_pool
	);
	VK_CHECK(vkAllocateCommandBuffers(device, &cmd_alloc_info, &imm_command_buffer));

	main_deletion_queue.push_function([&]()
	{
		vkDestroyCommandPool(device, imm_command_pool, nullptr);
	});
}

void VulkanEngine::init_sync_structures()
{
	VkFenceCreateInfo fence_info = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
	VkSemaphoreCreateInfo semaphore_info = vkinit::semaphore_create_info();

	for (auto& frame : frames)
	{
		VK_CHECK(vkCreateFence(device, &fence_info, nullptr, &frame.render_fence));

		VK_CHECK(vkCreateSemaphore(device, &semaphore_info, nullptr, &frame.image_acquired_semaphore));
	}

	render_done_semaphores.resize(swapchain_images.size());
	for (auto& sem : render_done_semaphores)
	{
	    VK_CHECK(vkCreateSemaphore(device, &semaphore_info, nullptr, &sem));
	}

	VK_CHECK(vkCreateFence(device, &fence_info, nullptr, &imm_fence));

	main_deletion_queue.push_function([&]()
	{
		vkDestroyFence(device, imm_fence, nullptr);
	});
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height)
{
	vkb::SwapchainBuilder swapchainBuilder{ chosen_gpu, device, surface };

	swapchain_image_format = VK_FORMAT_B8G8R8A8_UNORM;
	// swapchain_image_format = VK_FORMAT_B8G8R8A8_SRGB;

	vkb::Swapchain vkbSwapchain = swapchainBuilder
	                                  //.use_default_format_selection()
	                                  .set_desired_format(VkSurfaceFormatKHR{ .format = swapchain_image_format, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
	                                  // .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
	                                  .set_desired_present_mode(VK_PRESENT_MODE_IMMEDIATE_KHR)
	                                  .set_desired_extent(width, height)
	                                  .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
	                                  .build()
	                                  .value();

	swapchain_extent = vkbSwapchain.extent;
	// store swapchain and its related images
	swapchain = vkbSwapchain.swapchain;
	swapchain_images = vkbSwapchain.get_images().value();
	swapchain_image_views = vkbSwapchain.get_image_views().value();
}

void VulkanEngine::destroy_swapchain()
{
	// destroys images held
	vkDestroySwapchainKHR(device, swapchain, nullptr);

	for (auto& swapchain_image_view : swapchain_image_views)
	{
		vkDestroyImageView(device, swapchain_image_view, nullptr);
	}
}

void VulkanEngine::init_descriptors()
{
	//> building scene descriptor layout
	{
		DescriptorLayoutBuilder builder{};
		builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_COMPUTE_BIT);
		scene_descriptor_layout = builder.build(device);
	}

	std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> uniform_sizes = {
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 5 },
	};

	DescriptorWriter writer{};
	for (auto& frame : frames)
	{
		frame.frame_descriptor_allocator.init(device, 1, uniform_sizes);

		main_deletion_queue.push_function([&]()
		{
			frame.frame_descriptor_allocator.destroy_pools(device);
		});

		frame.scene_buffer = create_buffer(allocator, sizeof(SceneData), VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	}

	std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {
		{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_SAMPLER, 20 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
		{ VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 }
	};

	global_descriptor_allocator.init(device, 1, sizes);

	{
		DescriptorLayoutBuilder builder{};
		builder.add_binding(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
		builder.bindings[0].descriptorCount = 1000; // UPPER BOUND

		std::array<VkDescriptorBindingFlags, 1> flags{};
		flags[0] = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
		//| VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;

		VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_info{};
		binding_flags_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
		binding_flags_info.bindingCount = 1;
		binding_flags_info.pBindingFlags = flags.data();

		bindless_tex_layout = builder.build(device, &binding_flags_info);

		builder.clear();
		builder.add_binding(0, VK_DESCRIPTOR_TYPE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
		builder.bindings[0].descriptorCount = 10; // validation layer not reporting if this is higher than pool maximum

		bindless_sampler_layout = builder.build(device, &binding_flags_info);

		builder.clear();
		builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT);
		builder.bindings[0].descriptorCount = 1000;

		bindless_image_layout = builder.build(device, &binding_flags_info);

		builder.clear();
		builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT);
		builder.bindings[0].descriptorCount = 1;

		rasterizer_ordered_buf_layout = builder.build(device);

		builder.clear();
		builder.add_binding(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_COMPUTE_BIT);
		builder.bindings[0].descriptorCount = 1;

		as_layout = builder.build(device);
	}

	main_deletion_queue.push_function([&]()
	{
		global_descriptor_allocator.destroy_pools(device);
		vkDestroyDescriptorSetLayout(device, scene_descriptor_layout, nullptr);
		vkDestroyDescriptorSetLayout(device, bindless_tex_layout, nullptr);
		vkDestroyDescriptorSetLayout(device, bindless_sampler_layout, nullptr);
		vkDestroyDescriptorSetLayout(device, bindless_image_layout, nullptr);
		vkDestroyDescriptorSetLayout(device, rasterizer_ordered_buf_layout, nullptr);
		vkDestroyDescriptorSetLayout(device, as_layout, nullptr);
	});
}

void VulkanEngine::init_shaders()
{
	shader_cache.add_shader(device, "cluster_grid.slang", sizeof(ClusterGridPushConstants));
	shader_cache.add_shader(device, "light_culling.slang", sizeof(LightCullingPushConstants));
	shader_cache.add_shader(device, "hiz.slang", sizeof(DepthPyramidPushConstants));
	shader_cache.add_shader(device, "mesh_cull.slang", sizeof(CullData));
	shader_cache.add_shader(device, "meshlet_cull.slang", sizeof(ClusterCullData));
	shader_cache.add_shader(device, "shadow_cull.slang", sizeof(ShadowCullPushConstants));
	shader_cache.add_shader(device, "resolve_taa.slang", sizeof(TAAPushConstants));
	shader_cache.add_shader(device, "equirectangular_to_cubemap.slang", sizeof(IBLPushConstants));
	shader_cache.add_shader(device, "spherical_harmonics.slang", sizeof(SHPushConstants));
	shader_cache.add_shader(device, "irradiance.slang", sizeof(IBLPushConstants));
	shader_cache.add_shader(device, "prefiltered.slang", sizeof(IBLPushConstants));
	shader_cache.add_shader(device, "brdf.slang", sizeof(IBLPushConstants));
	shader_cache.add_shader(device, "luminance_histogram.slang", sizeof(LuminanceBinsPushConstants));
	shader_cache.add_shader(device, "luminance_avg.slang", sizeof(LuminanceBinsPushConstants));
	shader_cache.add_shader(device, "tonemap.slang", sizeof(TonemapPushConstants));
	shader_cache.add_shader(device, "resolve_vbuffer.slang", sizeof(DeferredPushConstants));
	shader_cache.add_shader(device, "resolve_gbuffer.slang", sizeof(DeferredPushConstants));
	shader_cache.add_shader(device, "compact_dispatch.slang", sizeof(CompactDispatchPushConstants));
	shader_cache.add_shader(device, "hiz_spd.slang", sizeof(SpdPushConstants));
	shader_cache.add_shader(device, "depth.slang", sizeof(ShadowPushConstants));
	shader_cache.add_shader(device, "vbuffer.slang", sizeof(GPUPushConstants));
	shader_cache.add_shader(device, "gbuffer_vert.slang", sizeof(GPUPushConstants));
	shader_cache.add_shader(device, "gbuffer_mesh.slang", sizeof(GPUPushConstants));
	shader_cache.add_shader(device, "mlab_vert.slang", sizeof(GPUPushConstants));
	shader_cache.add_shader(device, "mlab_mesh.slang", sizeof(GPUPushConstants));
	// shader_cache.add_shader(device, "rt.slang", sizeof(DeferredPushConstants));
}

void VulkanEngine::init_pipelines()
{
	std::vector<VkDescriptorSetLayout> descriptor_layouts = { scene_descriptor_layout, bindless_image_layout, bindless_tex_layout, bindless_sampler_layout };

	ComputePipelineBuilder compute_builder{};
	compute_builder.set_descriptor_layouts({ scene_descriptor_layout, bindless_image_layout, bindless_tex_layout, bindless_sampler_layout });

	PipelineBuilder builder{};
	builder.set_descriptor_layouts({ scene_descriptor_layout, bindless_image_layout, bindless_tex_layout, bindless_sampler_layout });

	shader_passes["cluster_grid"] = compute_builder.create_pipeline(device, shader_cache["cluster_grid.slang"]);
	shader_passes["light_culling"] = compute_builder.create_pipeline(device, shader_cache["light_culling.slang"]);
	shader_passes["hiz"] = compute_builder.create_pipeline(device, shader_cache["hiz.slang"]);
	shader_passes["mesh_cull"] = compute_builder.create_pipeline(device, shader_cache["mesh_cull.slang"]);
	shader_passes["meshlet_cull"] = compute_builder.create_pipeline(device, shader_cache["meshlet_cull.slang"]); // TODO: check if this is also culldata
	shader_passes["equirectangular_to_cubemap"] = compute_builder.create_pipeline(device, shader_cache["equirectangular_to_cubemap.slang"]);
	shader_passes["spherical_harmonics"] = compute_builder.create_pipeline(device, shader_cache["spherical_harmonics.slang"]);
	shader_passes["irradiance"] = compute_builder.create_pipeline(device, shader_cache["irradiance.slang"]);
	shader_passes["prefiltered"] = compute_builder.create_pipeline(device, shader_cache["prefiltered.slang"]);
	shader_passes["brdf"] = compute_builder.create_pipeline(device, shader_cache["brdf.slang"]);
	shader_passes["luminance_histogram"] = compute_builder.create_pipeline(device, shader_cache["luminance_histogram.slang"]);
	shader_passes["luminance_avg"] = compute_builder.create_pipeline(device, shader_cache["luminance_avg.slang"]);
	shader_passes["tonemap"] = compute_builder.create_pipeline(device, shader_cache["tonemap.slang"]);
	shader_passes["shadow_cull"] = compute_builder.create_pipeline(device, shader_cache["shadow_cull.slang"]);
	shader_passes["compact_dispatch"] = compute_builder.create_pipeline(device, shader_cache["compact_dispatch.slang"]);
	shader_passes["resolve_taa"] = compute_builder.create_pipeline(device, shader_cache["resolve_taa.slang"]);
	shader_passes["hiz_spd"] = compute_builder.create_pipeline(device, shader_cache["hiz_spd.slang"]);
	compute_builder.set_descriptor_layouts({ scene_descriptor_layout, bindless_image_layout, bindless_tex_layout, bindless_sampler_layout, as_layout });
	shader_passes["resolve_gbuffer"] = compute_builder.create_pipeline(device, shader_cache["resolve_gbuffer.slang"]);
	shader_passes["resolve_vbuffer"] = compute_builder.create_pipeline(device, shader_cache["resolve_vbuffer.slang"]);
	// shader_passes["ray_tracing"] = compute_builder.create_pipeline(device, shader_cache["rt.slang"]);

	// mrt
	builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
	builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
	builder.set_multisampling_none();
	builder.enable_depth(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
	builder.set_depth_format(depth_image.format);
	builder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);

	std::vector<VkFormat> color_attachment_formats{};
	std::vector<VkPipelineColorBlendAttachmentState> color_blend_states{};
	for (size_t i = 0; i < GBUFFER_COUNT; i++)
	{
		color_attachment_formats.push_back(gbuffers[i].format);
		VkPipelineColorBlendAttachmentState state{};
		state.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		state.blendEnable = VK_FALSE;

		color_blend_states.push_back(state);
	}
	builder.set_color_attachment_format(color_attachment_formats);
	builder.set_blending_state(color_blend_states);
	builder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["geometry_vert"] = builder.create_pipeline(device, { shader_cache["gbuffer_vert.slang"] }, { VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT }, { "vs_main", "ps_main" }, { 1 });
	builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["geometry_vert_mask"] = builder.create_pipeline(device, { shader_cache["gbuffer_vert.slang"] }, { VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT }, { "vs_main", "ps_main" }, { 0 });

	builder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["geometry_mesh"] = builder.create_pipeline(device, { shader_cache["gbuffer_mesh.slang"] }, { VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_STAGE_FRAGMENT_BIT }, { "mesh_main", "ps_main" }, { 1 });
	builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["geometry_mesh_mask"] = builder.create_pipeline(device, { shader_cache["gbuffer_mesh.slang"] }, { VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_STAGE_FRAGMENT_BIT }, { "mesh_main", "ps_main" }, { 0 });

	color_attachment_formats.clear();
	color_attachment_formats.push_back(visibility_buffer.format);
	color_attachment_formats.push_back(velocity_buffer.format);
	builder.set_color_attachment_format(color_attachment_formats);
	color_blend_states.clear();
	color_blend_states.push_back(builder.disable_blending()); // 2 channel texture but RGBA write mask ok? no validation error
	color_blend_states.push_back(builder.disable_blending()); // 2 channel texture but RGBA write mask ok? no validation error
	builder.set_blending_state(color_blend_states);
	builder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["visibility_mesh"] = builder.create_pipeline(device, { shader_cache["vbuffer.slang"] }, { VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_STAGE_FRAGMENT_BIT }, { "mesh_main", "ps_main" }, { 1 });
	builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["visibility_mesh_mask"] = builder.create_pipeline(device, { shader_cache["vbuffer.slang"] }, {VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_STAGE_FRAGMENT_BIT}, { "mesh_main", "ps_main" }, { 0 });

	// single render target
	color_attachment_formats.clear();
	color_attachment_formats.push_back(VK_FORMAT_UNDEFINED);
	builder.set_color_attachment_format(color_attachment_formats);
	color_blend_states.clear();
	color_blend_states.push_back(builder.disable_blending());
	builder.set_blending_state(color_blend_states);
	builder.enable_depth(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
	builder.rasterization.depthClampEnable = VK_TRUE;
	builder.dynamic_state.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS);
	builder.set_cull_mode(VK_CULL_MODE_FRONT_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["depth"] = builder.create_pipeline(device, { shader_cache["depth.slang"] }, { VK_SHADER_STAGE_VERTEX_BIT }, { "vs_main" });
	builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["depth_mask"] = builder.create_pipeline(device, { shader_cache["depth.slang"] }, { VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT }, { "vs_main", "ps_main" }, { 0 });
	builder.dynamic_state.pop_back(); // reset
	builder.rasterization.depthClampEnable = VK_FALSE; // reset

	builder.enable_depth(false, VK_COMPARE_OP_GREATER_OR_EQUAL);
	color_attachment_formats.clear();
	color_attachment_formats.push_back(VK_FORMAT_UNDEFINED);
	builder.set_color_attachment_format(color_attachment_formats);
	builder.set_depth_format(depth_image.format);
	builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	builder.set_descriptor_layouts({ scene_descriptor_layout, bindless_image_layout, bindless_tex_layout, bindless_sampler_layout, rasterizer_ordered_buf_layout });
	shader_passes["mlab_vert"] = builder.create_pipeline(device, { shader_cache["mlab_vert.slang"] }, { VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT }, { "vs_main", "ps_main" });
	shader_passes["mlab_mesh"] = builder.create_pipeline(device, { shader_cache["mlab_mesh.slang"] }, { VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_STAGE_FRAGMENT_BIT }, { "mesh_main", "ps_main" });
}

void VulkanEngine::init_resources()
{
    {
    	VkSampler sampler{};
    	VkSamplerCreateInfo sampler_info{};
    	sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    	sampler_info.magFilter = VK_FILTER_LINEAR;
    	sampler_info.minFilter = VK_FILTER_LINEAR;

    	sampler_info.maxLod = VK_LOD_CLAMP_NONE;
    	sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    	// sampler_info.anisotropyEnable = VK_TRUE;
    	// sampler_info.maxAnisotropy = 16.0f;

    	vkCreateSampler(device, &sampler_info, nullptr, &sampler); // 0 linear
    	sampler_cache.add_sampler(sampler);

    	// sampler_info.anisotropyEnable = VK_FALSE;
    	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

    	vkCreateSampler(device, &sampler_info, nullptr, &sampler); // 1 cube map sampling
    	sampler_cache.add_sampler(sampler);

    	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; // tailored to our PCF sampling; manual OOB rejection required in shader
    	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    	sampler_info.maxLod = 1.0;

    	vkCreateSampler(device, &sampler_info, nullptr, &sampler); // 2 shadow map sampler
    	sampler_cache.add_sampler(sampler);

    	sampler_info.magFilter = VK_FILTER_LINEAR;
    	sampler_info.minFilter = VK_FILTER_LINEAR;
    	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    	sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    	sampler_info.maxLod = VK_LOD_CLAMP_NONE;

    	VkSamplerReductionModeCreateInfo reduction_info{};
    	reduction_info.sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO;
    	reduction_info.reductionMode = VK_SAMPLER_REDUCTION_MODE_MIN;

    	sampler_info.pNext = &reduction_info;

    	vkCreateSampler(device, &sampler_info, nullptr, &sampler); // 3 building hi-z
    	sampler_cache.add_sampler(sampler);

    	sampler_info.pNext = nullptr;
    	sampler_info.magFilter = VK_FILTER_NEAREST;
    	sampler_info.minFilter = VK_FILTER_NEAREST;
    	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    	sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;

    	vkCreateSampler(device, &sampler_info, nullptr, &sampler);
    	sampler_cache.add_sampler(sampler); // 4 nearest clamp to border

    	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    	vkCreateSampler(device, &sampler_info, nullptr, &sampler);
    	sampler_cache.add_sampler(sampler); // 5 nearest clamp to edge

    	sampler_info.magFilter = VK_FILTER_LINEAR;
    	sampler_info.minFilter = VK_FILTER_LINEAR;
    	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    	sampler_info.maxLod = VK_LOD_CLAMP_NONE;
    	sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    	vkCreateSampler(device, &sampler_info, nullptr, &sampler);
    	sampler_cache.add_sampler(sampler); // 6 linear clamp to edge
    }

	// shadowmaps
	for (size_t idx = 0; idx < cascade_data.size(); idx++)
	{
		cascade_data[idx].shadow_map = create_image(
		    device, allocator, VkExtent3D{ SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1 },
		    VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT
		);
		auto id = texture_cache.add_texture(cascade_data[idx].shadow_map.view);
		if (idx == 0)
			texture_cache.set_shadowmap(id);
	}

	main_deletion_queue.push_function([&]()
	{
		for (auto& cascade : cascade_data)
		{
			destroy_image(device, allocator, cascade.shadow_map);
		}
	});

	//> create depth pyramid
	VkExtent3D depth_pyramid_extent{};
	depth_pyramid_extent.width = nearest_pow2(draw_extent.width);
	depth_pyramid_extent.height = nearest_pow2(draw_extent.height);
	depth_pyramid_extent.depth = 1;

	depth_pyramid = create_image(device, allocator, depth_pyramid_extent, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, true);

	// sampling in occlusion culling
	auto id = texture_cache.add_texture(depth_pyramid.view);
	texture_cache.set_depth_pyramid_image(id);

	uint32_t mip_levels = static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(std::max(depth_pyramid_extent.width, depth_pyramid_extent.height))))) + 1;

	std::vector<VkImageView> pyramid_views(mip_levels);
	VkImageViewCreateInfo img_view_info = vkinit::imageview_create_info(VK_FORMAT_R32_SFLOAT, depth_pyramid.image, VK_IMAGE_ASPECT_COLOR_BIT);
	img_view_info.subresourceRange.levelCount = 1;
	img_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
	img_view_info.subresourceRange.baseMipLevel = 0;
	vkCreateImageView(device, &img_view_info, nullptr, &pyramid_views[0]);
	id = image_cache.add_texture(pyramid_views[0]);
	image_cache.set_depth_pyramid_image(id);

	for (uint32_t mip = 1; mip < mip_levels; mip++)
	{
		img_view_info.subresourceRange.baseMipLevel = mip;
		vkCreateImageView(device, &img_view_info, nullptr, &pyramid_views[mip]);
		image_cache.add_texture(pyramid_views[mip]);
	}

	main_deletion_queue.push_function([&, pyramid_views]()
	{
		destroy_image(device, allocator, depth_pyramid);
		for (auto pyramid_view : pyramid_views)
		{
			vkDestroyImageView(device, pyramid_view, nullptr);
		}
	});

	// global light list
	std::mt19937 mt(42);
	std::uniform_real_distribution<float> pos_dist(-1.0f, 1.0f);
	std::uniform_real_distribution<float> color_dist(0.f, 1.0f);

	std::vector<PointLight> light_data(MAX_POINT_LIGHTS);

	float light_area = 10.f; // in radius
	float light_radius = 1.f;

	for (size_t i = 0; i < MAX_POINT_LIGHTS; i++)
	{
		light_data[i].pos = glm::vec4(pos_dist(mt) * light_area, std::abs(pos_dist(mt) * light_area), pos_dist(mt) * light_area, light_radius); // pos & radius
		light_data[i].color = glm::vec4(color_dist(mt), color_dist(mt), color_dist(mt), 1.0);
	}

	light_buffer = upload_buffer(this, allocator, light_data.data(), MAX_POINT_LIGHTS * sizeof(PointLight));
	fmt::println("light_buffer: {}mb", size_in_bytes(light_buffer.info.size));

	const uint32_t total_clusters = CLUSTER_X * CLUSTER_Y * CLUSTER_DEPTH_SLICES;

	light_cluster_buffer = create_buffer(allocator, total_clusters * sizeof(ClusterAABB), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
	light_index_buffer = create_buffer(allocator, total_clusters * MAX_POINT_LIGHTS * sizeof(uint32_t), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT); // could use smaller more conservative size
	light_grid_buffer = create_buffer(allocator, total_clusters * sizeof(LightGrid), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
	light_count_buffer = create_buffer(allocator, sizeof(uint32_t), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

	// gi
	const char* hdri_path = { "assets/pisa.hdr" };
	float* data{};

	int width{};
	int height{};
	int channels{};

	data = stbi_loadf(hdri_path, &width, &height, &channels, STBI_rgb_alpha);

	auto extent = VkExtent3D(static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1);

	hdri = upload_image(this, device, allocator, (void*)data, extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

	auto hdri_id = texture_cache.add_texture(hdri.view);
	texture_cache.set_hdri(hdri_id);

	extent.width /= 4;
	extent.height = extent.width;

	hdri_cubemap = create_cubemap(device, allocator, extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, true);

	scene_data.textures[0] = static_cast<float>(texture_cache.add_texture(hdri_cubemap.view));
	auto hdri_cubemap_id = image_cache.add_texture(hdri_cubemap.view);
	image_cache.set_hdri(hdri_cubemap_id);

	irradiance_cubemap = create_cubemap(device, allocator, VkExtent3D{ 64, 64, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

	scene_data.textures[1] = static_cast<float>(texture_cache.add_texture(irradiance_cubemap.view));
	image_cache.add_texture(irradiance_cubemap.view);

	prefiltered_envmap = create_cubemap(device, allocator, VkExtent3D{ 512, 512, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, true);

	scene_data.textures[2] = static_cast<float>(texture_cache.add_texture(prefiltered_envmap.view));

	auto prefiltered_mips = static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(std::max(prefiltered_envmap.extent.width, prefiltered_envmap.extent.height))))) + 1;

	std::vector<VkImageView> prefiltered_views(prefiltered_mips);
	auto imageview_info = vkinit::imageview_create_info(VK_FORMAT_R32G32B32A32_SFLOAT, prefiltered_envmap.image, VK_IMAGE_ASPECT_COLOR_BIT);
	imageview_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
	for (uint32_t i = 0; i < prefiltered_mips; ++i)
	{
		imageview_info.subresourceRange.baseMipLevel = i;
		imageview_info.subresourceRange.levelCount = 1;
		vkCreateImageView(device, &imageview_info, nullptr, &prefiltered_views[i]);
		image_cache.add_texture(prefiltered_views[i]);
	}

	brdf_lut = create_image(device, allocator, VkExtent3D{ 128, 128, 1 }, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

	scene_data.textures[3] = static_cast<float>(texture_cache.add_texture(brdf_lut.view));
	image_cache.add_texture(brdf_lut.view);

	main_deletion_queue.push_function([&, prefiltered_mips, prefiltered_views]()
	{
		destroy_buffer(allocator, light_buffer);
		destroy_buffer(allocator, light_cluster_buffer);
		destroy_buffer(allocator, light_index_buffer);
		destroy_buffer(allocator, light_grid_buffer);
		destroy_buffer(allocator, light_count_buffer);
		destroy_image(device, allocator, hdri);
		destroy_image(device, allocator, hdri_cubemap);
		destroy_image(device, allocator, irradiance_cubemap);
		destroy_image(device, allocator, prefiltered_envmap);
		destroy_image(device, allocator, brdf_lut);
		for (uint32_t i = 0; i < prefiltered_mips; i++)
		{
			vkDestroyImageView(device, prefiltered_views[i], nullptr);
		}
	});
}

void VulkanEngine::init_renderables(int argc, char** argv)
{
	auto start = std::chrono::system_clock::now();

	{
	    std::vector<std::string> file_paths(argc - 1);
		for (int i = 1; i < argc; i++)
		{
		    file_paths[i-1] = argv[i];
		}
		auto asset_file = load_gltfs(this, file_paths);
		assert(asset_file.has_value());
		loaded_scene = std::move(*asset_file);
	}

	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	float ret = static_cast<float>(elapsed.count()) / 1000.0f;
	fmt::println("load gltf: {}ms", ret);

	VkBufferUsageFlags ray_tracing_flags = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

	render_scene.vertex_buffer = upload_buffer(this, allocator, loaded_scene->vertices.data(), loaded_scene->vertices.size() * sizeof(Vertex), ray_tracing_flags);
	render_scene.index_buffer = upload_buffer(this, allocator, loaded_scene->indices.data(), loaded_scene->indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | ray_tracing_flags);
	render_scene.meshlet_indices = upload_buffer(this, allocator, loaded_scene->meshlet_indices.data(), loaded_scene->meshlet_indices.size() * sizeof(uint32_t));
	render_scene.meshlet_buffer = upload_buffer(this, allocator, loaded_scene->meshlets.data(), loaded_scene->meshlets.size() * sizeof(Meshlet));
	render_scene.material_buffer = upload_buffer(this, allocator, loaded_scene->materials.data(), loaded_scene->materials.size() * sizeof(MaterialData));
	fmt::println("vertex_buffer: {}mb", size_in_bytes(render_scene.vertex_buffer.info.size));
	fmt::println("index_buffer: {}mb", size_in_bytes(render_scene.index_buffer.info.size));
	fmt::println("meslet_indices: {}mb", size_in_bytes(render_scene.meshlet_indices.info.size));
	fmt::println("meshlet_buffer: {}mb", size_in_bytes(render_scene.meshlet_buffer.info.size));
	fmt::println("material_buffer: {}mb", size_in_bytes(render_scene.material_buffer.info.size));

	for (const auto& n : loaded_scene->top_nodes)
	{
		register_object(n.get(), glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, 2)));
	}

#ifndef SINGLE
	std::mt19937 mt(42);
	auto draw_radius = 400.0f;
	auto draw_count = 500'000;

	for (size_t i = 0; i < draw_count; i++)
	{
		const float x = static_cast<float>(mt()) / static_cast<float>(mt.max()) * draw_radius - draw_radius * 0.5f;
		const float y = static_cast<float>(mt()) / static_cast<float>(mt.max()) * draw_radius - draw_radius * 0.5f;
		const float z = static_cast<float>(mt()) / static_cast<float>(mt.max()) * -draw_radius;

		glm::mat4 t = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
		glm::vec3 axis = glm::normalize(
		    glm::vec3(static_cast<float>(mt()) / static_cast<float>(mt.max()), static_cast<float>(mt()) / static_cast<float>(mt.max()), static_cast<float>(mt()) / static_cast<float>(mt.max()))
		);
		glm::mat4 r = glm::rotate(glm::mat4(1.0f), glm::radians(static_cast<float>(mt()) / static_cast<float>(mt.max()) * 360.0f), axis);
		glm::mat4 s = glm::scale(glm::mat4(1.0f), glm::vec3(static_cast<float>(mt()) / static_cast<float>(mt.max())) + 1.0f);
		const auto transform = t * r * s;

		for (const auto& n : loaded_scene->top_nodes)
		{
			register_object(n.get(), transform);
		}
	}
#endif

	uint32_t meshlet_visibility_offset{};
	for (auto& renderable : render_scene.renderables)
	{
		uint32_t meshlet_count = renderable.meshlet_bits; // meshlet count for LOD 0 only
		renderable.meshlet_bits = meshlet_visibility_offset; // TODO: rename meshlet_bits
		meshlet_visibility_offset += meshlet_count;
	}
	render_scene.total_meshlets_bits = meshlet_visibility_offset;
}

void VulkanEngine::update_descriptors()
{
	std::array<uint32_t, 1> variable_desc_counts = {
		static_cast<uint32_t>(texture_cache.image_infos.size())
	};

	VkDescriptorSetVariableDescriptorCountAllocateInfo variable_desc_info{};
	variable_desc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
	variable_desc_info.pDescriptorCounts = variable_desc_counts.data();
	variable_desc_info.descriptorSetCount = static_cast<uint32_t>(variable_desc_counts.size());

	bindless_tex_descriptor = global_descriptor_allocator.allocate(device, bindless_tex_layout, &variable_desc_info);
	variable_desc_counts[0] = static_cast<uint32_t>(sampler_cache.image_infos.size());
	bindless_sampler_descriptor = global_descriptor_allocator.allocate(device, bindless_sampler_layout, &variable_desc_info);
	variable_desc_counts[0] = static_cast<uint32_t>(image_cache.image_infos.size());
	bindless_image_descriptor = global_descriptor_allocator.allocate(device, bindless_image_layout, &variable_desc_info);

	std::vector<VkWriteDescriptorSet> writes{};

	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = bindless_tex_descriptor;
	write.dstBinding = 0;
	write.descriptorCount = static_cast<uint32_t>(texture_cache.image_infos.size()); // validation layer does not report if smaller count than req used
	write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
	write.pImageInfo = texture_cache.image_infos.data();
	writes.push_back(write);

	write.dstSet = bindless_sampler_descriptor;
	write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
	write.pImageInfo = sampler_cache.image_infos.data();
	write.descriptorCount = static_cast<uint32_t>(sampler_cache.image_infos.size());
	writes.push_back(write);

	write.dstSet = bindless_image_descriptor;
	write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	write.pImageInfo = image_cache.image_infos.data();
	write.descriptorCount = static_cast<uint32_t>(image_cache.image_infos.size());
	writes.push_back(write);

	vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

	rasterizer_ordered_buf_descriptor = global_descriptor_allocator.allocate(device, rasterizer_ordered_buf_layout);

	DescriptorWriter writer{};
	writer.clear();
	writer.write_buffer(0, render_scene.oit_buffer.buffer, VK_WHOLE_SIZE, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	writer.update_set(device, rasterizer_ordered_buf_descriptor);

	as_descriptor = global_descriptor_allocator.allocate(device, as_layout);

	VkWriteDescriptorSetAccelerationStructureKHR as_info{};
	as_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
	as_info.accelerationStructureCount = 1;
	as_info.pAccelerationStructures = &tlas_as;

	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.pNext = &as_info;
	write.dstSet = as_descriptor;
	write.descriptorCount = 1;
	write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

	vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

	write.pNext = nullptr; // in case of future reuse

	for (auto& frame : frames)
	{
    	frame.scene_descriptor = frame.frame_descriptor_allocator.allocate(device, scene_descriptor_layout);

    	writer.clear();
    	writer.write_buffer(0, frame.scene_buffer.buffer, sizeof(SceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    	writer.update_set(device, frame.scene_descriptor);
	}
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

			RenderObject obj{};

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
			obj.meshlet_bits = mesh.meshlet_bits;

			if (found)
			{
				obj.mesh_id = static_cast<uint32_t>(handle + i);
			}
			else
			{
				obj.mesh_id = static_cast<uint32_t>(render_scene.meshes.size());

				render_scene.meshes.emplace_back(Mesh{ .center = mesh.center, .radius = mesh.radius, .lod_count = mesh.lod_count, .vertex_offset = mesh.vertex_offset, .mesh_lods = mesh.mesh_lods });
			}

			switch (mesh.pass)
			{
			case MaterialPass::Mask:
				obj.post_pass = 1;
				break;
			case MaterialPass::Blend:
				obj.post_pass = 2;
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
	main_camera.far = static_cast<float>(CVAR_MISC_DRAW_DISTANCE.get());
	main_camera.update(static_cast<float>(stats.deltatime));

	scene_data.view = main_camera.get_view_matrix();
	scene_data.proj = main_camera.perspective;

	if (CVAR_RENDER_TAA.get())
	{
		auto idx = frame_number % 8;
		auto offset_projection = glm::translate(glm::mat4(1.0f), glm::vec3(jitter_offset[idx].x, jitter_offset[idx].y, 0.0));
		scene_data.proj = offset_projection * scene_data.proj;
	}

	// TODO: how we handling frame 0?
	scene_data.previous_viewproj = scene_data.viewproj;
	scene_data.viewproj = scene_data.proj * scene_data.view;
	scene_data.inverse_viewproj = glm::inverse(scene_data.viewproj);
	last_view = freeze_camera ? last_view : scene_data.view;
	last_proj = freeze_camera ? last_proj : scene_data.proj;

	scene_data.sunlight_dir = glm::vec4(7.75, 12.5, 12.5, 1.);
	// scene_data.sunlight_dir = glm::vec4(0.001, 12.0, 0.0, 1.);
	scene_data.sunlight_color = glm::vec4(15.0, 15.0, 15.0, 1.0);

	if (CVAR_RENDER_SHADOWS.get() && !CVAR_RENDER_SHADOWS_RT.get())
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
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& func) const
{
	VK_CHECK(vkResetFences(device, 1, &imm_fence));
	VK_CHECK(vkResetCommandPool(device, imm_command_pool, 0));

	VkCommandBufferBeginInfo cmd_begin_info = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(imm_command_buffer, &cmd_begin_info));

	func(imm_command_buffer);

	VK_CHECK(vkEndCommandBuffer(imm_command_buffer));

	VkCommandBufferSubmitInfo cmd_submit_info = vkinit::command_buffer_submit_info(imm_command_buffer);

	VkSubmitInfo2 submit = vkinit::submit_info(&cmd_submit_info, nullptr, nullptr);

	VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit, imm_fence));

	VK_CHECK(vkWaitForFences(device, 1, &imm_fence, true, 9999999999));
}

void VulkanEngine::resolve_taa(VkCommandBuffer cmd)
{
	ShaderPass current_pass = *shader_passes["resolve_taa"];
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);

	TAAPushConstants pc{};
	auto jitter_count = jitter_offset.size();
	auto current_jitter = jitter_offset[frame_number % jitter_count];
	auto previous_jitter = jitter_offset[(frame_number - 1) % jitter_count];
	pc.jitter_offset = glm::vec4(current_jitter, previous_jitter);
	pc.screen_size = glm::vec2(static_cast<float>(draw_extent.width), static_cast<float>(draw_extent.height));
	pc.current_id = texture_cache.get_draw_image();
	pc.history_id = texture_cache.get_accumulation_buffer((frame_number + 1) % 2);
	pc.resolve_id = image_cache.get_accumulation_buffer(frame_number % 2);
	pc.depth_id = texture_cache.get_depth_image();
	pc.velocity_id = texture_cache.get_visibility_buffer() + 1; // TODO: hardcoded, maybe give velocity its own setter/getter?
	pc.variance_clipping = CVAR_TAA_VARIANCE_CLIP.get();
	pc.history_filter = CVAR_TAA_CATMULL_ROM.get();
	pc.local_filter = CVAR_TAA_MITCHELL.get();
	pc.ycocg = CVAR_TAA_YCOCG.get();
	pc.valid_history = first_frame ? 0 : 1;
	pc.dynamic = CVAR_TAA_DYNAMIC.get();
	first_frame = false; // set this elsewhere?

	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TAAPushConstants), &pc);
	auto groupcount_x = get_groupcount(draw_extent.width, WARP_SIZE);
	auto groupcount_y = get_groupcount(draw_extent.height, WARP_SIZE);
	vkCmdDispatch(cmd, groupcount_x, groupcount_y, 1);
}

void VulkanEngine::update_cascade()
{
	// https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-10-parallel-split-shadow-maps-programmable-gpus
	float near = static_cast<float>(CVAR_SHADOWS_DISTANCE.get());
	float far = main_camera.far;
	float m = static_cast<float>(NUMBER_OF_CASCADES);
	float range = far - near;
	float ratio = far / near;
	float lambda = CVAR_SHADOWS_CASCADE_SPLIT.get();

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

	// TODO: refactor when implementing window resize
	glm::mat4 proj = glm::perspective(glm::radians(main_camera.fov), static_cast<float>(draw_extent.width) / static_cast<float>(draw_extent.height), static_cast<float>(CVAR_SHADOWS_DISTANCE.get()), main_camera.near);
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
	init_info.PhysicalDevice = chosen_gpu;
	init_info.Device = device;
	init_info.QueueFamily = graphics_queue_family;
	init_info.Queue = graphics_queue;
	init_info.DescriptorPool = imgui_pool;
	init_info.MinImageCount = 2;
	init_info.ImageCount = 2;
	init_info.UseDynamicRendering = true;
	VkPipelineRenderingCreateInfo render_info{};
	render_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
	render_info.colorAttachmentCount = 1;
	render_info.pColorAttachmentFormats = &swapchain_image_format;
	init_info.PipelineInfoMain.PipelineRenderingCreateInfo = render_info;

	// https://github.com/ocornut/imgui/issues/4854#issuecomment-1362380609 FOR VOLK COMPATIBILITY
	ImGui_ImplVulkan_LoadFunctions(0, [](const char* function_name, void* vulkan_instance)
	{ return vkGetInstanceProcAddr(*(static_cast<VkInstance*>(vulkan_instance)), function_name); }, &instance);

	ImGui_ImplVulkan_Init(&init_info);

	main_deletion_queue.push_function([&, imgui_pool]()
	{
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplSDL3_Shutdown();
		ImGui::DestroyContext();
		vkDestroyDescriptorPool(device, imgui_pool, nullptr);
	});
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView swapchain_view)
{
	VkRenderingAttachmentInfo color_attachment = vkinit::attachment_info(swapchain_view, nullptr);
	VkRenderingInfo render_info = vkinit::rendering_info(swapchain_extent, &color_attachment, nullptr);

	vkCmdBeginRendering(cmd, &render_info);

	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

	vkCmdEndRendering(cmd);
}

void VulkanEngine::upload_buffers()
{

	render_scene.object_buffer = upload_buffer(this, allocator, render_scene.renderables.data(), render_scene.renderables.size() * sizeof(ObjectData), 0);
	fmt::println("object_buffer: {}mb", size_in_bytes(render_scene.object_buffer.info.size));

	render_scene.mesh_buffer = upload_buffer(this, allocator, render_scene.meshes.data(), render_scene.meshes.size() * sizeof(Mesh), 0);
	fmt::println("mesh_buffer: {}mb", size_in_bytes(render_scene.mesh_buffer.info.size));

	render_scene.sh_buffer = create_buffer(allocator, 27 * sizeof(float), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
	fmt::println("sh_buffer: {}mb", size_in_bytes(render_scene.sh_buffer.info.size));

	render_scene.luminance_buffer = create_buffer(allocator, 256 * sizeof(uint32_t), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
	fmt::println("luminance_buffer: {}mb", size_in_bytes(render_scene.luminance_buffer.info.size));

	render_scene.luminance_avg_buffer = create_buffer(allocator, sizeof(float), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
	fmt::println("luminance_avg_buffer: {}mb", size_in_bytes(render_scene.luminance_avg_buffer.info.size));

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

		render_scene.indices_buffer = upload_buffer(this, allocator, staging.data(), total * sizeof(uint32_t));
		fmt::println("indices_buffer: {}mb", size_in_bytes(render_scene.indices_buffer.info.size));
	}

	// allocating for worst case
	render_scene.vis_buffer = create_buffer(allocator, render_scene.renderables.size() * sizeof(uint32_t), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
	fmt::println("vis_buffer: {}mb", size_in_bytes(render_scene.vis_buffer.info.size));

	// TODO: implement limit, currently shader side has 1000000 hardcoded
	// TODO: modify with shadows in mind
	render_scene.prefix_sum_buffer = create_buffer(allocator, sizeof(uint64_t) + render_scene.renderables.size() * sizeof(PrefixSumData), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
	fmt::println("prefix_sum_buffer: {}mb", size_in_bytes(render_scene.prefix_sum_buffer.info.size));

	immediate_submit([&](VkCommandBuffer cmd)
	{
		vkCmdFillBuffer(cmd, render_scene.vis_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, render_scene.prefix_sum_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
	});

	render_scene.dispatch_buffer = create_buffer(allocator, 3 * sizeof(uint32_t), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT);
	// TODO: can we combine both of these?
	render_scene.meshlet_dispatch_buffer = create_buffer(allocator, 3 * sizeof(uint32_t), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT);

	auto count_size = 2 * sizeof(uint32_t);
	auto draw_commands_size = (MAX_OPAQUE_DRAWS + MAX_ALPHACLIP_DRAWS) * sizeof(VkDrawIndexedIndirectCommand);
	render_scene.draw_indirect_buffer = create_buffer(allocator, (count_size + draw_commands_size) * NUMBER_OF_CASCADES, 0, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
	fmt::println("draw_indirect_buffer: {}mb", size_in_bytes(render_scene.draw_indirect_buffer.info.size));

	// limit of ~16.7 meshlets, ~64mb
	// TODO: implement error handling/limit check in shader; just drop the meshlets?
	render_scene.cluster_indices = create_buffer(allocator, MESHLET_LIMIT * sizeof(uint32_t), 0, VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT);
	fmt::println("cluster_indices: {}mb", size_in_bytes(render_scene.cluster_indices.info.size));

	{
		size_t meshlet_visibility_size = (render_scene.total_meshlets_bits + 31) / 32;
		render_scene.meshlet_vis_buffer = create_buffer(allocator, meshlet_visibility_size * sizeof(uint32_t), 0, VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
		fmt::println("number of instances: {}", render_scene.renderables.size());
		fmt::println("meshlet_vis_buffer: {}mb", size_in_bytes(render_scene.meshlet_vis_buffer.info.size));

		immediate_submit([&](VkCommandBuffer cmd)
		{
			vkCmdFillBuffer(cmd, render_scene.meshlet_vis_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		});
	}

	// TODO: refactor if window resize
	{
		auto screen_pixels = window_extent.width * window_extent.height;
		render_scene.oit_buffer = create_buffer(allocator, screen_pixels * sizeof(OITData), 0, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
		fmt::println("oit_buffer: {}mb", size_in_bytes(render_scene.oit_buffer.info.size));

		immediate_submit([&](VkCommandBuffer cmd)
		{
			vkCmdFillBuffer(cmd, render_scene.oit_buffer.buffer, 0, VK_WHOLE_SIZE, 0x3F800000);
		});
	}

	{
	    render_scene.spd_counter_buffer = create_buffer(allocator, sizeof(uint32_t), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
		immediate_submit([&](VkCommandBuffer cmd)
		{
			vkCmdFillBuffer(cmd, render_scene.spd_counter_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		});
	}

	// TODO: move filling of buffers to a single location?
}

// indices address, count, late & post_pass set in executecomputecull
void VulkanEngine::ready_mesh_cull(RenderScene::MeshPass& pass, CullData& cull_data, glm::mat4& proj)
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
	cull_data.object_buffer_address = get_buffer_address(device, render_scene.object_buffer.buffer);
	cull_data.mesh_buffer_address = get_buffer_address(device, render_scene.mesh_buffer.buffer);
	cull_data.draw_indirect_address = get_buffer_address(device, render_scene.draw_indirect_buffer.buffer);
	cull_data.dispatch_buffer_address = get_buffer_address(device, render_scene.dispatch_buffer.buffer);
	cull_data.vis_buffer_address = get_buffer_address(device, render_scene.vis_buffer.buffer);
	cull_data.prefix_sum_buffer = get_buffer_address(device, render_scene.prefix_sum_buffer.buffer);

	// cull_data.count = static_cast<uint32_t>(pass.unbatched_objects.size()); // set during execute
	cull_data.texture_id = texture_cache.get_depth_pyramid_image();
	cull_data.occlusion_enabled = CVAR_RENDER_OCCLUSION_CULL.get();

	cull_data.p00 = proj[0][0];
	cull_data.p11 = proj[1][1]; // equivalent to 1 / tan(fovy/2)
	cull_data.near = main_camera.near;
	cull_data.far = main_camera.far;

	cull_data.resolution = glm::vec2(depth_pyramid.extent.width, depth_pyramid.extent.height);
	cull_data.texture_lod = static_cast<float>(std::floor(std::log2(static_cast<float>(std::max(depth_pyramid.extent.width, depth_pyramid.extent.height)))) + 1);
	cull_data.lod_distance_factor = 2.0f / (cull_data.p11 * static_cast<float>(draw_extent.height));
	cull_data.lod_enabled = CVAR_RENDER_LOD.get();
	cull_data.task_submit = CVAR_RENDER_MESH_SHADERS.get();
}

// count, late & post_pass set in executecomputecull
void VulkanEngine::ready_meshlet_cull(RenderScene::MeshPass& pass, ClusterCullData& cull_data, glm::mat4& proj)
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

	cull_data.object_buffer_address = get_buffer_address(device, render_scene.object_buffer.buffer);
	cull_data.meshlet_buffer_address = get_buffer_address(device, render_scene.meshlet_buffer.buffer);
	cull_data.cluster_indices_address = get_buffer_address(device, render_scene.cluster_indices.buffer);
	cull_data.meshlet_dispatch_address = get_buffer_address(device, render_scene.meshlet_dispatch_buffer.buffer);
	cull_data.cluster_vis_address = get_buffer_address(device, render_scene.meshlet_vis_buffer.buffer);
	cull_data.prefix_sum_buffer = get_buffer_address(device, render_scene.prefix_sum_buffer.buffer);

	// cull_data.count = static_cast<uint32_t>(pass.unbatched_objects.size()); // unused
	cull_data.texture_id = texture_cache.get_depth_pyramid_image();
	cull_data.occlusion_enabled = CVAR_RENDER_OCCLUSION_CULL.get();

	cull_data.p00 = proj[0][0];
	cull_data.p11 = proj[1][1];
	cull_data.near = main_camera.near;
	cull_data.far = main_camera.far;

	cull_data.resolution = glm::vec2(depth_pyramid.extent.width, depth_pyramid.extent.height);
	cull_data.texture_lod = static_cast<float>(std::floor(std::log2(static_cast<float>(std::max(depth_pyramid.extent.width, depth_pyramid.extent.height)))) + 1);
	cull_data.lod_distance_factor = 2.0f / (cull_data.p11 * static_cast<float>(draw_extent.height));
	cull_data.lod_enabled = CVAR_RENDER_LOD.get();
	cull_data.task_submit = CVAR_RENDER_MESH_SHADERS.get();
}

void VulkanEngine::execute_compact_dispatch(VkCommandBuffer cmd)
{
	ShaderPass current_pass = *shader_passes["compact_dispatch"];

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);

	CompactDispatchPushConstants pc{};
	pc.prefix_sum_buffer = get_buffer_address(device, render_scene.prefix_sum_buffer.buffer);
	pc.dispatch_buffer = get_buffer_address(device, render_scene.dispatch_buffer.buffer);

	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CompactDispatchPushConstants), &pc);
	vkCmdDispatch(cmd, 1, 1, 1);
}

void VulkanEngine::execute_compute_cull(VkCommandBuffer cmd, RenderScene::MeshPass& pass, CullData& cull_data, bool late, uint32_t post_pass)
{
	ShaderPass current_pass = *shader_passes["mesh_cull"];
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);
	cull_data.indices_buffer_address = get_buffer_address(device, render_scene.indices_buffer.buffer);
	cull_data.indices_buffer_address += pass.indices_offset * sizeof(uint32_t);

	cull_data.count = static_cast<uint32_t>(pass.unbatched_objects.size());
	cull_data.late = late ? 1 : 0;
	cull_data.post_pass = post_pass;

	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullData), &cull_data);
	auto groupcount_x = get_groupcount(static_cast<uint32_t>(pass.unbatched_objects.size()), CULL_WGSIZE);
	vkCmdDispatch(cmd, groupcount_x, 1, 1);
}

void VulkanEngine::execute_compute_cull(VkCommandBuffer cmd, ClusterCullData& cull_data, VkBuffer dispatch_buffer, uint32_t offset, bool late, uint32_t post_pass)
{
	ShaderPass current_pass = *shader_passes["meshlet_cull"];
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);

	// cull_data.count; // unused
	cull_data.late = late ? 1 : 0;
	cull_data.post_pass = post_pass;

	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ClusterCullData), &cull_data);

	// TODO: offset is always 0 as alphaclip and transparent (latter probably leaving it as is in the future) are not culled with opaque, hence we write over opaque's space
	vkCmdDispatchIndirect(cmd, dispatch_buffer, offset);
}

void VulkanEngine::execute_shadow_cull(VkCommandBuffer cmd)
{
	ShaderPass current_pass = *shader_passes["shadow_cull"];
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

	ShadowCullPushConstants pc{};
	pc.object_buffer_address = get_buffer_address(device, render_scene.object_buffer.buffer);
	pc.mesh_buffer_address = get_buffer_address(device, render_scene.mesh_buffer.buffer);
	pc.indices_buffer_address = get_buffer_address(device, render_scene.indices_buffer.buffer);
	pc.draw_buffer_address = get_buffer_address(device, render_scene.draw_indirect_buffer.buffer);

	std::vector<RenderScene::MeshPass*> passes = { &render_scene.opaque_pass, &render_scene.mask_pass };
	uint32_t cull_count{};
	for (const auto& pass : passes)
	{
		cull_count += static_cast<uint32_t>(pass->unbatched_objects.size());
	}
	pc.count = cull_count;
	pc.lod_enabled = CVAR_RENDER_LOD.get();

	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShadowCullPushConstants), &pc);
	auto groupcount_x = get_groupcount(cull_count, CULL_WGSIZE);
	vkCmdDispatch(cmd, groupcount_x, 1, 1);
}

void VulkanEngine::render(VkCommandBuffer cmd, bool late, uint32_t post_pass, uint32_t query)
{
	vkCmdBeginQuery(cmd, get_current_frame().query_pool_pipelines, query, 0);

	// deferred
	VkClearColorValue clear_color_value{ { 0.f, 0.f, 0.f, 1.0f } };
	VkClearValue clear_value{ .color = clear_color_value };
	VkClearColorValue clear_color_value2{ { 1.f, 1.f, 0.f, 1.0f } };
	VkClearValue clear_value2{ .color = clear_color_value2 };

	std::vector<VkRenderingAttachmentInfo> rendering_attachment_infos{};
	bool visibility_rendering = CVAR_RENDER_VBUFFER.get() && CVAR_RENDER_MESH_SHADERS.get();
	if (visibility_rendering)
	{
		rendering_attachment_infos.push_back(late ? vkinit::attachment_info(visibility_buffer.view, nullptr) : vkinit::attachment_info(visibility_buffer.view, &clear_value));
		rendering_attachment_infos.push_back(late ? vkinit::attachment_info(velocity_buffer.view, nullptr) : vkinit::attachment_info(velocity_buffer.view, &clear_value));
	}
	else
	{
		for (int i = 0; i < GBUFFER_COUNT - 1; i++)
		{
			rendering_attachment_infos.push_back(late ? vkinit::attachment_info(gbuffers[i].view, nullptr) : vkinit::attachment_info(gbuffers[i].view, &clear_value));
		}
		rendering_attachment_infos.push_back(late ? vkinit::attachment_info(gbuffers[GBUFFER_COUNT - 1].view, nullptr) : vkinit::attachment_info(gbuffers[GBUFFER_COUNT - 1].view, &clear_value2));
	}

	VkRenderingAttachmentInfo depth_attachment = vkinit::depth_attachment_info(depth_image.view);
	depth_attachment.loadOp = late ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
	VkRenderingInfo render_info = vkinit::rendering_info(draw_extent, rendering_attachment_infos.data(), &depth_attachment);
	render_info.colorAttachmentCount = static_cast<uint32_t>(rendering_attachment_infos.size());

	vkCmdBeginRendering(cmd, &render_info);

	VkViewport viewport{};
	viewport.x = 0;
	viewport.y = static_cast<float>(draw_extent.height);
	viewport.width = static_cast<float>(draw_extent.width);
	viewport.height = -static_cast<float>(draw_extent.height);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	vkCmdSetViewport(cmd, 0, 1, &viewport);

	VkRect2D scissor{};
	scissor.offset.x = 0;
	scissor.offset.y = 0;
	scissor.extent.width = draw_extent.width;
	scissor.extent.height = draw_extent.height;
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	// auto depth_bias = 0.f;
	// auto slope_scaled_depth_bias = 0.f;
	//  TODO: refactor to account for different geometry (double-sided or back face culled)
	// vkCmdSetDepthBias(cmd, -depth_bias, 0.0f, -slope_scaled_depth_bias);

	GPUPushConstants pc{};
	pc.object_buffer_address = get_buffer_address(device, render_scene.object_buffer.buffer);
	pc.vertex_buffer_address = get_buffer_address(device, render_scene.vertex_buffer.buffer);
	pc.meshlet_buffer_address = get_buffer_address(device, render_scene.meshlet_buffer.buffer);
	pc.meshlet_indices_buffer_address = get_buffer_address(device, render_scene.meshlet_indices.buffer);
	pc.cluster_indices_address = get_buffer_address(device, render_scene.cluster_indices.buffer);
	pc.material_buffer_address = get_buffer_address(device, render_scene.material_buffer.buffer);
	pc.prefix_sum_buffer = get_buffer_address(device, render_scene.prefix_sum_buffer.buffer);
	auto jitter_count = jitter_offset.size();
	auto current_jitter = jitter_offset[frame_number % jitter_count];
	auto previous_jitter = jitter_offset[(frame_number - 1) % jitter_count];
	pc.screen_size = glm::uvec2(window_extent.width, window_extent.height);
	pc.jitter_offset = glm::vec4(current_jitter, previous_jitter);

	if (!CVAR_RENDER_MESH_SHADERS.get())
	{
		ShaderPass current_pass = post_pass == 0 ? *shader_passes["geometry_vert"] : *shader_passes["geometry_vert_mask"];

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);

		vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GPUPushConstants), &pc);

		vkCmdBindIndexBuffer(cmd, render_scene.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);

		// reuse opaque section for rendering alphaClipped geometry; alphaClipped reserved for shadows
		vkCmdDrawIndexedIndirectCount(cmd, render_scene.draw_indirect_buffer.buffer, 2 * sizeof(uint32_t), render_scene.draw_indirect_buffer.buffer, 0, MAX_MESH_DRAWS, sizeof(VkDrawIndexedIndirectCommand));
	}
	else // mesh shading path
	{
		ShaderPass current_pass{};
		if (visibility_rendering)
		{
			current_pass = post_pass == 0 ? *shader_passes["visibility_mesh"] : *shader_passes["visibility_mesh_mask"];
		}
		else // deferred rendering
		{
			current_pass = post_pass == 0 ? *shader_passes["geometry_mesh"] : *shader_passes["geometry_mesh_mask"];
		}

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);

		vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GPUPushConstants), &pc);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
		vkCmdDrawMeshTasksIndirectEXT(cmd, render_scene.meshlet_dispatch_buffer.buffer, 0, 1, 0);
	}

	vkCmdEndRendering(cmd);
	vkCmdEndQuery(cmd, get_current_frame().query_pool_pipelines, query);
}

void VulkanEngine::render_transparent(VkCommandBuffer cmd, uint32_t query)
{
	vkCmdBeginQuery(cmd, get_current_frame().query_pool_pipelines, query, 0);

	VkRenderingAttachmentInfo depth_attachment = vkinit::depth_attachment_info(depth_image.view);
	depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;

	VkRenderingInfo render_info = vkinit::rendering_info(draw_extent, nullptr, &depth_attachment);

	vkCmdBeginRendering(cmd, &render_info);

	VkViewport viewport{};
	viewport.x = 0;
	viewport.y = static_cast<float>(draw_extent.height);
	viewport.width = static_cast<float>(draw_extent.width);
	viewport.height = -static_cast<float>(draw_extent.height);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	vkCmdSetViewport(cmd, 0, 1, &viewport);

	VkRect2D scissor{};
	scissor.offset.x = 0;
	scissor.offset.y = 0;
	scissor.extent.width = draw_extent.width;
	scissor.extent.height = draw_extent.height;
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	GPUPushConstants pc{};
	pc.object_buffer_address = get_buffer_address(device, render_scene.object_buffer.buffer);
	pc.vertex_buffer_address = get_buffer_address(device, render_scene.vertex_buffer.buffer);
	pc.meshlet_buffer_address = get_buffer_address(device, render_scene.meshlet_buffer.buffer);
	pc.meshlet_indices_buffer_address = get_buffer_address(device, render_scene.meshlet_indices.buffer);
	pc.cluster_indices_address = get_buffer_address(device, render_scene.cluster_indices.buffer);
	pc.material_buffer_address = get_buffer_address(device, render_scene.material_buffer.buffer);
	pc.prefix_sum_buffer = get_buffer_address(device, render_scene.prefix_sum_buffer.buffer);
	// TODO: handle jitter offset for transparency
	pc.screen_size = glm::uvec2(window_extent.width, window_extent.height);

	if (!CVAR_RENDER_MESH_SHADERS.get())
	{
		ShaderPass current_pass = *shader_passes["mlab_vert"];

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 4, 1, &rasterizer_ordered_buf_descriptor, 0, nullptr);

		vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GPUPushConstants), &pc);

		vkCmdBindIndexBuffer(cmd, render_scene.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);

		vkCmdDrawIndexedIndirectCount(cmd, render_scene.draw_indirect_buffer.buffer, 2 * sizeof(uint32_t), render_scene.draw_indirect_buffer.buffer, 0, MAX_MESH_DRAWS, sizeof(VkDrawIndexedIndirectCommand));
	}
	else // mesh shading path
	{
		ShaderPass current_pass = *shader_passes["mlab_mesh"];

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 4, 1, &rasterizer_ordered_buf_descriptor, 0, nullptr);

		vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GPUPushConstants), &pc);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
		vkCmdDrawMeshTasksIndirectEXT(cmd, render_scene.meshlet_dispatch_buffer.buffer, 0, 1, 0);
	}

	vkCmdEndRendering(cmd);
	vkCmdEndQuery(cmd, get_current_frame().query_pool_pipelines, query);
}

void VulkanEngine::render_shadows(VkCommandBuffer cmd, uint32_t cascade_idx, uint32_t query)
{
	vkCmdBeginQuery(cmd, get_current_frame().query_pool_pipelines, query, 0);
	VkRenderingAttachmentInfo depth_attachment = vkinit::depth_attachment_info(cascade_data[cascade_idx].shadow_map.view);

	auto shadow_extent = VkExtent2D{ SHADOW_MAP_SIZE, SHADOW_MAP_SIZE };
	VkRenderingInfo render_info = vkinit::rendering_info(shadow_extent, nullptr, &depth_attachment);

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

	ShadowPushConstants pc{};
	pc.viewproj = cascade_data[cascade_idx].viewproj;
	pc.material_buffer_address = get_buffer_address(device, render_scene.material_buffer.buffer);
	pc.object_buffer_address = get_buffer_address(device, render_scene.object_buffer.buffer);
	pc.vertex_buffer_address = get_buffer_address(device, render_scene.vertex_buffer.buffer);

	{
		ShaderPass current_pass = *shader_passes["depth"];
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);

		vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPushConstants), &pc);

		vkCmdBindIndexBuffer(cmd, render_scene.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

		auto cascade_offset = cascade_idx * (2 * sizeof(uint32_t) + (MAX_OPAQUE_DRAWS + MAX_ALPHACLIP_DRAWS) * sizeof(VkDrawIndexedIndirectCommand));

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
		vkCmdDrawIndexedIndirectCount(cmd, render_scene.draw_indirect_buffer.buffer, 2 * sizeof(uint32_t) + cascade_offset, render_scene.draw_indirect_buffer.buffer, 0 + cascade_offset, MAX_OPAQUE_DRAWS, sizeof(VkDrawIndexedIndirectCommand));

		if (CVAR_RENDER_ALPHACLIP.get())
		{
			current_pass = *shader_passes["depth_mask"];
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);
			vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ShadowPushConstants), &pc);
			vkCmdDrawIndexedIndirectCount(cmd, render_scene.draw_indirect_buffer.buffer, 2 * sizeof(uint32_t) + MAX_OPAQUE_DRAWS * sizeof(VkDrawIndexedIndirectCommand) + cascade_offset, render_scene.draw_indirect_buffer.buffer, sizeof(uint32_t) + cascade_offset, MAX_ALPHACLIP_DRAWS, sizeof(VkDrawIndexedIndirectCommand));
		}
	}

	vkCmdEndRendering(cmd);
	vkCmdEndQuery(cmd, get_current_frame().query_pool_pipelines, query);
}

void VulkanEngine::execute_spd(VkCommandBuffer cmd)
{
    ShaderPass current_pass = *shader_passes["hiz_spd"];
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);

	auto width = next_pow2(draw_extent.width);
	auto height = next_pow2(draw_extent.height);
	auto groupcount_x = get_groupcount(width, 64);
	auto groupcount_y = get_groupcount(height, 64);

	SpdPushConstants pc{};
	pc.spd_counter_buffer = get_buffer_address(device, render_scene.spd_counter_buffer.buffer);
	pc.rcp_resolution = glm::vec2(1.0) / glm::vec2(width, height);
    pc.mips = static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(std::max(depth_pyramid.extent.width, depth_pyramid.extent.height))))) + 1;
    pc.num_wgs = groupcount_x * groupcount_y;
    pc.src_id = texture_cache.get_depth_image();
    pc.dst_id = image_cache.get_depth_pyramid_image();
    pc.sampler_id = DEPTH_REDUCTION_SAMPLER;

    vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SpdPushConstants), &pc);
    vkCmdDispatch(cmd, groupcount_x, groupcount_y, 1);
}

void VulkanEngine::build_depth_pyramid(VkCommandBuffer cmd)
{
	ShaderPass current_pass = *shader_passes["hiz"];
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);

	DepthPyramidPushConstants depth_pc{};

	uint32_t mip_levels = static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(std::max(depth_pyramid.extent.width, depth_pyramid.extent.height))))) + 1;

	for (uint32_t i = 0; i < mip_levels; i++)
	{
		int32_t width = std::max(static_cast<int32_t>(depth_pyramid.extent.width) >> i, 1);
		int32_t height = std::max(static_cast<int32_t>(depth_pyramid.extent.height) >> i, 1);
		depth_pc.image_size = { width, height };
		depth_pc.texture_id = i == 0 ? texture_cache.get_depth_image() : texture_cache.get_depth_pyramid_image();
		depth_pc.image_id = image_cache.get_depth_pyramid_image() + i;
		depth_pc.lod = i == 0 ? 0 : i - 1;

		vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DepthPyramidPushConstants), &depth_pc);
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

	VkCommandBufferBeginInfo cmd_begin_info = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));

	ShaderPass current_pass = *shader_passes["cluster_grid"];
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

	ClusterGridPushConstants pc{};
	pc.inverse_proj = glm::inverse(main_camera.perspective);
	pc.light_cluster_buffer_address = get_buffer_address(device, light_cluster_buffer.buffer);
	pc.screen_size = glm::vec2(window_extent.width, window_extent.height);
	auto cluster_x = ceil(static_cast<float>(window_extent.width) / CLUSTER_X); // # cluster dim
	auto cluster_y = ceil(static_cast<float>(window_extent.height) / CLUSTER_Y); // # cluster dim
	pc.cluster_dim = glm::vec2(cluster_x, cluster_y);
	pc.near = main_camera.near; // reverse-z
	pc.far = main_camera.far;
	pc.depth_slices = CLUSTER_DEPTH_SLICES;

	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ClusterGridPushConstants), &pc);
	vkCmdDispatch(cmd, 1, 1, CLUSTER_DEPTH_SLICES);

	VK_CHECK(vkEndCommandBuffer(cmd));

	VkCommandBufferSubmitInfo cmd_info = vkinit::command_buffer_submit_info(cmd);
	VkSubmitInfo2 submit = vkinit::submit_info(&cmd_info, nullptr, nullptr);

	VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit, imm_fence));
	VK_CHECK(vkWaitForFences(device, 1, &imm_fence, true, 9999999999));
}

void VulkanEngine::execute_light_culling(VkCommandBuffer cmd)
{
	ShaderPass current_pass = *shader_passes["light_culling"];
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

	LightCullingPushConstants pc{};

	pc.view = scene_data.view;
	pc.light_rot = scene_data.light_rot;

	pc.light_cluster_buffer_address = get_buffer_address(device, light_cluster_buffer.buffer);
	pc.light_buffer_address = get_buffer_address(device, light_buffer.buffer);
	pc.light_index_buffer_address = get_buffer_address(device, light_index_buffer.buffer);
	pc.light_grid_buffer_address = get_buffer_address(device, light_grid_buffer.buffer);
	pc.light_count_buffer_address = get_buffer_address(device, light_count_buffer.buffer);

	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LightCullingPushConstants), &pc);
	vkCmdDispatch(cmd, 1, 1, CLUSTER_DEPTH_SLICES / CLUSTER_Z);
}

void VulkanEngine::execute_shading(VkCommandBuffer cmd)
{
	ShaderPass current_pass{};
	bool visibility_rendering = CVAR_RENDER_VBUFFER.get() && CVAR_RENDER_MESH_SHADERS.get();
	if (visibility_rendering)
	{
		current_pass = *shader_passes["resolve_vbuffer"];
    	// if (CVAR_RENDER_RT.get())
    	//     current_pass = *shader_passes["ray_tracing"];
	}
	else
	{
		current_pass = *shader_passes["resolve_gbuffer"];
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 4, 1, &as_descriptor, 0, nullptr);

	DeferredPushConstants pc{};

	auto cluster_x = ceil(static_cast<float>(window_extent.width) / CLUSTER_X); // # cluster dim
	auto cluster_y = ceil(static_cast<float>(window_extent.height) / CLUSTER_Y); // # cluster dim
	pc.cluster_size = glm::vec4(cluster_x, cluster_y, CLUSTER_DEPTH_SLICES, 0.0);
	pc.screen_size = glm::vec2(window_extent.width, window_extent.height);

	pc.light_buffer_address = get_buffer_address(device, light_buffer.buffer);
	pc.light_index_buffer_address = get_buffer_address(device, light_index_buffer.buffer);
	pc.light_grid_buffer_address = get_buffer_address(device, light_grid_buffer.buffer);
	pc.oit_buffer_address = get_buffer_address(device, render_scene.oit_buffer.buffer);
	pc.meshlet_indices_address = get_buffer_address(device, render_scene.meshlet_indices.buffer);
	pc.meshlet_buffer_address = get_buffer_address(device, render_scene.meshlet_buffer.buffer);
	pc.vertex_buffer_address = get_buffer_address(device, render_scene.vertex_buffer.buffer);
	pc.object_buffer_address = get_buffer_address(device, render_scene.object_buffer.buffer);
	pc.material_buffer_address = get_buffer_address(device, render_scene.material_buffer.buffer);
	pc.index_buffer_address = get_buffer_address(device, render_scene.index_buffer.buffer);
	pc.mesh_buffer_address = get_buffer_address(device, render_scene.mesh_buffer.buffer);
	pc.sh_buffer_address = get_buffer_address(device, render_scene.sh_buffer.buffer);

	pc.depth_id = texture_cache.get_depth_image();
	pc.gbuffer_id = visibility_rendering ? texture_cache.get_visibility_buffer() : texture_cache.get_first_gbuffer();
	pc.shadow_id = texture_cache.get_shadowmap();
	pc.light_culling = CVAR_RENDER_POINT_LIGHTS.get();
	pc.near = main_camera.near;

	const float ratio = main_camera.far / main_camera.near;
	pc.scale = static_cast<float>(CLUSTER_DEPTH_SLICES) / std::log(ratio);
	pc.bias = static_cast<float>(CLUSTER_DEPTH_SLICES) * std::log(main_camera.near) / std::log(ratio);
	pc.resolve_transparent = CVAR_RENDER_TRANSPARENT.get();
	pc.shadows = CVAR_RENDER_SHADOWS.get();
	pc.shadows_rt = CVAR_RENDER_SHADOWS_RT.get();
	pc.max_prefiltered_lod = static_cast<float>(std::floor(std::log2(static_cast<float>(std::max(prefiltered_envmap.extent.width, prefiltered_envmap.extent.height))))) + 1;
	pc.metallic = CVAR_PBR_METALLIC.get();
	pc.roughness = CVAR_PBR_ROUGHNESS.get();
	pc.debug = CVAR_DEBUG_TEXTURES.get();

	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DeferredPushConstants), &pc);
	auto groupcount_x = get_groupcount(draw_extent.width, 8);
	auto groupcount_y = get_groupcount(draw_extent.height, 8);
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
    VkDeviceAddress vb_address = get_buffer_address(device, render_scene.vertex_buffer.buffer);
    VkDeviceAddress ib_address = get_buffer_address(device, render_scene.index_buffer.buffer);

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
    scratch_buffer = create_buffer(allocator, total_scratch_size, 0, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    fmt::println("blas_buffer: {}", size_in_bytes(total_as_size));
    fmt::println("scratch_buffer: {}", size_in_bytes(total_scratch_size));

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

    immediate_submit([&](VkCommandBuffer cmd){
        vkCmdBuildAccelerationStructuresKHR(cmd, static_cast<uint32_t>(build_infos.size()), build_infos.data(), build_ranges_ptrs.data());
    });

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
        RenderObject obj = render_scene.renderables[i];

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

    tlas_instance_buffer = upload_buffer(this, allocator, instances.data(), instances.size() * sizeof(VkAccelerationStructureInstanceKHR), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

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
    fmt::println("tlas scratch_buffer: {}", size_in_bytes(build_size.buildScratchSize));
    fmt::println("tlas instance buffer: {}", size_in_bytes(tlas_instance_buffer.info.size));
    fmt::println("tlas buffer: {}", size_in_bytes(build_size.accelerationStructureSize));

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

    immediate_submit([&](VkCommandBuffer cmd){
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &build_range_ptr);
    });

    destroy_buffer(allocator, scratch_buffer);
    destroy_buffer(allocator, tlas_instance_buffer);

    main_deletion_queue.push_function([&, blas_handles](){
        destroy_buffer(allocator, blas_buffer);
        destroy_buffer(allocator, tlas_buffer);

        for (auto& blas : blas_handles)
        {
            vkDestroyAccelerationStructureKHR(device, blas, nullptr);
        }

        vkDestroyAccelerationStructureKHR(device, tlas_as, nullptr);
    });
}
