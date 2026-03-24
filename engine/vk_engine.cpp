#include "common.h"
#include "config.h"
#include "vk_math.h"
#include "vk_engine.h"
#include "cvars.h"
#include "vk_descriptors.h"
#include "resources.h"
#include "vk_initializers.h"
#include "vk_loader.h"
#include "vk_pipelines.h"
#include "vk_scene.h"
#include "cache.h"

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

VulkanEngine* loaded_engine{};

VulkanEngine& VulkanEngine::get() { return *loaded_engine; }

#ifdef NDEBUG
constexpr bool USE_VALIDATION_LAYERS = false;
#else
constexpr bool USE_VALIDATION_LAYERS = true;
#endif

// #define SINGLE // uncomment if loading a proper scene

AutoCVar_Int CVAR_RENDER_VBUFFER{ "render.vbuffer", "Vbuffer path", 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_RENDER_MESH_SHADERS{ "render.mesh_shaders", "Mesh shaders path", 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_RENDER_ALPHACLIP{ "render.alphaclip", "Alphaclip", 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_RENDER_TRANSPARENT{ "render.transparent", "Transparent", 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_RENDER_POINT_LIGHTS{ "render.point_lights", "Point lights", 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_RENDER_OCCLUSION_CULL{ "render.occlusion_cull", "Occlusion culling", 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_RENDER_LOD{ "render.lod", "LODs", 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_RENDER_SHADOWS{ "render.shadows", "Shadows", 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_RENDER_TAA{ "render.taa", "TAA", 0, CVarFlags::EditCheckbox | CVarFlags::EditHide };

AutoCVar_Int CVAR_SHADOWS_PCF{ "shadows.pcf", "PCF", 1, CVarFlags::EditCheckbox };
AutoCVar_Float CVAR_SHADOWS_CASCADE_SPLIT{ "shadows.cascade_split", "Cascades log factor", 0.95f, CVarFlags::EditDragFloat, 0.f, 1.f, 0.005f };
AutoCVar_Int CVAR_SHADOWS_DISTANCE{ "shadows.distance", "Shadow draw distance", 48, CVarFlags::EditSliderInt, 20, 200, 5 };
AutoCVar_Int CVAR_SHADOWS_CASCADE_SELECTION{ "shadows.cascade_selection", "Map based cascade selection", 1, CVarFlags::EditCheckbox };

AutoCVar_Int CVAR_DEBUG_TEXTURES{ "debug.textures", "Debug textures", 0, CVarFlags::EditSliderInt, 0, DEBUG_COUNT, 1 };

AutoCVar_Int CVAR_MISC_DRAW_DISTANCE{ "misc.draw_distance", "Draw distance", 1000, CVarFlags::EditSliderInt, 100, 1000, 100 };
AutoCVar_Int CVAR_MISC_AUTOEXPOSURE{ "misc.autoexposure", "Autoexposure", 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_MISC_TONEMAP{ "misc.tonemap", "Tonemapping", 0, CVarFlags::EditSliderInt, 0, 1, 1 };
AutoCVar_Int CVAR_MISC_FREEZE_CAMERA{ "misc.freeze_camera", "Freeze camera", 0, CVarFlags::EditCheckbox };

AutoCVar_Int CVAR_TAA_VARIANCE_CLIP{ "taa.variance_clip", "Variance clipping", 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_TAA_CATMULL_ROM{ "taa.catmull_rom", "Catmull filter", 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_TAA_MITCHELL{ "taa.mitchell", "Mitchell filter", 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_TAA_YCOCG{ "taa.ycogy", "YCoCg", 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_TAA_DEPTH_DILATION{ "taa.depth_dilation", "Depth dilation", 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_TAA_LUMINANCE_WEIGHING{ "taa.luminance_weighing", "Luminance weighing", 1, CVarFlags::EditCheckbox };

AutoCVar_Float CVAR_PBR_METALLIC{ "pbr.metallic", "Metallic", 0.0f, CVarFlags::EditDragFloat, 0.f, 1.f, 0.05f };
AutoCVar_Float CVAR_PBR_ROUGHNESS{ "pbr.roughness", "Roughness", 0.5f, CVarFlags::EditDragFloat, 0.f, 1.f, 0.05f };

namespace
{
	uint32_t nearest_pow2(uint32_t extent)
	{
		return 1 << static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(extent))));
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
}

void VulkanEngine::init(const std::string& file_path)
{
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

	init_pipelines();

	main_camera.position = glm::vec3(0, 0, 5);
	main_camera.near = static_cast<float>(CVAR_MISC_DRAW_DISTANCE.get());
	main_camera.far = 0.5f;
	main_camera.fov = 70.0f;
	// TODO: refactor if window resize
	main_camera.set_perspective_matrix(glm::radians(main_camera.fov), static_cast<float>(draw_extent.width) / static_cast<float>(draw_extent.height), main_camera.far);

	init_default_data();

	init_renderables(file_path);

	init_bindless();

	init_imgui();

	upload_buffers();

	build_cluster_grid(); // TODO: support draw distance change
	init_gi();

	// first frame transitions to avoid validation errors
	{
		immediate_submit([&](VkCommandBuffer cmd)
			{
				// previous frame's output buffer
				vkutil::transition_image(cmd, accumulation_buffers[(frame_number + 1) % 2].image,
					VK_IMAGE_LAYOUT_UNDEFINED,
					VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					0, 0, 0, 0
				);

				vkCmdFillBuffer(cmd, render_scene.luminance_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
				// TODO: figure out a good default for luminance avg
				vkCmdFillBuffer(cmd, render_scene.luminance_avg_buffer.buffer, 0, VK_WHOLE_SIZE, 0x40000000); // 0x40000000 = 2.0f

			vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
			}
		);
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
			vkDestroySemaphore(device, frame.swapchain_semaphore, nullptr);
			vkDestroySemaphore(device, frame.render_semaphore, nullptr);

			destroy_buffer(allocator, frame.scene_buffer);

			frame.deletion_queue.flush();

			vkDestroyQueryPool(device, frame.query_pool_timestamps, nullptr);
			vkDestroyQueryPool(device, frame.query_pool_pipelines, nullptr);
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
		destroy_buffer(allocator, render_scene.meshtask_indirect_buffer);
		destroy_buffer(allocator, render_scene.meshlet_vis_buffer);
		destroy_buffer(allocator, render_scene.cluster_count_buffer);
		destroy_buffer(allocator, render_scene.cluster_indices);

		destroy_buffer(allocator, render_scene.oit_buffer);
		destroy_buffer(allocator, render_scene.indices_buffer);

		destroy_buffer(allocator, render_scene.sh_buffer);
		destroy_buffer(allocator, render_scene.luminance_buffer);
		destroy_buffer(allocator, render_scene.luminance_avg_buffer);

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

void VulkanEngine::init_gi()
{
	VK_CHECK(vkResetFences(device, 1, &imm_fence));
	VK_CHECK(vkResetCommandBuffer(imm_command_buffer, 0));
	VkCommandBufferBeginInfo cmd_begin_info = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	VK_CHECK(vkBeginCommandBuffer(imm_command_buffer, &cmd_begin_info));

	// spherical map -> cubemap
	{
		vkutil::transition_image(imm_command_buffer, hdri_cubemap.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0, VK_ACCESS_2_SHADER_WRITE_BIT);

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

		vkutil::transition_image(imm_command_buffer, hdri_cubemap.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

		vkutil::generate_mipmaps(imm_command_buffer, hdri_cubemap.image, VkExtent2D(hdri_cubemap.extent.width, hdri_cubemap.extent.height), 6);

		vkutil::transition_image(imm_command_buffer, hdri_cubemap.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, VK_ACCESS_2_SHADER_READ_BIT);
	}

	// compute SH coefficients
	{
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

	// compute irradiance cubemap for SH reference
	{
		vkutil::transition_image(imm_command_buffer, irradiance_cubemap.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0, VK_ACCESS_2_SHADER_WRITE_BIT);

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

		vkutil::transition_image(imm_command_buffer, irradiance_cubemap.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
	}

	// prefiltered envmap
	uint32_t brdf_id{};
	{
		vkutil::transition_image(imm_command_buffer, prefiltered_envmap.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0, VK_ACCESS_2_SHADER_WRITE_BIT);

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
		vkutil::transition_image(imm_command_buffer, prefiltered_envmap.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
	}

	// brdf lut
	{
		vkutil::transition_image(imm_command_buffer, brdf_lut.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0, VK_ACCESS_2_SHADER_WRITE_BIT);

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

		vkutil::transition_image(imm_command_buffer, brdf_lut.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
	}

	// for SH coefficients buffer
	vkutil::transition_buffer(imm_command_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

	VK_CHECK(vkEndCommandBuffer(imm_command_buffer));
	VkCommandBufferSubmitInfo cmd_info = vkinit::command_buffer_submit_info(imm_command_buffer);
	VkSubmitInfo2 submit = vkinit::submit_info(&cmd_info, nullptr, nullptr);
	VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit, imm_fence));
	VK_CHECK(vkWaitForFences(device, 1, &imm_fence, true, 9999999999));
}

void VulkanEngine::draw()
{
	// clang-format off
	{
		VK_CHECK(vkWaitForFences(device, 1, &get_current_frame().render_fence, true, 1000000000));
	}

	VK_CHECK(vkResetFences(device, 1, &get_current_frame().render_fence));

	get_current_frame().deletion_queue.flush();

	bool visibility_rendering = CVAR_RENDER_VBUFFER.get() && CVAR_RENDER_MESH_SHADERS.get();
	buffer_barriers.clear();
	image_barriers.clear();

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
		VK_CHECK(vkAcquireNextImageKHR(device, swapchain, 1000000000, get_current_frame().swapchain_semaphore, nullptr, &swapchain_image_idx));
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

	VK_CHECK(vkResetCommandBuffer(cmd, 0));

	VkCommandBufferBeginInfo cmd_begin_info = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));

	// two-pass mesh/cluster occlusion culling
	{
		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

		vkCmdFillBuffer(cmd, render_scene.dispatch_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, render_scene.cluster_count_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, render_scene.draw_indirect_buffer.buffer, 0, VK_WHOLE_SIZE, 0);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 0);
		execute_compute_cull(cmd, render_scene.opaque_pass, forward_mesh_cull_data, false, 0);
		if (CVAR_RENDER_MESH_SHADERS.get())
		{
			vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);

			execute_compute_cull(cmd, render_scene.opaque_pass, forward_cluster_cull_data, render_scene.dispatch_buffer.buffer, 0, false, 0);
		}
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 1);

		buffer_barriers.emplace_back(buffer_barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT));
		image_barriers.emplace_back(image_barrier(depth_image.image,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
				VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, // TODO: do we need fragment shader bit? cc deferred.frag
				VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
				VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, // TODO: do we need shader sample? cc deferred.frag
				VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
				VK_IMAGE_ASPECT_DEPTH_BIT
			)
		);

		// TODO: we don't need to transition all
		if (visibility_rendering)
		{
			image_barriers.emplace_back(image_barrier(
					visibility_buffer.image,
					VK_IMAGE_LAYOUT_UNDEFINED,
					VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
					VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
					VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
					VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
					VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
				)
			);
			image_barriers.emplace_back(image_barrier(
					velocity_buffer.image,
					VK_IMAGE_LAYOUT_UNDEFINED,
					VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
					VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
					VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
					VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
					VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
				)
			);
		}
		else
		{
			for (int i = 0; i < GBUFFER_COUNT; i++)
			{
				image_barriers.emplace_back(image_barrier(
						gbuffers[i].image,
						VK_IMAGE_LAYOUT_UNDEFINED,
						VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
						VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
						VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
						VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
						VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
					)
				);
			}
		}

		pipeline_barrier(cmd, buffer_barriers.data(), buffer_barriers.size(), image_barriers.data(), image_barriers.size());
		buffer_barriers.clear();
		image_barriers.clear();

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 2);
		render(cmd, false, 0, 0);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 3);

		if (!freeze_camera)
		{
			image_barriers.emplace_back(image_barrier(
					depth_image.image,
					VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
					VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
					VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
					VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
					VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
					VK_IMAGE_ASPECT_DEPTH_BIT
				)
			);

			image_barriers.emplace_back(image_barrier(
					depth_pyramid.image,
					VK_IMAGE_LAYOUT_UNDEFINED,
					VK_IMAGE_LAYOUT_GENERAL,
					VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, // debugging in fragment shader ????????????????
					VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
					VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
					VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
					VK_IMAGE_ASPECT_COLOR_BIT
				)
			);

			pipeline_barrier(cmd, nullptr, 0, image_barriers.data(), image_barriers.size());
			image_barriers.clear();

			build_depth_pyramid(cmd);

			// next use in compute occlusion cull; if freeze_camera, we will always be in the right image layout
			vkutil::transition_image(
				cmd,
				depth_pyramid.image,
				VK_IMAGE_LAYOUT_GENERAL,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
				VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
				VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
				VK_IMAGE_ASPECT_COLOR_BIT
			);
		}

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

		vkCmdFillBuffer(cmd, render_scene.dispatch_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, render_scene.cluster_count_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, render_scene.draw_indirect_buffer.buffer, 0, VK_WHOLE_SIZE, 0);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 4);
		execute_compute_cull(cmd, render_scene.opaque_pass, forward_mesh_cull_data, true, 0);

		if (CVAR_RENDER_MESH_SHADERS.get())
		{
			vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);

			execute_compute_cull(cmd, render_scene.opaque_pass, forward_cluster_cull_data, render_scene.dispatch_buffer.buffer, 0, true, 0);
		}

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 5);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

		if (!freeze_camera)
		{
			// last use was for building hi-z; it never leaves depth attachment layout if freeze_camera == true
			vkutil::transition_image(
			    cmd,
			    depth_image.image,
			    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
			    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
			    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
			    VK_IMAGE_ASPECT_DEPTH_BIT
			);
		}

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 6);
		render(cmd, true, 0, 1);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 7);
	}

	// third pass - masked geometry; we are still using the same hi-z for culling here. we could build an updated hi-z.
	if (CVAR_RENDER_ALPHACLIP.get())
	{
		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

		vkCmdFillBuffer(cmd, render_scene.dispatch_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, render_scene.cluster_count_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, render_scene.draw_indirect_buffer.buffer, 0, VK_WHOLE_SIZE, 0);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 8);
		execute_compute_cull(cmd, render_scene.mask_pass, forward_mesh_cull_data, true, 1);

		if (CVAR_RENDER_MESH_SHADERS.get())
		{
			vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);

			execute_compute_cull(cmd, render_scene.mask_pass, forward_cluster_cull_data, render_scene.dispatch_buffer.buffer, 0, true, 1);
		}

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 9);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 10);
		render(cmd, true, 1, 2);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 11);
	}
	else
	{
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 8);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 9);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 10);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 11);
		vkCmdBeginQuery(cmd, get_current_frame().query_pool_pipelines, 2, 0);
		vkCmdEndQuery(cmd, get_current_frame().query_pool_pipelines, 2);
	}

	// fourth pass - transparent geometry/MLAB
	if (CVAR_RENDER_TRANSPARENT.get())
	{
		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

		vkCmdFillBuffer(cmd, render_scene.dispatch_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, render_scene.cluster_count_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, render_scene.draw_indirect_buffer.buffer, 0, VK_WHOLE_SIZE, 0);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 16);
		execute_compute_cull(cmd, render_scene.transparent_pass, forward_mesh_cull_data, true, 2);

		if (CVAR_RENDER_MESH_SHADERS.get())
		{
			vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);

			execute_compute_cull(cmd, render_scene.transparent_pass, forward_cluster_cull_data, render_scene.dispatch_buffer.buffer, 0, true, 2);
		}

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 17);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

		// is this necessary - ensure depth image in use is final; skipping this worked ok, not sure if pixel interlock interference compensates for it
		 vkutil::transition_image(
			cmd,
			depth_image.image,
			VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
			VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
			VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
			VK_IMAGE_ASPECT_DEPTH_BIT
		);

		// TODO: barrier for OIT buffer from last frame? this is likely unnecessary
		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 18);
		// note: this reuses hi-z from first pass. we could technically update and rebuild hi-z after second pass.
		// TODO: perform some sort of sorting or compaction to guarantee insertion order, which is what i suspect to be causing flickering only for overflowed pixels at the moment
		render_transparent(cmd, 3);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 19);
	}
	else
	{
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 16);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 17);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 18);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 19);
		vkCmdBeginQuery(cmd, get_current_frame().query_pool_pipelines, 3, 0);
		vkCmdEndQuery(cmd, get_current_frame().query_pool_pipelines, 3);
	}

	// light culling pass
	if (CVAR_RENDER_POINT_LIGHTS.get())
	{
		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_SHADER_READ_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

		vkCmdFillBuffer(cmd, light_count_buffer.buffer, 0, VK_WHOLE_SIZE, 0);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 12);
		execute_light_culling(cmd);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 13);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
	}
	else
	{
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 12);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 13);
	}

	// shadow pass
	if (CVAR_RENDER_SHADOWS.get())
	{
		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

		vkCmdFillBuffer(cmd, render_scene.dispatch_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		// vkCmdFillBuffer(cmd, render_scene.cluster_count_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, render_scene.draw_indirect_buffer.buffer, 0, VK_WHOLE_SIZE, 0);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 20);
		execute_shadow_cull(cmd);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 21);

		buffer_barriers.emplace_back(buffer_barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, // | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
								  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT));

		for (auto& i : cascade_data)
		{
			image_barriers.emplace_back(image_barrier(i.shadow_map.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT, VK_IMAGE_ASPECT_DEPTH_BIT));
		}
		pipeline_barrier(cmd, buffer_barriers.data(), buffer_barriers.size(), image_barriers.data(), image_barriers.size());
		buffer_barriers.clear();
		image_barriers.clear();

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 22);
		uint32_t query_index = 4;
		for (size_t i = 0; i < cascade_data.size(); i++)
		{
			render_shadows(cmd, static_cast<uint32_t>(i), query_index);
			query_index++;
		}
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 23);

		for (auto& i : cascade_data)
		{
			image_barriers.emplace_back(image_barrier(i.shadow_map.image, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_ASPECT_DEPTH_BIT));
		}
		pipeline_barrier(cmd, nullptr, 0, image_barriers.data(), image_barriers.size());
		image_barriers.clear();
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

	// deferred lighting pass
	{
		if (visibility_rendering)
		{
			image_barriers.emplace_back(image_barrier(
					visibility_buffer.image,
					VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
					VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
					VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
					VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
					VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
				)
			);
			// could technically transition this only if TAA is enabled, but we transition anyway
			image_barriers.emplace_back(image_barrier(
					velocity_buffer.image,
					VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
					VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
					VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
					VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
					VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
				)
			);
		}
		else
		{
			for (int i = 0; i < GBUFFER_COUNT; i++)
			{
				image_barriers.emplace_back(image_barrier(
						gbuffers[i].image,
						VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
						VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
						VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
						VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
						VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
						VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
					)
				);
			}
		}

		image_barriers.emplace_back(image_barrier(
				depth_image.image,
				VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
				VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
				VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
				VK_IMAGE_ASPECT_DEPTH_BIT
			)
		);

		// obsolete: if TAA, this is used in resolve attachment; else as deferred lighting attachment
		// image_barriers.emplace_back(image_barrier(
		// 		draw_image.image,
		// 		VK_IMAGE_LAYOUT_UNDEFINED, // from prev frame's blit to swapchain
		// 		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		// 		VK_PIPELINE_STAGE_2_BLIT_BIT,
		// 		VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		// 		VK_ACCESS_2_TRANSFER_READ_BIT,
		// 		VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
		// 	)
		// );

		image_barriers.emplace_back(image_barrier(
				draw_image.image,
				VK_IMAGE_LAYOUT_UNDEFINED, // from prev frame's blit to swapchain
				VK_IMAGE_LAYOUT_GENERAL,
				VK_PIPELINE_STAGE_2_BLIT_BIT,
				VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
				VK_ACCESS_2_TRANSFER_READ_BIT,
				VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
			)
		);

		if (CVAR_RENDER_TAA.get())
		{
			// previous frame history buffer
			image_barriers.emplace_back(image_barrier(
					accumulation_buffers[frame_number % 2].image,
					VK_IMAGE_LAYOUT_UNDEFINED,
					VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
					VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
					VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
					VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
					VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
				)
			);
		}

		pipeline_barrier(cmd, nullptr, 0, image_barriers.data(), image_barriers.size());
		image_barriers.clear();

		// TODO: likely obsolete, revisit when fixing TAA
		VkImageView view = CVAR_RENDER_TAA.get() ? accumulation_buffers[frame_number % 2].view : draw_image.view;
		if (first_frame)
		{
			first_frame = false;
			view = draw_image.view;
		}

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 14);
		resolve_shading(cmd);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 15);
	}

	if (CVAR_RENDER_TAA.get())
	{
		// current frame output buffer
		image_barriers.emplace_back(image_barrier(
				accumulation_buffers[frame_number % 2].image,
				VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
				VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
				VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
				VK_ACCESS_2_SHADER_READ_BIT
			)
		);

		// previous frame output buffer
		image_barriers.emplace_back(image_barrier(
				accumulation_buffers[(frame_number + 1) % 2].image,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_2_BLIT_BIT,
				VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
				VK_ACCESS_2_TRANSFER_WRITE_BIT,
				VK_ACCESS_2_SHADER_READ_BIT
			)
		);

		pipeline_barrier(cmd, nullptr, 0, image_barriers.data(), image_barriers.size());
		image_barriers.clear();

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 24);
		if (!first_frame)
			execute_taa_resolve(cmd, draw_image.view);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 25);

		// copy resolve to current output buffer
		image_barriers.emplace_back(image_barrier(
				accumulation_buffers[frame_number % 2].image,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
				VK_PIPELINE_STAGE_2_BLIT_BIT,
				VK_ACCESS_2_SHADER_READ_BIT,
				VK_ACCESS_2_TRANSFER_WRITE_BIT
			)
		);
	}
	else
	{
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 24);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 25);
	}

	// TODO: verify that luminance histogram is built post TAA resolve
	image_barriers.emplace_back(image_barrier(
			draw_image.image,
			VK_IMAGE_LAYOUT_GENERAL,
			VK_IMAGE_LAYOUT_GENERAL,
			VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
			VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
		)
	);

	// last frame luminance avg & luminance buffer
	if (CVAR_MISC_AUTOEXPOSURE.get())
	{
		buffer_barriers.emplace_back(buffer_barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT));
	}

	pipeline_barrier(cmd, buffer_barriers.data(), buffer_barriers.size(), image_barriers.data(), image_barriers.size());
	buffer_barriers.clear();
	image_barriers.clear();

	// TODO: frame n applies exposure from frame n's luminance avg instead of previous frame's, correct this in the future?
	// build luminance histogram & luminance avg
	if (CVAR_MISC_AUTOEXPOSURE.get())
	{
		ShaderPass current_pass = *shader_passes["luminance_histogram"];
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);

		LuminanceBinsPC pc{};
		pc.luminance_buffer = get_buffer_address(device, render_scene.luminance_buffer.buffer);
		pc.luminance_avg_buffer = get_buffer_address(device, render_scene.luminance_avg_buffer.buffer);
		pc.screen_size = glm::vec2(draw_image.extent.width, draw_image.extent.height);
		pc.image_id = image_cache.get_draw_image();
		pc.min_log_luminance = -10.0f;
		float max_log_luminance = 2.0f;
		pc.one_over_log_luminance_range = 1.0f / (max_log_luminance - pc.min_log_luminance);
		pc.pixel_count = draw_image.extent.width * draw_image.extent.height;
		pc.tau = 2.f;
		pc.delta_time = static_cast<float>(stats.deltatime);

		vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LuminanceBinsPC), &pc);
		auto groupcount_x = get_groupcount(draw_image.extent.width, LUMINANCE_BINS);
		auto groupcount_y = get_groupcount(draw_image.extent.height, LUMINANCE_BINS);
		vkCmdDispatch(cmd, groupcount_x, groupcount_y, 1);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
		);

		current_pass = *shader_passes["luminance_avg"];
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
		vkCmdDispatch(cmd, 1, 1, 1);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT
		);
	}

	// tonemapping pass
	if (CVAR_DEBUG_TEXTURES.get() == 0)
	{
		ShaderPass current_pass = *shader_passes["tonemap"];
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);

		TonemapPC pc{};
		pc.luminance_avg_buffer = get_buffer_address(device, render_scene.luminance_avg_buffer.buffer);
		pc.screen_size = glm::vec2(draw_image.extent.width, draw_image.extent.height);
		pc.image_id = image_cache.get_draw_image();
		pc.autoexposure = CVAR_MISC_AUTOEXPOSURE.get();
		pc.tonemap_func = CVAR_MISC_TONEMAP.get();

		vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TonemapPC), &pc);
		auto groupcount_x = get_groupcount(draw_image.extent.width, WARP_SIZE);
		auto groupcount_y = get_groupcount(draw_image.extent.height, WARP_SIZE);
		vkCmdDispatch(cmd, groupcount_x, groupcount_y, 1);
	}

	// OBSOLETE ----------- TAA or not, this is a color attachment prior to using as blit src
	// NEW ---------------- draw_image from luminance sampling -> tonemap -> blit
	image_barriers.emplace_back(image_barrier(
		    draw_image.image,
		    VK_IMAGE_LAYOUT_GENERAL,
		    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		    VK_PIPELINE_STAGE_2_BLIT_BIT,
		    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		    VK_ACCESS_2_TRANSFER_READ_BIT
	    )
	);

	image_barriers.emplace_back(image_barrier(
		    swapchain_images[swapchain_image_idx],
		    VK_IMAGE_LAYOUT_UNDEFINED,
		    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		    0,
		    VK_PIPELINE_STAGE_2_BLIT_BIT,
		    0,
		    VK_ACCESS_2_TRANSFER_WRITE_BIT
		)
	);

	pipeline_barrier(cmd, nullptr, 0, image_barriers.data(), image_barriers.size());
	image_barriers.clear();

	if (CVAR_RENDER_TAA.get())
	{
		vkutil::copy_image(cmd, draw_image.image, accumulation_buffers[frame_number % 2].image, draw_extent, draw_extent);
	}
	vkutil::copy_image(cmd, draw_image.image, swapchain_images[swapchain_image_idx], draw_extent, swapchain_extent);

	vkutil::transition_image(
	    cmd,
	    swapchain_images[swapchain_image_idx],
	    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	    VK_PIPELINE_STAGE_2_BLIT_BIT,
	    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
	    VK_ACCESS_2_TRANSFER_WRITE_BIT,
	    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
	);

	{
		if (render_imgui)
			draw_imgui(cmd, swapchain_image_views[swapchain_image_idx]);
	}

	vkutil::transition_image(
	    cmd,
	    swapchain_images[swapchain_image_idx],
	    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
	    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
	    VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
	    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
	    0
	);
	// TracyVkCollect(tracy_ctx, get_current_frame().main_command_buffer);
	VK_CHECK(vkEndCommandBuffer(cmd));

	VkCommandBufferSubmitInfo cmd_info = vkinit::command_buffer_submit_info(cmd);

	VkSemaphoreSubmitInfo wait_info = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, get_current_frame().swapchain_semaphore);
	VkSemaphoreSubmitInfo submit_info = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, get_current_frame().render_semaphore); // all graphics bit?

	VkSubmitInfo2 submit = vkinit::submit_info(&cmd_info, &submit_info, &wait_info);

	{
		VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit, get_current_frame().render_fence));
	}

	VkPresentInfoKHR present_info{};
	present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	present_info.waitSemaphoreCount = 1;
	present_info.pWaitSemaphores = &get_current_frame().render_semaphore; // TODO: use swapchain image count number of semaphores instead of frame in flight? see: vulkanised 2026 frames in flight talk
	present_info.swapchainCount = 1;
	present_info.pSwapchains = &swapchain;
	present_info.pImageIndices = &swapchain_image_idx;

	VK_CHECK(vkQueuePresentKHR(graphics_queue, &present_info));
	// FrameMark;
	frame_number++;
	// clang-format on;
}

void VulkanEngine::run()
{
	SDL_Event e;
	bool bQuit = false;

	auto last_frame = std::chrono::system_clock::now();

	while (!bQuit)
	{
		auto start = std::chrono::system_clock::now();
		auto deltatime = std::chrono::duration_cast<std::chrono::microseconds>(start - last_frame);
		stats.deltatime = static_cast<float>(deltatime.count()) / 1000000.0f; // microseconds to seconds
		stats.frame_avg = stats.frame_avg * 0.95 + stats.deltatime * 0.05;
		last_frame = start;

		// Handle events on queue
		while (SDL_PollEvent(&e) != 0)
		{
			if (e.type == SDL_EVENT_QUIT)
				bQuit = true;

			if (e.type == SDL_EVENT_WINDOW_MINIMIZED)
				stop_rendering = true;
			if (e.type == SDL_EVENT_WINDOW_RESTORED)
				stop_rendering = false;

			if (e.type == SDL_EVENT_KEY_DOWN)
			{
				if (e.key.repeat == 0 && e.key.key == SDLK_SPACE)
				{
					stop_movement = !stop_movement;
					stop_movement ? SDL_SetWindowRelativeMouseMode(window, false) : SDL_SetWindowRelativeMouseMode(window, true);
				}

				// toggle IMGUI render
				if (e.key.repeat == 0 && e.key.key == SDLK_R)
				{
					render_imgui = !render_imgui;
				}
				// TAA
				if (e.key.repeat == 0 && e.key.key == SDLK_Z)
				{
					if (CVAR_TAA_VARIANCE_CLIP.get() == 1)
						CVAR_TAA_VARIANCE_CLIP.set(0);
					else
						CVAR_TAA_VARIANCE_CLIP.set(1);
				}
				if (e.key.repeat == 0 && e.key.key == SDLK_X)
				{
					if (CVAR_TAA_CATMULL_ROM.get() == 1)
						CVAR_TAA_CATMULL_ROM.set(0);
					else
						CVAR_TAA_CATMULL_ROM.set(1);
				}
				if (e.key.repeat == 0 && e.key.key == SDLK_C)
				{
					if (CVAR_TAA_YCOCG.get() == 1)
						CVAR_TAA_YCOCG.set(0);
					else
						CVAR_TAA_YCOCG.set(1);
				}
				if (e.key.repeat == 0 && e.key.key == SDLK_V)
				{
					if (CVAR_TAA_LUMINANCE_WEIGHING.get() == 1)
						CVAR_TAA_LUMINANCE_WEIGHING.set(0);
					else
						CVAR_TAA_LUMINANCE_WEIGHING.set(1);
				}
				if (e.key.repeat == 0 && e.key.key == SDLK_B)
				{
					if (CVAR_TAA_DEPTH_DILATION.get() == 1)
						CVAR_TAA_DEPTH_DILATION.set(0);
					else
						CVAR_TAA_DEPTH_DILATION.set(1);
				}
				if (e.key.repeat == 0 && e.key.key == SDLK_G)
				{
					if (CVAR_MISC_AUTOEXPOSURE.get() == 1)
						CVAR_MISC_AUTOEXPOSURE.set(0);
					else
						CVAR_MISC_AUTOEXPOSURE.set(1);
				}
				if (e.key.repeat == 0 && e.key.key == SDLK_T)
				{
					if (CVAR_SHADOWS_CASCADE_SELECTION.get() == 1)
						CVAR_SHADOWS_CASCADE_SELECTION.set(0);
					else
						CVAR_SHADOWS_CASCADE_SELECTION.set(1);
				}
			}

			if (!stop_movement)
				main_camera.process_sdl_event(e);

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
			ImGui::Text("Frametime:            %.3f ms", stats.frame_avg * 1000.0f);
			ImGui::Text("Draw calls:           %i", stats.draw_count);
			// ImGui::Text("scene update time %f ms", stats.scene_update_time);
			ImGui::Text("Early cull:           %.3f ms", stats.early_cull);
			ImGui::Text("Late  cull:           %.3f ms", stats.late_cull);
			ImGui::Text("Early render:         %.3f ms", stats.early_indirect);
			ImGui::Text("Late render:          %.3f ms", stats.late_indirect);
			ImGui::Text("Light culling:        %.3f ms", stats.light_culling);
			ImGui::Text("Deferred shading:     %.3f ms", stats.deferred_shading);
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
	}
}

void VulkanEngine::init_vulkan()
{
	vkb::InstanceBuilder builder{};

	// create vulkan instance, with basic debug features
	auto inst_ret = builder.set_app_name("Example Vulkan Application")
	                    .request_validation_layers(USE_VALIDATION_LAYERS)
	                    .use_default_debug_messenger()
	                    .require_api_version(1, 3, 0)
	                    .build();

	vkb::Instance vkb_inst = inst_ret.value();

	// grab the instance
	instance = vkb_inst.instance;
	debug_messenger = vkb_inst.debug_messenger;

	volkLoadInstanceOnly(instance);

	SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface);

	// vulkan 1.3 features
	VkPhysicalDeviceVulkan13Features features13{};
	features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
	features13.dynamicRendering = true;
	features13.synchronization2 = true;
	features13.maintenance4 = true;

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

	// vulkan 1.1 features
	VkPhysicalDeviceVulkan11Features features11{};
	features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
	features11.storageBuffer16BitAccess = true;

	// vulkan 1.0 features
	VkPhysicalDeviceFeatures features10{};
	features10.multiDrawIndirect = true;
	features10.pipelineStatisticsQuery = true;
	// features10.samplerAnisotropy = true;
	features10.depthClamp = true;
	features10.shaderInt16 = true;

	VkPhysicalDeviceMeshShaderFeaturesEXT mesh_shader_features{};
	mesh_shader_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
	mesh_shader_features.meshShader = true;

	VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT fragment_shader_interlock_features{};
	fragment_shader_interlock_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT;
	fragment_shader_interlock_features.fragmentShaderPixelInterlock = true;

	// use vkbootstrap to select a gpu.
	// we want a gpu that can write to the SDL surface and supports vulkan 1.3 with the correct features
	vkb::PhysicalDeviceSelector selector{ vkb_inst };
	vkb::PhysicalDevice physicalDevice = selector
	                                     .set_minimum_version(1, 3)
	                                     .set_required_features(features10)
	                                     .set_required_features_13(features13)
	                                     .set_required_features_12(features12)
	                                     .set_required_features_11(features11)
	                                     .add_required_extension("VK_KHR_calibrated_timestamps")
	                                     .add_required_extension("VK_EXT_mesh_shader")
	                                     .add_required_extension("VK_EXT_fragment_shader_interlock")
	                                     .add_required_extension_features(mesh_shader_features)
	                                     .add_required_extension_features(fragment_shader_interlock_features)
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
	allocator_info.vulkanApiVersion = VK_API_VERSION_1_3;
	allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT; // allows usage of GPU pointers

	VmaVulkanFunctions vulkan_functions{};
	VK_CHECK(vmaImportVulkanFunctionsFromVolk(&allocator_info, &vulkan_functions));
	allocator_info.pVulkanFunctions = &vulkan_functions;
	vmaCreateAllocator(&allocator_info, &allocator);

	main_deletion_queue.push_function([&]()
	                                  { vmaDestroyAllocator(allocator); });

	vkGetPhysicalDeviceProperties(chosen_gpu, &device_properties);
	assert(device_properties.limits.timestampComputeAndGraphics);
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

		auto accum_id = 0;
		for (int i = 0; i < 2; ++i) // ping pong
		{
			accumulation_buffers[i] = create_image(device, allocator, draw_image_extent, VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT
			);
			accum_id = texture_cache.add_texture(accumulation_buffers[i].view);
			if (i == 0)
			{
				texture_cache.set_accumulation_buffer(accum_id);
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
	depth_image.extent = draw_image_extent;

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
		}
	);
}

void VulkanEngine::init_commands()
{
	VkCommandPoolCreateInfo command_pool_info = vkinit::command_pool_create_info(
	    graphics_queue_family,
	    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
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
		}
	);
}

void VulkanEngine::init_sync_structures()
{
	VkFenceCreateInfo fence_info = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
	VkSemaphoreCreateInfo semaphore_info = vkinit::semaphore_create_info();

	for (auto& frame : frames)
	{
		VK_CHECK(vkCreateFence(device, &fence_info, nullptr, &frame.render_fence));

		VK_CHECK(vkCreateSemaphore(device, &semaphore_info, nullptr, &frame.swapchain_semaphore));
		VK_CHECK(vkCreateSemaphore(device, &semaphore_info, nullptr, &frame.render_semaphore));
	}

	VK_CHECK(vkCreateFence(device, &fence_info, nullptr, &imm_fence));

	main_deletion_queue.push_function([&]()
	                                  { vkDestroyFence(device, imm_fence, nullptr); });
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height)
{
	vkb::SwapchainBuilder swapchainBuilder{ chosen_gpu, device, surface };

	// swapchain_image_format = VK_FORMAT_B8G8R8A8_UNORM;
	swapchain_image_format = VK_FORMAT_B8G8R8A8_SRGB;

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
		                                  { frame.frame_descriptor_allocator.destroy_pools(device); });

		frame.scene_buffer = create_buffer(allocator, sizeof(SceneData), VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

		frame.scene_descriptor = frame.frame_descriptor_allocator.allocate(device, scene_descriptor_layout);

		writer.clear();
		writer.write_buffer(0, frame.scene_buffer.buffer, sizeof(SceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
		writer.update_set(device, frame.scene_descriptor);
	}

	std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {
		{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_SAMPLER, 20 },
	};

	global_descriptor_allocator.init(device, 1, sizes);

	//> building bindless layouts
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
	}

	main_deletion_queue.push_function([&]()
		{
			global_descriptor_allocator.destroy_pools(device);
			vkDestroyDescriptorSetLayout(device, scene_descriptor_layout, nullptr);
			vkDestroyDescriptorSetLayout(device, bindless_tex_layout, nullptr);
			vkDestroyDescriptorSetLayout(device, bindless_sampler_layout, nullptr);
			vkDestroyDescriptorSetLayout(device, bindless_image_layout, nullptr);
		}
	);
}

void VulkanEngine::init_pipelines()
{
	// compute pipeline
	shader_cache.add_shader(device, "cluster_grid.comp", VK_SHADER_STAGE_COMPUTE_BIT);
	shader_cache.add_shader(device, "light_culling.comp", VK_SHADER_STAGE_COMPUTE_BIT);
	shader_cache.add_shader(device, "depth_pyramid.comp", VK_SHADER_STAGE_COMPUTE_BIT);
	shader_cache.add_shader(device, "mesh_cull.comp", VK_SHADER_STAGE_COMPUTE_BIT);
	shader_cache.add_shader(device, "meshlet_cull.comp", VK_SHADER_STAGE_COMPUTE_BIT);
	shader_cache.add_shader(device, "shadow_cull.comp", VK_SHADER_STAGE_COMPUTE_BIT);

	// graphics pipeline
	shader_cache.add_shader(device, "mesh.vert", VK_SHADER_STAGE_VERTEX_BIT);
	shader_cache.add_shader(device, "geometry.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
	shader_cache.add_shader(device, "meshlet.mesh", VK_SHADER_STAGE_MESH_BIT_EXT);
	shader_cache.add_shader(device, "full_screen.vert", VK_SHADER_STAGE_VERTEX_BIT);
	shader_cache.add_shader(device, "mlab.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
	shader_cache.add_shader(device, "depth.vert", VK_SHADER_STAGE_VERTEX_BIT);
	shader_cache.add_shader(device, "depth.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

	// vis buffer
	shader_cache.add_shader(device, "vis_buffer.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
	shader_cache.add_shader(device, "vis_meshlet.mesh", VK_SHADER_STAGE_MESH_BIT_EXT);

	// taa
	shader_cache.add_shader(device, "taa_resolve.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

	// gi
	shader_cache.add_shader(device, "equirectangular_to_cubemap.comp", VK_SHADER_STAGE_COMPUTE_BIT);
	shader_cache.add_shader(device, "spherical_harmonics.comp", VK_SHADER_STAGE_COMPUTE_BIT);
	shader_cache.add_shader(device, "irradiance.comp", VK_SHADER_STAGE_COMPUTE_BIT);
	shader_cache.add_shader(device, "prefiltered.comp", VK_SHADER_STAGE_COMPUTE_BIT);
	shader_cache.add_shader(device, "brdf.comp", VK_SHADER_STAGE_COMPUTE_BIT);
	shader_cache.add_shader(device, "luminance_histogram.comp", VK_SHADER_STAGE_COMPUTE_BIT);
	shader_cache.add_shader(device, "luminance_avg.comp", VK_SHADER_STAGE_COMPUTE_BIT);
	shader_cache.add_shader(device, "tonemap.comp", VK_SHADER_STAGE_COMPUTE_BIT);

	shader_cache.add_shader(device, "resolve_vbuffer.comp", VK_SHADER_STAGE_COMPUTE_BIT);
	shader_cache.add_shader(device, "resolve_gbuffer.comp", VK_SHADER_STAGE_COMPUTE_BIT);

#ifdef NDEBUG
	fmt::println("running Release mode"); // ensuring no clion shenanigans
#else
	fmt::println("running Debug mode");
#endif

	std::vector<VkDescriptorSetLayout> descriptor_layouts = { scene_descriptor_layout, bindless_image_layout, bindless_tex_layout, bindless_sampler_layout };

	ComputePipelineBuilder compute_builder{};
	PipelineBuilder builder{};

	shader_passes["cluster_grid"] = vkutil::build_shader(device, compute_builder, shader_cache["cluster_grid.comp"], descriptor_layouts, sizeof(ClusterGridPushConstants));
	shader_passes["light_culling"] = vkutil::build_shader(device, compute_builder, shader_cache["light_culling.comp"], descriptor_layouts, sizeof(LightCullingPushConstants));

	shader_passes["depth_pyramid"] = vkutil::build_shader(device, compute_builder, shader_cache["depth_pyramid.comp"], descriptor_layouts, sizeof(DepthPyramidPushConstants));
	shader_passes["mesh_cull"] = vkutil::build_shader(device, compute_builder, shader_cache["mesh_cull.comp"], descriptor_layouts, sizeof(CullData));
	shader_passes["meshlet_cull"] = vkutil::build_shader(device, compute_builder, shader_cache["meshlet_cull.comp"], descriptor_layouts, sizeof(ClusterCullData)); // TODO: check if this is also culldata
	shader_passes["equirectangular_to_cubemap"] = vkutil::build_shader(device, compute_builder, shader_cache["equirectangular_to_cubemap.comp"], descriptor_layouts, sizeof(IBLPushConstants));
	shader_passes["spherical_harmonics"] = vkutil::build_shader(device, compute_builder, shader_cache["spherical_harmonics.comp"], descriptor_layouts, sizeof(SHPushConstants));
	shader_passes["irradiance"] = vkutil::build_shader(device, compute_builder, shader_cache["irradiance.comp"], descriptor_layouts, sizeof(IBLPushConstants));
	shader_passes["prefiltered"] = vkutil::build_shader(device, compute_builder, shader_cache["prefiltered.comp"], descriptor_layouts, sizeof(IBLPushConstants));
	shader_passes["brdf"] = vkutil::build_shader(device, compute_builder, shader_cache["brdf.comp"], descriptor_layouts, sizeof(IBLPushConstants));
	shader_passes["luminance_histogram"] = vkutil::build_shader(device, compute_builder, shader_cache["luminance_histogram.comp"], descriptor_layouts, sizeof(LuminanceBinsPC));
	shader_passes["luminance_avg"] = vkutil::build_shader(device, compute_builder, shader_cache["luminance_avg.comp"], descriptor_layouts, sizeof(LuminanceBinsPC));
	shader_passes["tonemap"] = vkutil::build_shader(device, compute_builder, shader_cache["tonemap.comp"], descriptor_layouts, sizeof(TonemapPC));
	shader_passes["shadow_cull"] = vkutil::build_shader(device, compute_builder, shader_cache["shadow_cull.comp"], descriptor_layouts, sizeof(ShadowCullPushConstants));

	shader_passes["resolve_vbuffer"] = vkutil::build_shader(device, compute_builder, shader_cache["resolve_vbuffer.comp"], descriptor_layouts, sizeof(DeferredPushConstants));
	shader_passes["resolve_gbuffer"] = vkutil::build_shader(device, compute_builder, shader_cache["resolve_gbuffer.comp"], descriptor_layouts, sizeof(DeferredPushConstants));

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

	struct GBufferSpecializationData
	{
		uint32_t opaque = 1;
	};

	GBufferSpecializationData specialization_data{};
	std::array<VkSpecializationMapEntry, 1> specialization_entries{};
	specialization_entries[0].constantID = 0;
	specialization_entries[0].offset = 0;
	specialization_entries[0].size = sizeof(specialization_data.opaque);

	VkSpecializationInfo specialization_info{};
	specialization_info.mapEntryCount = static_cast<uint32_t>(specialization_entries.size());
	specialization_info.pMapEntries = specialization_entries.data();
	specialization_info.dataSize = sizeof(GBufferSpecializationData);
	specialization_info.pData = &specialization_data;

	builder.set_shaders({ shader_cache["mesh.vert"], shader_cache["geometry.frag"] });
	builder.set_shader_specialization(&specialization_info, 1);
	specialization_data.opaque = 1;
	builder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["geometry_vert"] = vkutil::build_shader(device, builder, {}, descriptor_layouts, sizeof(GPUPushConstants));
	specialization_data.opaque = 0;
	builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["geometry_vert_mask"] = vkutil::build_shader(device, builder, {}, descriptor_layouts, sizeof(GPUPushConstants));

	builder.set_shaders({ shader_cache["meshlet.mesh"], shader_cache["geometry.frag"] });
	builder.set_shader_specialization(&specialization_info, 1);
	builder.shader_stages[1].pSpecializationInfo = &specialization_info;
	specialization_data.opaque = 1;
	builder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["geometry_mesh"] = vkutil::build_shader(device, builder, {}, descriptor_layouts, sizeof(GPUPushConstants));
	specialization_data.opaque = 0;
	builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["geometry_mesh_mask"] = vkutil::build_shader(device, builder, {}, descriptor_layouts, sizeof(GPUPushConstants));

	color_attachment_formats.clear();
	color_attachment_formats.push_back(visibility_buffer.format);
	color_attachment_formats.push_back(velocity_buffer.format);
	builder.set_color_attachment_format(color_attachment_formats);
	color_blend_states.clear();
	color_blend_states.push_back(builder.disable_blending()); // 2 channel texture but RGBA write mask ok? no validation error
	color_blend_states.push_back(builder.disable_blending()); // 2 channel texture but RGBA write mask ok? no validation error
	builder.set_blending_state(color_blend_states);
	builder.set_shaders({ shader_cache["vis_meshlet.mesh"], shader_cache["vis_buffer.frag"] });
	builder.set_shader_specialization(&specialization_info, 1);
	specialization_data.opaque = 1;
	builder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["visibility_mesh"] = vkutil::build_shader(device, builder, {}, descriptor_layouts, sizeof(GPUPushConstants));
	specialization_data.opaque = 0;
	builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["visibility_mesh_mask"] = vkutil::build_shader(device, builder, {}, descriptor_layouts, sizeof(GPUPushConstants));

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
	builder.set_shaders({ shader_cache["depth.vert"], shader_cache["depth.frag"] });
	builder.set_shader_specialization(&specialization_info, 1);
	specialization_data.opaque = 1;
	builder.set_cull_mode(VK_CULL_MODE_FRONT_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	// builder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["depth"] = vkutil::build_shader(device, builder, {}, descriptor_layouts, sizeof(ShadowPushConstants));
	specialization_data.opaque = 0;
	builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["depth_mask"] = vkutil::build_shader(device, builder, {}, descriptor_layouts, sizeof(ShadowPushConstants));
	builder.dynamic_state.pop_back(); // reset
	builder.rasterization.depthClampEnable = VK_FALSE; // reset

	color_attachment_formats.clear();
	color_attachment_formats.push_back(draw_image.format);
	builder.set_color_attachment_format(color_attachment_formats);
	builder.disable_depth();
	builder.set_depth_format(VK_FORMAT_UNDEFINED);
	builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["taa_resolve"] = vkutil::build_shader(device, builder, { shader_cache["full_screen.vert"], shader_cache["taa_resolve.frag"] }, descriptor_layouts, sizeof(TAAResolvePC));

	builder.enable_depth(false, VK_COMPARE_OP_GREATER_OR_EQUAL);
	color_attachment_formats.clear();
	color_attachment_formats.push_back(VK_FORMAT_UNDEFINED);
	builder.set_color_attachment_format(color_attachment_formats);
	builder.set_depth_format(depth_image.format);
	builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["mlab_vert"] = vkutil::build_shader(device, builder, { shader_cache["mesh.vert"], shader_cache["mlab.frag"] }, descriptor_layouts, sizeof(GPUPushConstants));
	shader_passes["mlab_mesh"] = vkutil::build_shader(device, builder, { shader_cache["meshlet.mesh"], shader_cache["mlab.frag"] }, descriptor_layouts, sizeof(GPUPushConstants));

	for (const auto& [_, shader_program] : shader_cache.data)
	{
		vkDestroyShaderModule(device, shader_program.get()->module, nullptr);
	}
}

void VulkanEngine::init_default_data()
{
	//> default textures
	uint32_t magenta_color = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
	std::array<uint32_t, 16 * 16> pixels{}; // for 16x16 checkerboard texture
	for (int x = 0; x < 16; x++)
	{
		for (int y = 0; y < 16; y++)
		{
			pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta_color : 0;
		}
	}

	VkSampler sampler{};
	VkSamplerCreateInfo sampler_info{};
	sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sampler_info.magFilter = VK_FILTER_LINEAR;
	sampler_info.minFilter = VK_FILTER_LINEAR;

	sampler_info.maxLod = VK_LOD_CLAMP_NONE;
	sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

	// sampler_info.anisotropyEnable = VK_TRUE;
	// sampler_info.maxAnisotropy = 16.0f;

	vkCreateSampler(device, &sampler_info, nullptr, &sampler); // 1 linear
	sampler_cache.add_sampler(sampler);

	// sampler_info.anisotropyEnable = VK_FALSE;
	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

	vkCreateSampler(device, &sampler_info, nullptr, &sampler); // 2 cube map sampling
	sampler_cache.add_sampler(sampler);

	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; // tailored to our PCF sampling; manual OOB rejection required in shader
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.maxLod = 1.0;

	vkCreateSampler(device, &sampler_info, nullptr, &sampler); // 3 shadow map sampler
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

	vkCreateSampler(device, &sampler_info, nullptr, &sampler); // 4 building hi-z
	sampler_cache.add_sampler(sampler);

	sampler_info.pNext = nullptr;
	sampler_info.magFilter = VK_FILTER_NEAREST;
	sampler_info.minFilter = VK_FILTER_NEAREST;
	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;

	vkCreateSampler(device, &sampler_info, nullptr, &sampler);
	sampler_cache.add_sampler(sampler); // 5 nearest clamp to border

	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	vkCreateSampler(device, &sampler_info, nullptr, &sampler);
	sampler_cache.add_sampler(sampler); // 6 nearest clamp to edge

	sampler_info.magFilter = VK_FILTER_LINEAR;
	sampler_info.minFilter = VK_FILTER_LINEAR;
	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.maxLod = VK_LOD_CLAMP_NONE;
	sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	vkCreateSampler(device, &sampler_info, nullptr, &sampler);
	sampler_cache.add_sampler(sampler); // 7 linear clamp to edge

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
		}
	);

	//> init scene
	render_scene.init();

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
	    }
	);

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

	auto clusters_x = get_groupcount(window_extent.width, CLUSTER_DIM);
	auto clusters_y = get_groupcount(window_extent.height, CLUSTER_DIM);
	const uint32_t total_clusters = clusters_x * clusters_y * CLUSTER_DEPTH_SLICES;

	light_cluster_buffer = create_buffer(allocator, total_clusters * sizeof(ClusterAABB), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
	light_index_buffer = create_buffer(allocator, total_clusters * MAX_POINT_LIGHTS * sizeof(uint32_t), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT); // could use smaller more conservative size
	light_grid_buffer = create_buffer(allocator, total_clusters * sizeof(LightGrid), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
	light_count_buffer = create_buffer(allocator, sizeof(uint32_t), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

	// gi
	const char* hdri_path = {"assets/pisa.hdr"};
	float* data{};

	int width{};
	int height{};
	int channels{};

	data = stbi_loadf(hdri_path, &width, &height, &channels, STBI_rgb_alpha);

	auto extent = VkExtent3D(static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1);

	hdri = upload_image(this, device, allocator, (void*)data, extent,
		VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT
	);

	auto hdri_id = texture_cache.add_texture(hdri.view);
	texture_cache.set_hdri(hdri_id);

	extent.width /= 4;
	extent.height = extent.width;

	hdri_cubemap = create_cubemap(device, allocator, extent, VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, true
	);

	scene_data.textures[0] = static_cast<float>(texture_cache.add_texture(hdri_cubemap.view));
	auto hdri_cubemap_id = image_cache.add_texture(hdri_cubemap.view);
	image_cache.set_hdri(hdri_cubemap_id);

	irradiance_cubemap = create_cubemap(device, allocator, VkExtent3D{ 64, 64, 1}, VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT
	);

	scene_data.textures[1] = static_cast<float>(texture_cache.add_texture(irradiance_cubemap.view));
	image_cache.add_texture(irradiance_cubemap.view);

	prefiltered_envmap = create_cubemap(device, allocator, VkExtent3D{ 512, 512, 1}, VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, true
	);

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

	brdf_lut = create_image(device, allocator, VkExtent3D{ 128, 128, 1}, VK_FORMAT_R16G16_SFLOAT,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT
	);

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
	    }
	);

	// initialize jitter offsets
	{
		float offset_x = 1.0f / (2.0f * static_cast<float>(draw_extent.width)); // TODO: if swapchain resized, need to amend this
		float offset_y = 1.0f / (2.0f * static_cast<float>(draw_extent.height));
		jx = { offset_x, offset_x, -offset_x, -offset_x };
		jy = { offset_y, -offset_y, offset_y, -offset_y };
	}
}

void VulkanEngine::init_renderables(const std::string& file_path)
{
	auto start = std::chrono::system_clock::now();

	{
		auto asset_file = load_gltf(this, file_path);
		assert(asset_file.has_value());
		loaded_scene = std::move(*asset_file);
	}

	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	float ret = static_cast<float>(elapsed.count()) / 1000.0f;
	fmt::println("load gltf: {}ms", ret);

	render_scene.vertex_buffer = upload_buffer(this, allocator, loaded_scene->vertices.data(), loaded_scene->vertices.size() * sizeof(Vertex));
	render_scene.index_buffer = upload_buffer(this, allocator, loaded_scene->indices.data(), loaded_scene->indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
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
		uint32_t meshlet_count = renderable.meshlet_bits;
		renderable.meshlet_bits = meshlet_visibility_offset; // TODO: rename meshlet_bits
		render_scene.max_meshtask_commands += (meshlet_count + MESHLETS_PER_MESHTASKCOMMAND - 1) / MESHLETS_PER_MESHTASKCOMMAND;
		meshlet_visibility_offset += meshlet_count;
	}
	render_scene.total_meshlets_bits = meshlet_visibility_offset; // likely obsolete
}

void VulkanEngine::init_bindless()
{
	// TODO: move ds allocation out of this maybe?
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

				render_scene.meshes.emplace_back(Mesh{
					.center = mesh.center,
					.radius = mesh.radius,
					.mesh_lods = mesh.mesh_lods,
					.lod_count = mesh.lod_count,
					.vertex_offset = mesh.vertex_offset
				});
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
	stats.draw_count = 0;
	auto start = std::chrono::system_clock::now();

	main_camera.near = static_cast<float>(CVAR_MISC_DRAW_DISTANCE.get());
	main_camera.update(static_cast<float>(stats.deltatime));

	scene_data.view = main_camera.get_view_matrix();
	scene_data.proj = main_camera.perspective;

	if (CVAR_RENDER_TAA.get())
	{
		auto jitter_index = frame_number % 8;
		float halton_x = 2.0f * Halton(jitter_index + 1, 2) - 1.0f;
		float halton_y = 2.0f * Halton(jitter_index + 1, 3) - 1.0f;
		float x = halton_x / static_cast<float>(draw_extent.width);
		float y = halton_y / static_cast<float>(draw_extent.height);

		auto offset_projection = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, 0.0));
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

	if (CVAR_RENDER_SHADOWS.get())
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
	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	stats.scene_update_time = static_cast<float>(elapsed.count()) / 1000.0f; // milliseconds
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& func) const
{
	VK_CHECK(vkResetFences(device, 1, &imm_fence));
	VK_CHECK(vkResetCommandBuffer(imm_command_buffer, 0));

	VkCommandBufferBeginInfo cmd_begin_info = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(imm_command_buffer, &cmd_begin_info));

	func(imm_command_buffer);

	VK_CHECK(vkEndCommandBuffer(imm_command_buffer));

	VkCommandBufferSubmitInfo cmd_submit_info = vkinit::command_buffer_submit_info(imm_command_buffer);

	VkSubmitInfo2 submit = vkinit::submit_info(&cmd_submit_info, nullptr, nullptr);

	VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit, imm_fence));

	VK_CHECK(vkWaitForFences(device, 1, &imm_fence, true, 9999999999));
}

void VulkanEngine::execute_deferred_shading(VkCommandBuffer cmd, VkImageView view)
{
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

	VkRenderingAttachmentInfo color_attachment = vkinit::attachment_info(view, nullptr);
	VkRenderingInfo render_info = vkinit::rendering_info(draw_extent, &color_attachment, nullptr);

	vkCmdBeginRendering(cmd, &render_info);

	ShaderPass current_pass{};
	bool visibility_rendering = CVAR_RENDER_VBUFFER.get() && CVAR_RENDER_MESH_SHADERS.get();
	if (visibility_rendering)
	{
		current_pass = *shader_passes["resolve_vbuffer"];
	}
	else
	{
		current_pass = *shader_passes["resolve_gbuffer"];
	}
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);

	DeferredPushConstants pc{};

	uint32_t cluster_x = get_groupcount(window_extent.width, CLUSTER_DIM);
	uint32_t cluster_y = get_groupcount(window_extent.height, CLUSTER_DIM);
	pc.cluster_size = glm::vec4(cluster_x, cluster_y, CLUSTER_DEPTH_SLICES, CLUSTER_DIM);
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
	pc.sh_buffer_address = get_buffer_address(device, render_scene.sh_buffer.buffer);

	pc.depth_id = texture_cache.get_depth_image();
	pc.gbuffer_id = visibility_rendering ? texture_cache.get_visibility_buffer() : texture_cache.get_first_gbuffer();
	pc.shadow_id = texture_cache.get_shadowmap();
	pc.light_culling = CVAR_RENDER_POINT_LIGHTS.get();
	pc.near = main_camera.far;

	const float ratio = main_camera.near / main_camera.far;
	pc.scale = static_cast<float>(CLUSTER_DEPTH_SLICES) / std::log(ratio);
	pc.bias = static_cast<float>(CLUSTER_DEPTH_SLICES) * std::log(main_camera.far) / std::log(ratio);
	pc.resolve_transparent = CVAR_RENDER_TRANSPARENT.get();
	pc.shadows = CVAR_RENDER_SHADOWS.get();
	pc.pcf = CVAR_SHADOWS_PCF.get();
	pc.max_prefiltered_lod = static_cast<float>(std::floor(std::log2(static_cast<float>(std::max(prefiltered_envmap.extent.width, prefiltered_envmap.extent.height))))) + 1;
	pc.metallic = CVAR_PBR_METALLIC.get();
	pc.roughness = CVAR_PBR_ROUGHNESS.get();
	pc.debug = CVAR_DEBUG_TEXTURES.get();
	pc.map = CVAR_SHADOWS_CASCADE_SELECTION.get();

	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(DeferredPushConstants), &pc);
	vkCmdDraw(cmd, 3, 1, 0, 0);

	stats.draw_count++;

	vkCmdEndRendering(cmd);
}

void VulkanEngine::execute_taa_resolve(VkCommandBuffer cmd, VkImageView view)
{
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

	VkRenderingAttachmentInfo color_attachment = vkinit::attachment_info(view, nullptr); // TODO: load_op_dont_care?
	VkRenderingInfo render_info = vkinit::rendering_info(draw_extent, &color_attachment, nullptr); // TODO: do we need depth?

	vkCmdBeginRendering(cmd, &render_info);

	ShaderPass current_pass = *shader_passes["taa_resolve"];
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);

	TAAResolvePC pc{};
	pc.screen_size = glm::vec2(static_cast<float>(draw_extent.width), static_cast<float>(draw_extent.height));
	auto jitter_index = frame_number % 8;
	float halton_x = 2.0f * Halton(jitter_index + 1, 2) - 1.0f;
	float halton_y = 2.0f * Halton(jitter_index + 1, 3) - 1.0f;
	float x = halton_x / static_cast<float>(draw_extent.width);
	float y = halton_y / static_cast<float>(draw_extent.height);
	pc.current_jitter = glm::vec2(x, y); // current jitter only
	pc.accum_id = texture_cache.get_accumulation_buffer((frame_number + 1) % 2);
	pc.color_id = texture_cache.get_accumulation_buffer(frame_number % 2);
	pc.depth_id = texture_cache.get_depth_image();
	pc.velocity_id = texture_cache.get_visibility_buffer() + 1; // TODO: hardcoded, maybe give velocity its own setter/getter?
	pc.variance_clipping = CVAR_TAA_VARIANCE_CLIP.get();
	pc.history_filter = CVAR_TAA_CATMULL_ROM.get();
	pc.local_filter = CVAR_TAA_MITCHELL.get();
	pc.ycocg = CVAR_TAA_YCOCG.get();
	pc.depth_dilation = CVAR_TAA_DEPTH_DILATION.get();
	pc.weigh_luminance = CVAR_TAA_LUMINANCE_WEIGHING.get();

	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(TAAResolvePC), &pc);
	vkCmdDraw(cmd, 3, 1, 0, 0);

	stats.draw_count++;

	vkCmdEndRendering(cmd);
}

void VulkanEngine::update_cascade()
{
	// https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-10-parallel-split-shadow-maps-programmable-gpus
	float far = static_cast<float>(CVAR_SHADOWS_DISTANCE.get());
	float near = main_camera.far;
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
	glm::mat4 proj = glm::perspective(glm::radians(main_camera.fov), static_cast<float>(draw_extent.width) / static_cast<float>(draw_extent.height), static_cast<float>(CVAR_SHADOWS_DISTANCE.get()), main_camera.far);
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
	init_info.ApiVersion = VK_API_VERSION_1_3;
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
		}
	);
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
	// clang-format off

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
	render_scene.vis_buffer = create_buffer(allocator,render_scene.renderables.size() * sizeof(uint32_t), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
	fmt::println("vis_buffer: {}mb", size_in_bytes(render_scene.vis_buffer.info.size));

	immediate_submit([&](VkCommandBuffer cmd)
		{
			vkCmdFillBuffer(cmd, render_scene.vis_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		}
	);

	render_scene.dispatch_buffer = create_buffer(allocator, 3 * sizeof(uint32_t), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT);
	// TODO: can we combine both of these?
	render_scene.cluster_count_buffer = create_buffer(allocator,3 * sizeof(uint32_t),0,VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT);

	auto count_size = 2 * sizeof(uint32_t);
	auto draw_commands_size = (MAX_OPAQUE_DRAWS + MAX_ALPHACLIP_DRAWS) * sizeof(VkDrawIndexedIndirectCommand);
	render_scene.draw_indirect_buffer = create_buffer(allocator, (count_size + draw_commands_size) * NUMBER_OF_CASCADES, 0, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
	fmt::println("draw_indirect_buffer: {}mb", size_in_bytes(render_scene.draw_indirect_buffer.info.size));

	// limit of ~16.7 meshlets, ~64mb
	// TODO: implement error handling/limit check in shader; just drop the meshlets?
	render_scene.cluster_indices = create_buffer( allocator, MESHLET_LIMIT * sizeof(uint32_t), 0, VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT);
	fmt::println("cluster_indices: {}mb", size_in_bytes(render_scene.cluster_indices.info.size));

	// TODO: impose a limit on this
	render_scene.meshtask_indirect_buffer = create_buffer(allocator, render_scene.max_meshtask_commands * sizeof(MeshTaskCommand), 0, VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT);
	fmt::println("meshtask_indirect_buffer: {}mb", size_in_bytes(render_scene.meshtask_indirect_buffer.info.size));

	{
		size_t meshlet_visibility_size = (render_scene.total_meshlets_bits + 31) / 32;
		render_scene.meshlet_vis_buffer = create_buffer(allocator, meshlet_visibility_size * sizeof(uint32_t), 0, VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
		fmt::println("meshlet_vis_buffer: {}mb", size_in_bytes(render_scene.meshlet_vis_buffer.info.size));

		immediate_submit([&](VkCommandBuffer cmd)
			{
				vkCmdFillBuffer(cmd, render_scene.meshlet_vis_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
			}
		);
	}

	// TODO: refactor if window resize
	{
		auto screen_pixels = window_extent.width * window_extent.height;
		render_scene.oit_buffer = create_buffer(allocator, screen_pixels * sizeof(OITData), 0, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
		fmt::println("oit_buffer: {}mb", size_in_bytes(render_scene.oit_buffer.info.size));

		immediate_submit([&](VkCommandBuffer cmd)
			{
				vkCmdFillBuffer(cmd, render_scene.oit_buffer.buffer, 0, VK_WHOLE_SIZE, 0x3F800000);
			}
		);
	}

	// clang-format on
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
	cull_data.count_buffer_address = get_buffer_address(device, render_scene.dispatch_buffer.buffer);;
	cull_data.vis_buffer_address = get_buffer_address(device, render_scene.vis_buffer.buffer);
	cull_data.meshtask_buffer_address = get_buffer_address(device, render_scene.meshtask_indirect_buffer.buffer);

	// cull_data.count = static_cast<uint32_t>(pass.unbatched_objects.size()); // set during execute
	cull_data.texture_id = texture_cache.get_depth_pyramid_image();
	cull_data.occlusion_enabled = CVAR_RENDER_OCCLUSION_CULL.get();

	cull_data.p00 = proj[0][0];
	cull_data.p11 = proj[1][1]; // equivalent to 1 / tan(fovy/2)
	cull_data.near = main_camera.far;
	cull_data.far = main_camera.near;

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
	cull_data.cluster_count_address = get_buffer_address(device, render_scene.cluster_count_buffer.buffer);
	cull_data.cluster_vis_address = get_buffer_address(device, render_scene.meshlet_vis_buffer.buffer);
	cull_data.meshtask_buffer_address = get_buffer_address(device, render_scene.meshtask_indirect_buffer.buffer);

	// cull_data.count = static_cast<uint32_t>(pass.unbatched_objects.size()); // unused
	cull_data.texture_id = texture_cache.get_depth_pyramid_image();
	cull_data.occlusion_enabled = CVAR_RENDER_OCCLUSION_CULL.get();

	cull_data.p00 = proj[0][0];
	cull_data.p11 = proj[1][1];
	cull_data.near = main_camera.far;
	cull_data.far = main_camera.near;

	cull_data.resolution = glm::vec2(depth_pyramid.extent.width, depth_pyramid.extent.height);
	cull_data.texture_lod = static_cast<float>(std::floor(std::log2(static_cast<float>(std::max(depth_pyramid.extent.width, depth_pyramid.extent.height)))) + 1);
	cull_data.lod_distance_factor = 2.0f / (cull_data.p11 * static_cast<float>(draw_extent.height));
	cull_data.lod_enabled = CVAR_RENDER_LOD.get();
	cull_data.task_submit = CVAR_RENDER_MESH_SHADERS.get();
}

void VulkanEngine::execute_compute_cull(VkCommandBuffer cmd, const RenderScene::MeshPass& pass, CullData& cull_data, bool late, uint32_t post_pass)
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

void VulkanEngine::execute_compute_cull(VkCommandBuffer cmd, RenderScene::MeshPass& pass, ClusterCullData& cull_data, VkBuffer count_buffer, uint32_t offset, bool late, uint32_t post_pass)
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

	vkCmdDispatchIndirect(cmd, count_buffer, offset);
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
	VkClearColorValue clear_color_value{{ 0.f, 0.f, 0.f, 1.0f }};
	VkClearValue clear_value{ .color = clear_color_value };
	VkClearColorValue clear_color_value2{{ 1.f, 1.f, 0.f, 1.0f }};
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
	pc.meshtask_buffer_address = get_buffer_address(device, render_scene.meshtask_indirect_buffer.buffer);
	pc.meshlet_buffer_address = get_buffer_address(device, render_scene.meshlet_buffer.buffer);
	pc.meshlet_indices_buffer_address = get_buffer_address(device, render_scene.meshlet_indices.buffer);
	pc.cluster_indices_address = get_buffer_address(device, render_scene.cluster_indices.buffer);
	pc.material_buffer_address = get_buffer_address(device, render_scene.material_buffer.buffer);
	pc.oit_buffer_address = get_buffer_address(device, render_scene.oit_buffer.buffer);

	// TODO: currently used by visibility path only
	pc.jitter_offset = glm::vec2(0.0);
	for (int i = 0; i < 2; ++i)
	{
		auto jitter_index = (frame_number - (i * -1)) % 8;
		float halton_x = 2.0f * Halton(jitter_index + 1, 2) - 1.0f;
		float halton_y = 2.0f * Halton(jitter_index + 1, 3) - 1.0f;
		float x = halton_x / static_cast<float>(draw_extent.width);
		float y = halton_y / static_cast<float>(draw_extent.height);
		if (i == 0)
			pc.jitter_offset += glm::vec2(x, y);
		else
			pc.jitter_offset -= glm::vec2(x, y);
	}

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

		stats.draw_count++;
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
		vkCmdDrawMeshTasksIndirectEXT(cmd, render_scene.cluster_count_buffer.buffer, 0, 1, 0);

		stats.draw_count++;
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
	pc.meshtask_buffer_address = get_buffer_address(device, render_scene.meshtask_indirect_buffer.buffer);
	pc.meshlet_buffer_address = get_buffer_address(device, render_scene.meshlet_buffer.buffer);
	pc.meshlet_indices_buffer_address = get_buffer_address(device, render_scene.meshlet_indices.buffer);
	pc.cluster_indices_address = get_buffer_address(device, render_scene.cluster_indices.buffer);
	pc.material_buffer_address = get_buffer_address(device, render_scene.material_buffer.buffer);
	pc.oit_buffer_address = get_buffer_address(device, render_scene.oit_buffer.buffer);

	if (!CVAR_RENDER_MESH_SHADERS.get())
	{
		ShaderPass current_pass = *shader_passes["mlab_vert"];

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);

		vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GPUPushConstants), &pc);

		vkCmdBindIndexBuffer(cmd, render_scene.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);

		vkCmdDrawIndexedIndirectCount(cmd, render_scene.draw_indirect_buffer.buffer, 2 * sizeof(uint32_t), render_scene.draw_indirect_buffer.buffer, 0, MAX_MESH_DRAWS, sizeof(VkDrawIndexedIndirectCommand));

		stats.draw_count++;
	}
	else // mesh shading path
	{
		ShaderPass current_pass = *shader_passes["mlab_mesh"];

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_image_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_tex_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 3, 1, &bindless_sampler_descriptor, 0, nullptr);

		vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GPUPushConstants), &pc);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
		vkCmdDrawMeshTasksIndirectEXT(cmd, render_scene.cluster_count_buffer.buffer, 0, 1, 0);

		stats.draw_count++;
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

		vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ShadowPushConstants), &pc);

		vkCmdBindIndexBuffer(cmd, render_scene.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

		auto cascade_offset = cascade_idx * (2 * sizeof(uint32_t) + (MAX_OPAQUE_DRAWS + MAX_ALPHACLIP_DRAWS) * sizeof(VkDrawIndexedIndirectCommand));

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
		vkCmdDrawIndexedIndirectCount(cmd, render_scene.draw_indirect_buffer.buffer, 2 * sizeof(uint32_t) + cascade_offset, render_scene.draw_indirect_buffer.buffer, 0 + cascade_offset, MAX_OPAQUE_DRAWS, sizeof(VkDrawIndexedIndirectCommand));
		stats.draw_count++;

		if (CVAR_RENDER_ALPHACLIP.get())
		{
			current_pass = *shader_passes["depth_mask"];
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
			vkCmdDrawIndexedIndirectCount(cmd, render_scene.draw_indirect_buffer.buffer, 2 * sizeof(uint32_t) + MAX_OPAQUE_DRAWS * sizeof(VkDrawIndexedIndirectCommand) + cascade_offset, render_scene.draw_indirect_buffer.buffer, sizeof(uint32_t) + cascade_offset, MAX_ALPHACLIP_DRAWS, sizeof(VkDrawIndexedIndirectCommand));
			stats.draw_count++;
		}
	}

	vkCmdEndRendering(cmd);
	vkCmdEndQuery(cmd, get_current_frame().query_pool_pipelines, query);
}

void VulkanEngine::build_depth_pyramid(VkCommandBuffer cmd)
{
	ShaderPass current_pass = *shader_passes["depth_pyramid"];
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
			barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
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

	VK_CHECK(vkResetCommandBuffer(cmd, 0));

	VkCommandBufferBeginInfo cmd_begin_info = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));

	ShaderPass current_pass = *shader_passes["cluster_grid"];
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

	ClusterGridPushConstants pc{};
	pc.inverse_proj = glm::inverse(main_camera.perspective);
	pc.light_cluster_buffer_address = get_buffer_address(device, light_cluster_buffer.buffer);
	pc.screen_size = glm::vec2(window_extent.width, window_extent.height);
	pc.cluster_dim = CLUSTER_DIM;
	pc.near = main_camera.far; // reverse-z
	pc.far = main_camera.near;

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
	auto groupcount_x = get_groupcount(window_extent.width, CLUSTER_DIM);
	auto groupcount_y = get_groupcount(window_extent.height, CLUSTER_DIM);
	vkCmdDispatch(cmd, groupcount_x, groupcount_y, CLUSTER_DIM);
}

void VulkanEngine::resolve_shading(VkCommandBuffer cmd)
{
ShaderPass current_pass{};
	bool visibility_rendering = CVAR_RENDER_VBUFFER.get() && CVAR_RENDER_MESH_SHADERS.get();
	if (visibility_rendering)
	{
		current_pass = *shader_passes["resolve_vbuffer"];
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

	DeferredPushConstants pc{};

	uint32_t cluster_x = get_groupcount(window_extent.width, CLUSTER_DIM);  // # of clusters in x
	uint32_t cluster_y = get_groupcount(window_extent.height, CLUSTER_DIM); // # of clusters in y
	pc.cluster_size = glm::vec4(cluster_x, cluster_y, CLUSTER_DEPTH_SLICES, CLUSTER_DIM);
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
	pc.sh_buffer_address = get_buffer_address(device, render_scene.sh_buffer.buffer);

	pc.depth_id = texture_cache.get_depth_image();
	pc.gbuffer_id = visibility_rendering ? texture_cache.get_visibility_buffer() : texture_cache.get_first_gbuffer();
	pc.shadow_id = texture_cache.get_shadowmap();
	pc.light_culling = CVAR_RENDER_POINT_LIGHTS.get();
	pc.near = main_camera.far;

	const float ratio = main_camera.near / main_camera.far;
	pc.scale = static_cast<float>(CLUSTER_DEPTH_SLICES) / std::log(ratio);
	pc.bias = static_cast<float>(CLUSTER_DEPTH_SLICES) * std::log(main_camera.far) / std::log(ratio);
	pc.resolve_transparent = CVAR_RENDER_TRANSPARENT.get();
	pc.shadows = CVAR_RENDER_SHADOWS.get();
	pc.pcf = CVAR_SHADOWS_PCF.get();
	pc.max_prefiltered_lod = static_cast<float>(std::floor(std::log2(static_cast<float>(std::max(prefiltered_envmap.extent.width, prefiltered_envmap.extent.height))))) + 1;
	pc.metallic = CVAR_PBR_METALLIC.get();
	pc.roughness = CVAR_PBR_ROUGHNESS.get();
	pc.debug = CVAR_DEBUG_TEXTURES.get();
	pc.map = CVAR_SHADOWS_CASCADE_SELECTION.get();

	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DeferredPushConstants), &pc);
	auto groupcount_x = get_groupcount(draw_extent.width, 8);
	auto groupcount_y = get_groupcount(draw_extent.height, 8);
	vkCmdDispatch(cmd, groupcount_x, groupcount_y, 1);
}
