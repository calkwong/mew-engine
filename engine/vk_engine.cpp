#include "common.h"
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
// #include <tracy/Tracy.hpp>
// #include <tracy/TracyVulkan.hpp>
// #include <glm/gtx/string_cast.hpp>
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
#include <filesystem>
#include <functional>
#include <memory>
#include <random>
#include <ranges>
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

// #define IBL
#define SINGLE // uncomment if loading a proper scene

bool RENDER_IMGUI = true;

constexpr uint32_t SHADOW_MAP_SIZE{ 4096 };
constexpr int NUMBER_OF_CASCADES{ 4 };
constexpr int GBUFFER_COUNT{ 3 };
constexpr int LIGHT_COUNT{ 1000 };
constexpr int CLUSTER_DIM{ 64 };
constexpr int CLUSTER_SLICE_COUNT{ 24 };
constexpr int QUERY_COUNT{ 50 };
constexpr int TIMESTAMP_QUERIES{ 24 };
constexpr int PIPELINE_QUERIES{ 8 };
constexpr int MAX_OPAQUE_DRAWS{ 200000 };
constexpr int MAX_ALPHACLIP_DRAWS{ 200000 };

AutoCVar_Int CVAR_DRAW_DISTANCE{ "Draw distance", 100, 100, CVarFlags::EditSliderInt, 100, 1000, 100 };
AutoCVar_Int CVAR_TOGGLE_MESH_SHADING{ "Mesh shading", 1, 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_TOGGLE_OCCLUSION{ "Occlusion", 1, 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_TOGGLE_LOD{ "LOD", 1, 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_TOGGLE_FREEZE{ "Freeze rendering", 0, 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_TOGGLE_VIEW_MESHLETS{ "Visualize meshlets", 0, 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_TOGGLE_DEPTH_PYRAMID{ "Visualize Hi-Z", 0, 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_DEPTH_PYRAMID_LOD{ "Hi-Z LOD", 0, 0, CVarFlags::EditSliderInt, 0, 10, 1 };
AutoCVar_Int CVAR_TOGGLE_LIGHT_CULLING{ "Light clustered culling", 0, 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_TOGGLE_MASK{ "Render masked geometry", 1, 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_TOGGLE_TRANSPARENT{ "Render transparent geometry", 0, 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_TOGGLE_SHADOW{ "Render shadows", 0, 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_TOGGLE_SOFT_SHADOWS{ "PCF", 1, 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_TOGGLE_DEBUG_SHADOWMAP{ "Debug shadows", 0, 0, CVarFlags::EditSliderInt, 0, 4, 1 };
AutoCVar_Int CVAR_TOGGLE_DEBUG_CASCADES{ "Debug cascades", 0, 0, CVarFlags::EditCheckbox };
AutoCVar_Float CVAR_CSM_LAMBDA{ "CSM log factor", 0.95f, 0.95f, CVarFlags::EditSliderFloat, 0.f, 1.f, 0.05f };
AutoCVar_Int CVAR_SHADOW_DISTANCE{ "Shadow distance", 48, 48, CVarFlags::EditSliderInt, 20, 200, 5 };
AutoCVar_Int CVAR_TOGGLE_VIS_BUFFER{ "Visibility renderer", 1, 1, CVarFlags::EditCheckbox };

uint32_t nearest_pow2(uint32_t extent)
{
	return 1 << static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(extent))));
}

void VulkanEngine::init(std::vector<std::string>& file_paths)
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
	main_camera.near = static_cast<float>(CVAR_DRAW_DISTANCE.get());
	main_camera.far = 0.5f;
	// main_camera.far = 0.01f;
	main_camera.fov = 70.0f;
	// TODO: refactor if window resize
	main_camera.set_perspective_matrix(glm::radians(main_camera.fov), static_cast<float>(draw_extent.width) / static_cast<float>(draw_extent.height), main_camera.far);

	init_default_data();

	init_renderables(file_paths);

	init_bindless();

	init_imgui();

	build_cluster_grid(); // TODO: support draw distance change
	// init_precomputations();

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

		loaded_scenes.clear();

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

		// TODO: possibly destroy loadedgltf resources here instead?

		for (const auto& shader : std::views::values(shader_passes))
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

void VulkanEngine::draw()
{
	{
		VK_CHECK(vkWaitForFences(device, 1, &get_current_frame().render_fence, true, 1000000000));
	}

	VK_CHECK(vkResetFences(device, 1, &get_current_frame().render_fence));

	get_current_frame().deletion_queue.flush();

	auto* scene_uniform_data = static_cast<SceneData*>(get_current_frame().scene_buffer.info.pMappedData);
	*scene_uniform_data = scene_data;

	ready_mesh_draw();

	std::vector<RenderScene::MeshPass*> passes = { &render_scene.opaque_pass };

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

		// TODO: hardcoded - refactor
		std::vector<double*> stats_ref = { &stats.early_cull, &stats.early_indirect, &stats.late_cull, &stats.late_indirect, &stats.mask_cull, &stats.mask_indirect,
			                               &stats.light_culling, &stats.deferred_shading, &stats.transparent_cull, &stats.transparent_render, &stats.shadow_cull, &stats.shadow_render };
		for (size_t i = 0; i < timestamp_results.size(); i = i + 2)
		{
			{
				auto time = static_cast<double>(timestamp_results[i + 1] - timestamp_results[i]) * props.limits.timestampPeriod * 1e-6;
				*stats_ref[i / 2] = time;
			}
		}

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
		if (CVAR_TOGGLE_MESH_SHADING.get())
		{
			vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

			ShaderPass current_pass = *shader_passes["task_submit"];
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

			auto addr = get_buffer_address(device, render_scene.dispatch_buffer.buffer);

			vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkDeviceAddress), &addr);
			vkCmdDispatch(cmd, 1, 1, 1); // TODO: task submit - currently redundant, but may come useful as renderer becomes more complex

			vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);

			execute_compute_cull(cmd, render_scene.opaque_pass, forward_cluster_cull_data, render_scene.dispatch_buffer.buffer, 0, false, 0);
		}
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 1);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

		vkutil::transition_image(
		    cmd,
		    depth_image.image,
		    VK_IMAGE_LAYOUT_UNDEFINED,
		    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
		    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, // TODO: do we need fragment shader bit? cc deferred.frag
		    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
		    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, // TODO: do we need shader sample? cc deferred.frag
		    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
		    VK_IMAGE_ASPECT_DEPTH_BIT
		);

		for (int i = 0; i < GBUFFER_COUNT; i++)
		{
			vkutil::transition_image(
			    cmd,
			    gbuffers[i].image,
			    VK_IMAGE_LAYOUT_UNDEFINED,
			    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
			    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
			    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
			    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
			);
		}

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 2);
		render(cmd, false, 0, 0);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 3);

		if (!freeze_camera)
			build_depth_pyramid(cmd);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

		vkCmdFillBuffer(cmd, render_scene.dispatch_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, render_scene.cluster_count_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, render_scene.draw_indirect_buffer.buffer, 0, VK_WHOLE_SIZE, 0);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 4);
		execute_compute_cull(cmd, render_scene.opaque_pass, forward_mesh_cull_data, true, 0);

		if (CVAR_TOGGLE_MESH_SHADING.get())
		{
			vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

			ShaderPass current_pass = *shader_passes["task_submit"];
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

			auto addr = get_buffer_address(device, render_scene.dispatch_buffer.buffer);

			vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkDeviceAddress), &addr);
			vkCmdDispatch(cmd, 1, 1, 1); // TODO: task submit - currently redundant, but may come useful as renderer becomes more complex

			vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);

			execute_compute_cull(cmd, render_scene.opaque_pass, forward_cluster_cull_data, render_scene.dispatch_buffer.buffer, 0, true, 0);
		}

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 5);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

		if (!freeze_camera)
		{
			// last use was for building hi-z, transitioning back as depth attachment
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

		// for sampling/debugging hi-z
		vkutil::transition_image(
		    cmd,
		    depth_pyramid.image,
		    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
		    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
		    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
		    VK_IMAGE_ASPECT_COLOR_BIT
		);

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 6);
		render(cmd, true, 0, 1);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 7);
	}

	// third pass - masked geometry; we are still using the same hi-z for culling here. we could build an updated hi-z.
	if (CVAR_TOGGLE_MASK.get())
	{
		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

		vkCmdFillBuffer(cmd, render_scene.dispatch_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, render_scene.cluster_count_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, render_scene.draw_indirect_buffer.buffer, 0, VK_WHOLE_SIZE, 0);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 8);
		execute_compute_cull(cmd, render_scene.mask_pass, forward_mesh_cull_data, true, 1);

		if (CVAR_TOGGLE_MESH_SHADING.get())
		{
			vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

			ShaderPass current_pass = *shader_passes["task_submit"];
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

			auto addr = get_buffer_address(device, render_scene.dispatch_buffer.buffer);

			vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkDeviceAddress), &addr);
			vkCmdDispatch(cmd, 1, 1, 1); // TODO: task submit - currently redundant, but may come useful as renderer becomes more complex

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
	if (CVAR_TOGGLE_TRANSPARENT.get())
	{
		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

		vkCmdFillBuffer(cmd, render_scene.dispatch_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, render_scene.cluster_count_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, render_scene.draw_indirect_buffer.buffer, 0, VK_WHOLE_SIZE, 0);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 16);
		execute_compute_cull(cmd, render_scene.transparent_pass, forward_mesh_cull_data, true, 2);

		if (CVAR_TOGGLE_MESH_SHADING.get())
		{
			vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

			ShaderPass current_pass = *shader_passes["task_submit"];
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

			auto addr = get_buffer_address(device, render_scene.dispatch_buffer.buffer);

			vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkDeviceAddress), &addr);
			vkCmdDispatch(cmd, 1, 1, 1); // TODO: task submit - currently redundant, but may come useful as renderer becomes more complex

			vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);

			execute_compute_cull(cmd, render_scene.transparent_pass, forward_cluster_cull_data, render_scene.dispatch_buffer.buffer, 0, true, 2);
		}

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 17);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

		// is this necessary - ensure depth image in use is final; skipping this worked ok, not sure if pixel interlock interference compensates for it
		// vkutil::transition_image(
		//	cmd,
		//	depth_image.image,
		//	VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
		//	VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
		//	VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
		//	VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
		//	VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		//	VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
		//	VK_IMAGE_ASPECT_DEPTH_BIT
		//);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT); // barrier for OIT buffer from last frame?

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

	// necessary barrier for either visualizing hi-z or deferred shading
	vkutil::transition_image(
	    cmd,
	    draw_image.image,
	    VK_IMAGE_LAYOUT_UNDEFINED,
	    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	    VK_PIPELINE_STAGE_2_BLIT_BIT,
	    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
	    VK_ACCESS_2_TRANSFER_READ_BIT,
	    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
	); // from prev frame's blit to swapchain

	// light culling
	if (CVAR_TOGGLE_LIGHT_CULLING.get())
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

	// shadow
	if (CVAR_TOGGLE_SHADOW.get())
	{
		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

		vkCmdFillBuffer(cmd, render_scene.dispatch_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		// vkCmdFillBuffer(cmd, render_scene.cluster_count_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, render_scene.draw_indirect_buffer.buffer, 0, VK_WHOLE_SIZE, 0);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 20);
		execute_shadow_cull(cmd);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 21);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, // | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
		                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

		for (auto& i : cascade_data)
		{
			vkutil::transition_image(cmd, i.shadow_map.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
		}

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 22);
		uint32_t q = 4;
		for (size_t i = 0; i < cascade_data.size(); i++)
		{
			render_shadows(cmd, static_cast<uint32_t>(i), q);
			q++;
		}
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 23);

		for (auto& i : cascade_data)
		{
			vkutil::transition_image(cmd, i.shadow_map.image, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
		}
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

	// visualize hi-z
	if (CVAR_TOGGLE_DEPTH_PYRAMID.get())
	{
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 14);
		execute_debug_pass(cmd);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 15);
	}
	else // deferred shading; TODO - split this up instead of in an if block, currently sharing timestamp with if block
	{
		for (int i = 0; i < GBUFFER_COUNT; i++)
		{
			vkutil::transition_image(
			    cmd,
			    gbuffers[i].image,
			    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
			    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
			    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
			    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
			);
		}

		vkutil::transition_image(
		    cmd,
		    depth_image.image,
		    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
		    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
		    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
		    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
		    VK_IMAGE_ASPECT_DEPTH_BIT
		);

		if (CVAR_TOGGLE_TRANSPARENT.get())
		{
			vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT); // barrier for OIT buffer
		}

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame_query_pool_timestamps, 14);
		execute_deferred_shading(cmd);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame_query_pool_timestamps, 15);
	}

	vkutil::transition_image(
	    cmd,
	    draw_image.image,
	    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
	    VK_PIPELINE_STAGE_2_BLIT_BIT,
	    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
	    VK_ACCESS_2_TRANSFER_READ_BIT
	);

	vkutil::transition_image(
	    cmd,
	    swapchain_images[swapchain_image_idx],
	    VK_IMAGE_LAYOUT_UNDEFINED,
	    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	    0,
	    VK_PIPELINE_STAGE_2_BLIT_BIT,
	    0,
	    VK_ACCESS_2_TRANSFER_WRITE_BIT
	);

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
		if (RENDER_IMGUI)
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
	present_info.pWaitSemaphores = &get_current_frame().render_semaphore;
	present_info.swapchainCount = 1;
	present_info.pSwapchains = &swapchain;
	present_info.pImageIndices = &swapchain_image_idx;

	VK_CHECK(vkQueuePresentKHR(graphics_queue, &present_info));
	// FrameMark;
	frame_number++;
}

/*
void VulkanEngine::init_precomputations()
{
    //> draw
    VkCommandBuffer cmd = imm_command_buffer;
    VK_CHECK(vkResetFences(device, 1, &imm_fence));

    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    VkCommandBufferBeginInfo cmd_begin_info = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));

    vkutil::transition_image(
        cmd,
        cubemap_image.image,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        0,
        VK_ACCESS_2_SHADER_WRITE_BIT
    );

    //> cubemap pass
    ShaderPass current_pass = *shader_passes["equi_to_cube"];

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &bindless_image_descriptor, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_tex_descriptor, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_sampler_descriptor, 0, nullptr);
    IBLPushConstants pc{};
    pc.texture_id = bindless_texture.equi;
    pc.image_id = bindless_image.skybox;
    vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(IBLPushConstants), &pc);
    vkCmdDispatch(cmd, static_cast<uint32_t>(std::ceil(cubemap_image.extent.width / 16.0)), static_cast<uint32_t>(std::ceil(cubemap_image.extent.height / 16.0)), 1);

    vkutil::transition_image(
        cmd,
        cubemap_image.image,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT
    );

    vkutil::generate_mipmaps(cmd, cubemap_image.image, { cubemap_image.extent.width, cubemap_image.extent.height }, 6);

    vkutil::transition_image(
        cmd,
        cubemap_image.image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_ACCESS_2_SHADER_READ_BIT
    );

    //> irradiance pass
    current_pass = *shader_passes["irradiance"];

    vkutil::transition_image(
        cmd,
        irradiance_image.image,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        0,
        VK_ACCESS_2_SHADER_WRITE_BIT
    );

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
    pc.texture_id = bindless_texture.skybox;
    pc.image_id = bindless_image.irradiance;
    vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(IBLPushConstants), &pc);
    vkCmdDispatch(cmd, static_cast<uint32_t>(std::ceil(irradiance_image.extent.width / 8.0)), static_cast<uint32_t>(std::ceil(irradiance_image.extent.height / 8.0)), 1);

    vkutil::transition_image(
        cmd,
        irradiance_image.image,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_ACCESS_2_SHADER_READ_BIT
    );

    //> prefiltered pass
    current_pass = *shader_passes["prefiltered"];

    vkutil::transition_image(
        cmd,
        prefiltered_image.image,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        0,
        VK_ACCESS_2_SHADER_WRITE_BIT
    );

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

    const int mip_level = static_cast<int>(std::floor(std::log2(std::max(prefiltered_image.extent.width, prefiltered_image.extent.height)))) + 1;
    pc.texture_id = bindless_texture.skybox;
    for (int mip = 0; mip < mip_level; mip++)
    {
        pc.image_id = bindless_image.prefiltered + mip;
        pc.roughness = static_cast<float>(mip) / static_cast<float>(mip_level - 1);
        vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(IBLPushConstants), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(std::ceil((prefiltered_image.extent.width >> mip) / 8.0)), static_cast<uint32_t>(std::ceil((prefiltered_image.extent.height >> mip) / 8.0)), 1);
    }

    vkutil::transition_image(
        cmd,
        prefiltered_image.image,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_ACCESS_2_SHADER_READ_BIT
    );

    //> brdf pass
    current_pass = *shader_passes["brdf"];

    vkutil::transition_image(
        cmd,
        brdflut_image.image,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        0,
        VK_ACCESS_2_SHADER_WRITE_BIT
    );

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

    pc.image_id = bindless_image.brdf;
    vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(IBLPushConstants), &pc);
    vkCmdDispatch(cmd, static_cast<uint32_t>(std::ceil(brdflut_image.extent.width / 8.0)), static_cast<uint32_t>(std::ceil(brdflut_image.extent.height / 8.0)), 1);

    vkutil::transition_image(
        cmd,
        brdflut_image.image,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_ACCESS_2_SHADER_READ_BIT
    );

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmd_info = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmd_info, nullptr, nullptr);

    VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit, imm_fence));
    VK_CHECK(vkWaitForFences(device, 1, &imm_fence, true, 9999999999));
}
*/

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
					RENDER_IMGUI = !RENDER_IMGUI;
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

		freeze_camera = CVAR_TOGGLE_FREEZE.get();

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplSDL3_NewFrame();
		ImGui::NewFrame();

		// ImGui::ShowDemoWindow();
		CVarSystem::get()->draw_imgui_editor();

		{
			ImGui::Begin("Stats");
			ImGui::Text("Frametime:            %.3f ms", stats.deltatime * 1000.0f);
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

	// vulkan 1.0 features
	VkPhysicalDeviceFeatures features10{};
	features10.multiDrawIndirect = true;
	features10.pipelineStatisticsQuery = true;
	// features10.samplerAnisotropy = true;
	features10.depthClamp = true;

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

	vkGetPhysicalDeviceProperties(chosen_gpu, &props);
	assert(props.limits.timestampComputeAndGraphics);
}

void VulkanEngine::init_swapchain()
{
	create_swapchain(window_extent.width, window_extent.height);

	VkExtent3D draw_image_extent{ window_extent.width, window_extent.height, 1 };

	// TODO: after deferred - transfer_src & general only?
	VkImageUsageFlags draw_image_flags{
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | // this is now redundant?
		VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
		VK_IMAGE_USAGE_SAMPLED_BIT // for post FX sampling
	};

	draw_image = create_image(device, allocator, draw_image_extent, VK_FORMAT_R16G16B16A16_SFLOAT, draw_image_flags, VK_IMAGE_ASPECT_COLOR_BIT);

	auto id = texture_cache.add_texture(draw_image.view);
	assert(id == 0); // TODO: remove hardcoding drawimage1 to have texture id 0
	texture_cache.set_draw_image(id);

	// TODO: refactor prob necessary after implementing window/swapchain resize
	draw_extent.width = draw_image.extent.width;
	draw_extent.height = draw_image.extent.height;

	VkImageUsageFlags gbuffer_flags{
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
		VK_IMAGE_USAGE_SAMPLED_BIT // for deferred shading
	};

	visibility_buffer = create_image(device, allocator, draw_image_extent, VK_FORMAT_R32G32_UINT, gbuffer_flags, VK_IMAGE_ASPECT_COLOR_BIT);
	texture_cache.add_texture(visibility_buffer.view);

	// TODO: correct vk format and image aspect for every gbuffer?
	std::vector<VkFormat> gbuffer_formats = { VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8_SNORM };
	for (int i = 0; i < GBUFFER_COUNT; i++)
	{
		gbuffers.emplace_back(create_image(device, allocator, draw_image_extent, gbuffer_formats[i], gbuffer_flags, VK_IMAGE_ASPECT_COLOR_BIT));

		id = texture_cache.add_texture(gbuffers[i].view);
		if (i == 0)
		{
			texture_cache.set_gbuffers(id);
		}
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
		vkDestroyImageView(device, depth_image.view, nullptr);
		vmaDestroyImage(allocator, depth_image.image, depth_image.allocation);

		for (int i = 0; i < GBUFFER_COUNT; i++)
		{
			vkDestroyImageView(device, gbuffers[i].view, nullptr);
			vmaDestroyImage(allocator, gbuffers[i].image, gbuffers[i].allocation);
		} });
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
	                                  { vkDestroyCommandPool(device, imm_command_pool, nullptr); });
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
		{ VK_DESCRIPTOR_TYPE_SAMPLER, 10 },
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
		vkDestroyDescriptorSetLayout(device, bindless_image_layout, nullptr); });
}

void VulkanEngine::init_pipelines()
{
	// compute pipeline
	shader_cache.add_shader(device, "cluster_grid.comp", VK_SHADER_STAGE_COMPUTE_BIT);
	shader_cache.add_shader(device, "light_culling.comp", VK_SHADER_STAGE_COMPUTE_BIT);
	shader_cache.add_shader(device, "depth_pyramid.comp", VK_SHADER_STAGE_COMPUTE_BIT);
	shader_cache.add_shader(device, "mesh_cull.comp", VK_SHADER_STAGE_COMPUTE_BIT);
	shader_cache.add_shader(device, "meshlet_cull.comp", VK_SHADER_STAGE_COMPUTE_BIT);
	shader_cache.add_shader(device, "task_submit.comp", VK_SHADER_STAGE_COMPUTE_BIT);
	shader_cache.add_shader(device, "shadow_cull.comp", VK_SHADER_STAGE_COMPUTE_BIT);

	// graphics pipeline
	shader_cache.add_shader(device, "mesh.vert", VK_SHADER_STAGE_VERTEX_BIT);
	shader_cache.add_shader(device, "geometry.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
	shader_cache.add_shader(device, "meshlet.mesh.glsl", VK_SHADER_STAGE_MESH_BIT_EXT);
	shader_cache.add_shader(device, "full_screen.vert", VK_SHADER_STAGE_VERTEX_BIT);
	shader_cache.add_shader(device, "deferred.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
	shader_cache.add_shader(device, "debug.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
	shader_cache.add_shader(device, "mlab.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
	shader_cache.add_shader(device, "depth.vert", VK_SHADER_STAGE_VERTEX_BIT);
	shader_cache.add_shader(device, "depth.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

	// vis buffer
	shader_cache.add_shader(device, "vis_buffer.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
	shader_cache.add_shader(device, "vis_deferred.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
	shader_cache.add_shader(device, "vis_meshlet.mesh.glsl", VK_SHADER_STAGE_MESH_BIT_EXT);
#ifdef NDEBUG
	fmt::println("running Release mode"); // ensuring no clion shenanigans
#else
	fmt::println("running Debug mode");
#endif


	std::vector<VkDescriptorSetLayout> descriptor_layouts{};

	ComputePipelineBuilder compute_builder{};
	PipelineBuilder builder{};

	shader_passes["cluster_grid"] = vkutil::build_shader(device, compute_builder, shader_cache["cluster_grid.comp"], descriptor_layouts, sizeof(ClusterGridPushConstants));
	shader_passes["light_culling"] = vkutil::build_shader(device, compute_builder, shader_cache["light_culling.comp"], descriptor_layouts, sizeof(LightCullingPushConstants));
	shader_passes["task_submit"] = vkutil::build_shader(device, compute_builder, shader_cache["task_submit.comp"], descriptor_layouts, sizeof(VkDeviceAddress)); // TODO: redundant?

	descriptor_layouts = { bindless_image_layout, bindless_tex_layout, bindless_sampler_layout };
	shader_passes["depth_pyramid"] = vkutil::build_shader(device, compute_builder, shader_cache["depth_pyramid.comp"], descriptor_layouts, sizeof(DepthPyramidPushConstants));
	shader_passes["mesh_cull"] = vkutil::build_shader(device, compute_builder, shader_cache["mesh_cull.comp"], descriptor_layouts, sizeof(CullData));
	shader_passes["meshlet_cull"] = vkutil::build_shader(device, compute_builder, shader_cache["meshlet_cull.comp"], descriptor_layouts, sizeof(ClusterCullData)); // TODO: check if this is also culldata

	descriptor_layouts.clear();
	descriptor_layouts = { scene_descriptor_layout };
	shader_passes["shadow_cull"] = vkutil::build_shader(device, compute_builder, shader_cache["shadow_cull.comp"], descriptor_layouts, sizeof(ShadowCullPushConstants));

	// mrt
	descriptor_layouts.clear();
	descriptor_layouts = { scene_descriptor_layout, bindless_tex_layout, bindless_sampler_layout };

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
	builder.shader_stages[1].pSpecializationInfo = &specialization_info;
	specialization_data.opaque = 1;
	builder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["geometry_vert"] = vkutil::build_shader(device, builder, {}, descriptor_layouts, sizeof(GPUPushConstants));
	specialization_data.opaque = 0;
	builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["geometry_vert_mask"] = vkutil::build_shader(device, builder, {}, descriptor_layouts, sizeof(GPUPushConstants));

	builder.set_shaders({ shader_cache["meshlet.mesh.glsl"], shader_cache["geometry.frag"] });
	builder.shader_stages[1].pSpecializationInfo = &specialization_info;
	specialization_data.opaque = 1;
	builder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["geometry_mesh"] = vkutil::build_shader(device, builder, {}, descriptor_layouts, sizeof(GPUPushConstants));
	specialization_data.opaque = 0;
	builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["geometry_mesh_mask"] = vkutil::build_shader(device, builder, {}, descriptor_layouts, sizeof(GPUPushConstants));

	color_attachment_formats.clear();
	color_attachment_formats.push_back(visibility_buffer.format);
	builder.set_color_attachment_format(color_attachment_formats);
	color_blend_states.clear();
	color_blend_states.push_back(builder.disable_blending());
	builder.set_blending_state(color_blend_states);
	builder.set_shaders({ shader_cache["vis_meshlet.mesh.glsl"], shader_cache["vis_buffer.frag"] });
	builder.shader_stages[1].pSpecializationInfo = &specialization_info;
	specialization_data.opaque = 1;
	builder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["visibility_mesh"] = vkutil::build_shader(device, builder, {}, descriptor_layouts, sizeof(GPUPushConstants));
	specialization_data.opaque = 0;
	builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["visibility_mesh_mask"] = vkutil::build_shader(device, builder, {}, descriptor_layouts, sizeof(GPUPushConstants));

	// single render target
	descriptor_layouts.clear();
	descriptor_layouts = { scene_descriptor_layout, bindless_tex_layout, bindless_sampler_layout };

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
	builder.shader_stages[1].pSpecializationInfo = &specialization_info;
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
	// color_blend_states.clear();
	// color_blend_states.push_back(builder.disable_blending());
	// builder.set_blending_state(color_blend_states);
	builder.disable_depth();
	builder.set_depth_format(VK_FORMAT_UNDEFINED);
	builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["deferred"] = vkutil::build_shader(device, builder, { shader_cache["full_screen.vert"], shader_cache["deferred.frag"] }, descriptor_layouts, sizeof(DeferredPushConstants));
	shader_passes["vis_deferred"] = vkutil::build_shader(device, builder, { shader_cache["full_screen.vert"], shader_cache["vis_deferred.frag"] }, descriptor_layouts, sizeof(DeferredPushConstants));

	// builder.disable_depth();
	// builder.set_depth_format(VK_FORMAT_UNDEFINED);
	// builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["hi_z"] = vkutil::build_shader(device, builder, { shader_cache["full_screen.vert"], shader_cache["debug.frag"] }, descriptor_layouts, sizeof(DebugPushConstants));

	builder.enable_depth(false, VK_COMPARE_OP_GREATER_OR_EQUAL);
	color_attachment_formats.clear();
	color_attachment_formats.push_back(VK_FORMAT_UNDEFINED);
	builder.set_color_attachment_format(color_attachment_formats);
	builder.set_depth_format(depth_image.format);
	builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	shader_passes["mlab_vert"] = vkutil::build_shader(device, builder, { shader_cache["mesh.vert"], shader_cache["mlab.frag"] }, descriptor_layouts, sizeof(GPUPushConstants));
	shader_passes["mlab_mesh"] = vkutil::build_shader(device, builder, { shader_cache["meshlet.mesh.glsl"], shader_cache["mlab.frag"] }, descriptor_layouts, sizeof(GPUPushConstants));

	for (const auto& v : std::views::values(shader_cache.data))
	{
		vkDestroyShaderModule(device, v.get()->module, nullptr);
	}
}

AllocatedImage VulkanEngine::create_cubemap(VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags /*= 0*/, bool mipmapped /*= false*/)
{
	AllocatedImage new_image{};
	new_image.extent = extent;
	new_image.format = format;

	VkImageCreateInfo img_info{ vkinit::image_create_info(format, usage, new_image.extent) };
	img_info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
	img_info.arrayLayers = 6;

	if (mipmapped)
	{
		img_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(std::max(extent.width, extent.height))))) + 1;
		img_info.usage |= (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT); // TODO: possible refactor - prefiltered cubemap wont need these
	}

	VmaAllocationCreateInfo alloc_info{};
	alloc_info.flags = flags;
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
	alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	VK_CHECK(vmaCreateImage(allocator, &img_info, &alloc_info, &new_image.image, &new_image.allocation, nullptr));

	VkImageViewCreateInfo img_view_info{ vkinit::imageview_create_info(format, new_image.image, VK_IMAGE_ASPECT_COLOR_BIT) };
	img_view_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;

	VK_CHECK(vkCreateImageView(device, &img_view_info, nullptr, &new_image.view));

	return new_image;
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

	error_image = upload_image(device, graphics_queue, imm_command_buffer, imm_fence, allocator, static_cast<void*>(pixels.data()), VkExtent3D{ 16, 16, 1 }, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

	texture_cache.add_texture(error_image.view);

	VkSampler sampler{};
	VkSamplerCreateInfo sampler_info{};
	sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sampler_info.magFilter = VK_FILTER_LINEAR;
	sampler_info.minFilter = VK_FILTER_LINEAR;

	sampler_info.maxLod = VK_LOD_CLAMP_NONE;
	sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

	// sampler_info.anisotropyEnable = VK_TRUE;
	// sampler_info.maxAnisotropy = 16.0f;

	vkCreateSampler(device, &sampler_info, nullptr, &sampler); // linear
	sampler_cache.add_sampler(sampler);

	// sampler_info.anisotropyEnable = VK_FALSE;
	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

	vkCreateSampler(device, &sampler_info, nullptr, &sampler); // cube map sampling
	sampler_cache.add_sampler(sampler);

	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
	sampler_info.maxLod = 1.0;

	vkCreateSampler(device, &sampler_info, nullptr, &sampler); // shadow map sampler - potentially problematic, clamp to edge?
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

	vkCreateSampler(device, &sampler_info, nullptr, &sampler); // building hi-z
	sampler_cache.add_sampler(sampler);

	sampler_info.pNext = nullptr;
	sampler_info.magFilter = VK_FILTER_NEAREST;
	sampler_info.minFilter = VK_FILTER_NEAREST;
	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK; // reverse depth

	vkCreateSampler(device, &sampler_info, nullptr, &sampler);
	sampler_cache.add_sampler(sampler);

	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	vkCreateSampler(device, &sampler_info, nullptr, &sampler);
	sampler_cache.add_sampler(sampler); // nearest

	sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK; // reverse depth

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
		destroy_image(device, allocator, error_image);

		for (auto& cascade : cascade_data)
		{
			destroy_image(device, allocator, cascade.shadow_map);
		} });

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
		} });

	// global light list
	std::mt19937 mt(42);
	std::uniform_real_distribution<float> pos_dist(-1.0f, 1.0f);
	std::uniform_real_distribution<float> color_dist(0.f, 1.0f);

	std::vector<PointLight> light_data(LIGHT_COUNT);

	float light_area = 10.f; // in radius
	float light_radius = 1.f;

	for (size_t i = 0; i < LIGHT_COUNT; i++)
	{
		light_data[i].pos = glm::vec4(pos_dist(mt) * light_area, std::abs(pos_dist(mt) * light_area), pos_dist(mt) * light_area, light_radius); // pos & radius
		light_data[i].color = glm::vec4(color_dist(mt), color_dist(mt), color_dist(mt), 1.0);
	}

	light_data[LIGHT_COUNT - 1].pos.w = 0.0001f;

	light_buffer = upload_buffer(device, graphics_queue, imm_command_buffer, imm_fence, allocator, light_data.data(), LIGHT_COUNT * sizeof(PointLight));

	constexpr uint32_t cluster_size = 64; // TODO: hardcoded 64x64
	const uint32_t grid_x = (window_extent.width + cluster_size - 1) / cluster_size;
	const uint32_t grid_y = (window_extent.height + cluster_size - 1) / cluster_size;
	constexpr uint32_t grid_z = CLUSTER_SLICE_COUNT;
	const uint32_t total_clusters = grid_x * grid_y * grid_z;
	constexpr uint32_t max_lights_per_cluster = 10;

	light_cluster_buffer = create_buffer(allocator, total_clusters * sizeof(ClusterAABB), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
	light_index_buffer = create_buffer(allocator, total_clusters * max_lights_per_cluster * sizeof(uint32_t), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT); // could use smaller more conservative size
	light_grid_buffer = create_buffer(allocator, total_clusters * sizeof(LightGrid), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
	light_count_buffer = create_buffer(allocator, sizeof(uint32_t), 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

	main_deletion_queue.push_function([&]()
	                                  {
		destroy_buffer(allocator, light_buffer);
		destroy_buffer(allocator, light_cluster_buffer);
		destroy_buffer(allocator, light_index_buffer);
		destroy_buffer(allocator, light_grid_buffer);
		destroy_buffer(allocator, light_count_buffer); });
}

void VulkanEngine::init_renderables(std::vector<std::string>& file_paths)
{
#ifdef IBL
	/*
	const char* hdr_path{ "../../assets/pisa.hdr" };
	float* hdr_data{};

	int width{};
	int height{};
	int channels{};

	hdr_data = stbi_loadf(hdr_path, &width, &height, &channels, STBI_rgb_alpha);

	ibl_extent.width = static_cast<uint32_t>(width);
	ibl_extent.height = static_cast<uint32_t>(height);
	ibl_extent.depth = 1;

	equirectangular_image = create_image(static_cast<void*>(hdr_data), ibl_extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

	stbi_image_free(hdr_data);

	ibl_extent.width /= 4;
	ibl_extent.height = ibl_extent.width;

	cubemap_image = create_cubemap(ibl_extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, true);

	irradiance_image = create_cubemap({ 64, 64, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

	prefiltered_image = create_cubemap({ 512, 512, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, true);

	brdflut_image = create_image({ 128, 128, 1 }, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

	bindless_texture.equi = texture_cache.add_texture(equirectangular_image.view);
	bindless_texture.skybox = texture_cache.add_texture(cubemap_image.view);
	bindless_texture.irradiance = texture_cache.add_texture(irradiance_image.view);
	bindless_texture.prefiltered = texture_cache.add_texture(prefiltered_image.view);
	bindless_texture.brdf = texture_cache.add_texture(brdflut_image.view);

	scene_data.textures[0] = bindless_texture.irradiance;
	scene_data.textures[1] = bindless_texture.prefiltered;
	scene_data.textures[2] = bindless_texture.brdf;
	scene_data.textures[3] = bindless_texture.shadow;

	bindless_image.skybox = image_cache.add_texture(cubemap_image.view);
	bindless_image.irradiance = image_cache.add_texture(irradiance_image.view);

	// add each prefiltered mip level view (with all layers visible) to imagecache
	int mip_levels = int(std::floor(std::log2(std::max(prefiltered_image.extent.width, prefiltered_image.extent.height)))) + 1;
	std::vector<VkImageView> temporary_views(mip_levels);
	VkImageViewCreateInfo img_view_info = vkinit::imageview_create_info(VK_FORMAT_R32G32B32A32_SFLOAT, prefiltered_image.image, VK_IMAGE_ASPECT_COLOR_BIT);
	img_view_info.subresourceRange.levelCount = 1;
	img_view_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;

	img_view_info.subresourceRange.baseMipLevel = 0;
	vkCreateImageView(device, &img_view_info, nullptr, &temporary_views[0]);
	bindless_image.prefiltered = image_cache.add_texture(temporary_views[0]);

	for (int mip = 1; mip < mip_levels; mip++)
	{
	    img_view_info.subresourceRange.baseMipLevel = mip;
	    vkCreateImageView(device, &img_view_info, nullptr, &temporary_views[mip]);
	    image_cache.add_texture(temporary_views[mip]);
	}

	bindless_image.brdf = image_cache.add_texture(brdflut_image.view);

	main_deletion_queue.push_function([&, temporary_views]()
	                                  {
	    for (int mip = 0; mip < temporary_views.size(); mip++)
	    {
	        vkDestroyImageView(device, temporary_views[mip], nullptr);
	    } });

	main_deletion_queue.push_function([&]()
	                                  {
	    destroy_image(equirectangular_image);
	    destroy_image(cubemap_image);
	    destroy_image(irradiance_image);
	    destroy_image(prefiltered_image);
	    destroy_image(brdflut_image); });
	*/
#endif

	auto start = std::chrono::system_clock::now();
	Loader loader{};
	for (std::string& file_path : file_paths)
	{
		auto asset_file = load_gltf(this, loader, file_path);
		assert(asset_file.has_value());
		loaded_scenes[file_path] = *asset_file;
	}
	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	float ret = static_cast<float>(elapsed.count()) / 1000.0f;
	fmt::println("load gltf: {}ms", ret);

	render_scene.vertex_buffer = upload_buffer(device, graphics_queue, imm_command_buffer, imm_fence, allocator, loader.combined_vertices.data(), loader.combined_vertices.size() * sizeof(Vertex));
	render_scene.index_buffer = upload_buffer(device, graphics_queue, imm_command_buffer, imm_fence, allocator, loader.combined_indices.data(), loader.combined_indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
	render_scene.meshlet_indices = upload_buffer(device, graphics_queue, imm_command_buffer, imm_fence, allocator, loader.meshlet_indices.data(), loader.meshlet_indices.size() * sizeof(uint32_t));
	render_scene.meshlet_buffer = upload_buffer(device, graphics_queue, imm_command_buffer, imm_fence, allocator, loader.meshlets.data(), loader.meshlets.size() * sizeof(Meshlet));
	render_scene.material_buffer = upload_buffer(device, graphics_queue, imm_command_buffer, imm_fence, allocator, loader.materials.data(), loader.materials.size() * sizeof(MaterialData));

	for (const auto& scene : loaded_scenes | std::views::values)
	{
		for (const auto& n : scene->top_nodes)
		{
			register_object(n.get(), glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, 2)));
		}
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

		for (const auto& n : loaded_scenes[file_paths[0]]->top_nodes)
		{
			register_object(n.get(), transform);
		}
	}
#endif
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
	if (node->mesh != nullptr)
	{
		auto it = render_scene.mesh_cache.find(node->mesh.get());
		uint32_t handle = -1;
		bool found = it != render_scene.mesh_cache.end();
		if (found)
			handle = it->second.handle;
		else
			render_scene.mesh_cache[node->mesh.get()] = Handle<DrawPrimitive>{ static_cast<uint32_t>(render_scene.primitives.size()) };

		for (size_t i = 0; i < node->mesh->surfaces.size(); i++)
		{
			const GeoSurface& s = node->mesh->surfaces[i];

			RenderObject obj{};
			obj.transform = node_matrix;

			if (found)
			{
				obj.primitive_id.handle = static_cast<uint32_t>(handle + i);
			}
			else
			{
				obj.primitive_id.handle = static_cast<uint32_t>(render_scene.primitives.size());

				DrawPrimitive p{};
				p.center = s.bounds.origin;
				p.radius = s.bounds.radius;
				p.mesh_lods = s.mesh_lods;
				p.lod_count = s.lod_count;
				p.vertex_offset = s.vertex_offset;

				render_scene.primitives.push_back(p);
			}

			obj.material_id = s.material_id;
			obj.meshlet_bits = s.meshlet_bits;
			{
				switch (s.pass)
				{
				case MaterialPass::Mask:
					obj.post_pass = 1;
					break;
				case MaterialPass::Blend:
					obj.post_pass = 2;
					break;
				default: // Opaque
					obj.post_pass = 0;
				}
			}

			auto render_id = static_cast<uint32_t>(render_scene.renderables.size());
			render_scene.renderables.push_back(obj);

			switch (s.pass)
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

	main_camera.near = static_cast<float>(CVAR_DRAW_DISTANCE.get());
	main_camera.update(static_cast<float>(stats.deltatime));

	scene_data.view = main_camera.get_view_matrix();
	scene_data.proj = main_camera.perspective;
	scene_data.viewproj = scene_data.proj * scene_data.view;

	last_view = freeze_camera ? last_view : scene_data.view;
	last_proj = freeze_camera ? last_proj : scene_data.proj;

	scene_data.sunlight_dir = glm::vec4(7.75, 12.5, 12.5, 1.);
	// scene_data.sunlight_dir = glm::vec4(0.001, 12.0, 0.0, 1.);
	// scene_data.sunlight_dir = glm::vec4(0.0, 12.0, 12.0, 1.);
	scene_data.sunlight_color = glm::vec4(1);

	if (CVAR_TOGGLE_SHADOW.get())
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

void VulkanEngine::execute_debug_pass(VkCommandBuffer cmd)
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

	VkRenderingAttachmentInfo color_attachment = vkinit::attachment_info(draw_image.view, nullptr);
	VkRenderingInfo render_info = vkinit::rendering_info(draw_extent, &color_attachment, nullptr);

	vkCmdBeginRendering(cmd, &render_info);

	ShaderPass current_pass = *shader_passes["hi_z"];
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_sampler_descriptor, 0, nullptr);
	DebugPushConstants pc{};
	pc.texture_id = texture_cache.get_depth_pyramid_image();
	pc.lod = CVAR_DEPTH_PYRAMID_LOD.get();
	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(DebugPushConstants), &pc);
	vkCmdDraw(cmd, 3, 1, 0, 0);

	stats.draw_count++;

	vkCmdEndRendering(cmd);
}

void VulkanEngine::execute_deferred_shading(VkCommandBuffer cmd)
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

	VkRenderingAttachmentInfo color_attachment = vkinit::attachment_info(draw_image.view, nullptr);
	VkRenderingInfo render_info = vkinit::rendering_info(draw_extent, &color_attachment, nullptr);

	vkCmdBeginRendering(cmd, &render_info);

	ShaderPass current_pass{};
	bool visibility_rendering = CVAR_TOGGLE_VIS_BUFFER.get() && CVAR_TOGGLE_MESH_SHADING.get();
	if (visibility_rendering)
	{
		current_pass = *shader_passes["vis_deferred"];
	}
	else
	{
		current_pass = *shader_passes["deferred"];
	}
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_sampler_descriptor, 0, nullptr);

	DeferredPushConstants pc{};

	uint32_t cluster_x = (window_extent.width + CLUSTER_DIM - 1) / CLUSTER_DIM;
	uint32_t cluster_y = (window_extent.height + CLUSTER_DIM - 1) / CLUSTER_DIM;
	uint32_t cluster_z = CLUSTER_SLICE_COUNT; // TODO: hardcoded
	pc.cluster_size = glm::vec4(cluster_x, cluster_y, cluster_z, CLUSTER_DIM);
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

	pc.depth_id = texture_cache.get_depth_image();
	pc.albedo_id = texture_cache.get_first_gbuffer();
	pc.normal_id = pc.albedo_id + 1;
	pc.metalroughness_id = pc.albedo_id + 2; // TODO: loop based on size perhaps? remove hardcode
	pc.shadow_id = texture_cache.get_shadowmap();
	pc.light_culling = CVAR_TOGGLE_LIGHT_CULLING.get();
	pc.near = main_camera.far;

	const float ratio = main_camera.near / main_camera.far;
	pc.scale = static_cast<float>(cluster_z) / std::log(ratio);
	pc.bias = static_cast<float>(cluster_z) * std::log(main_camera.far) / std::log(ratio);
	pc.debug_meshlets = CVAR_TOGGLE_MESH_SHADING.get() ? CVAR_TOGGLE_VIEW_MESHLETS.get() : 0;
	pc.resolve_transparent = CVAR_TOGGLE_TRANSPARENT.get();
	pc.shadows = CVAR_TOGGLE_SHADOW.get();
	pc.pcf = CVAR_TOGGLE_SOFT_SHADOWS.get();
	pc.debug_shadowmap = CVAR_TOGGLE_DEBUG_SHADOWMAP.get();
	pc.debug_cascades = CVAR_TOGGLE_DEBUG_CASCADES.get();

	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(DeferredPushConstants), &pc);
	vkCmdDraw(cmd, 3, 1, 0, 0);

	stats.draw_count++;

	vkCmdEndRendering(cmd);
}

void VulkanEngine::update_cascade()
{
	// https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-10-parallel-split-shadow-maps-programmable-gpus
	float far = static_cast<float>(CVAR_SHADOW_DISTANCE.get());
	float near = main_camera.far;
	float m = static_cast<float>(NUMBER_OF_CASCADES);
	float range = far - near;
	float ratio = far / near;
	float lambda = CVAR_CSM_LAMBDA.get();

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

	// TODO: currently does not work with frozen camera
	// TODO: refactor when implementing window resize
	glm::mat4 view = main_camera.get_view_matrix();

	glm::mat4 proj = glm::perspective(glm::radians(main_camera.fov), static_cast<float>(draw_extent.width) / static_cast<float>(draw_extent.height), static_cast<float>(CVAR_SHADOW_DISTANCE.get()), main_camera.far);
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

		// TODO: try ritter's for tighter stable cascades?
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
	init_info.ApiVersion = VK_API_VERSION_1_3; // TODO: hardcoded
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
		vkDestroyDescriptorPool(device, imgui_pool, nullptr); });
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView swapchain_view)
{
	VkRenderingAttachmentInfo color_attachment = vkinit::attachment_info(swapchain_view, nullptr);
	VkRenderingInfo render_info = vkinit::rendering_info(swapchain_extent, &color_attachment, nullptr);

	vkCmdBeginRendering(cmd, &render_info);

	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

	vkCmdEndRendering(cmd);
}

void VulkanEngine::ready_mesh_draw()
{
	if (render_scene.object_buffer.info.size < render_scene.renderables.size() * sizeof(ObjectData))
	{
		fmt::println("object_buffer");
		render_scene.object_buffer = create_buffer(
		    allocator,
		    render_scene.renderables.size() * sizeof(ObjectData),
		    VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
		);

		render_scene.build_object_buffer();
	}

	if (render_scene.mesh_buffer.info.size < render_scene.primitives.size() * sizeof(DrawPrimitive))
	{
		fmt::println("mesh_buffer");
		render_scene.mesh_buffer = create_buffer(
		    allocator,
		    render_scene.primitives.size() * sizeof(DrawPrimitive),
		    VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT // ssbo usage?
		);

		render_scene.build_mesh_buffer();
	}

	std::vector<RenderScene::MeshPass*> passes = { &render_scene.opaque_pass, &render_scene.mask_pass, &render_scene.transparent_pass };

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
	if (render_scene.indices_buffer.info.size < total * sizeof(uint32_t))
		render_scene.indices_buffer = upload_buffer(device, graphics_queue, imm_command_buffer, imm_fence, allocator, staging.data(), total * sizeof(uint32_t));

	{
		if (render_scene.vis_buffer.info.size < render_scene.renderables.size()) // allocate for worst case
		{
			render_scene.vis_buffer = create_buffer(
			    allocator,
			    render_scene.renderables.size() * sizeof(uint32_t),
			    0,
			    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
			);
			fmt::println("visibility buffer: {}mb", render_scene.vis_buffer.info.size / 1e6);

			// clang-format off
			immediate_submit(device, graphics_queue, imm_command_buffer, imm_fence, [&](VkCommandBuffer cmd)
				{
					vkCmdFillBuffer(cmd, render_scene.vis_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
				}
			);
			// clang-format on
		}

		if (render_scene.dispatch_buffer.info.size < sizeof(uint32_t))
		{
			render_scene.dispatch_buffer = create_buffer(
			    allocator,
			    3 * sizeof(uint32_t),
			    0,
			    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
			);
			fmt::println("count_buffer, {}mb", render_scene.dispatch_buffer.info.size / 1e6);

			render_scene.cluster_count_buffer = create_buffer(
			    allocator,
			    3 * sizeof(uint32_t),
			    0,
			    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
			);
		}

		// if (render_scene.draw_indirect_buffer.info.size < render_scene.renderables.size() * sizeof(VkDrawIndexedIndirectCommand) + sizeof(uint32_t))
		if (render_scene.draw_indirect_buffer.info.size < ((MAX_OPAQUE_DRAWS + MAX_ALPHACLIP_DRAWS) * sizeof(VkDrawIndexedIndirectCommand) + 2 * sizeof(uint32_t)) * 4)
		{
			render_scene.draw_indirect_buffer = create_buffer(
			    allocator,
			    ((MAX_OPAQUE_DRAWS + MAX_ALPHACLIP_DRAWS) * sizeof(VkDrawIndexedIndirectCommand) + 2 * sizeof(uint32_t)) * 4,
			    0,
			    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
			);
			fmt::println("draw_indirect_buffer: {}mb", render_scene.draw_indirect_buffer.info.size / 1e6);
		}

		// TODO: resize - if we have 1m meshes with 300 clusters each = ~1.2GB buffer
		if (render_scene.cluster_indices.info.size < render_scene.total_meshlets_bits * sizeof(uint32_t))
		{
			render_scene.cluster_indices = create_buffer(
			    allocator,
			    render_scene.total_meshlets_bits * sizeof(uint32_t),
			    0,
			    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
			);

			fmt::println("cluster indices size: {}mb", static_cast<float>(render_scene.cluster_indices.info.size) / 1e6);
		}

		if (render_scene.meshtask_indirect_buffer.info.size < render_scene.max_meshtask_commands * sizeof(MeshTaskCommand))
		{
			render_scene.meshtask_indirect_buffer = create_buffer(
			    allocator,
			    render_scene.max_meshtask_commands * sizeof(MeshTaskCommand),
			    0,
			    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
			);
			fmt::println("meshtask_buffer size: {}mb", static_cast<float>(render_scene.meshtask_indirect_buffer.info.size) / 1e6);
		}

		size_t meshlet_visibility_size = (render_scene.total_meshlets_bits + 31) / 32;
		if (render_scene.meshlet_vis_buffer.info.size < meshlet_visibility_size * sizeof(uint32_t))
		{
			render_scene.meshlet_vis_buffer = create_buffer(
			    allocator,
			    meshlet_visibility_size * sizeof(uint32_t),
			    0,
			    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
			);
			fmt::println("meshlet visibility bits size: {}mb", static_cast<float>(render_scene.meshlet_vis_buffer.info.size) / 1e6);

			// clang-format off
			immediate_submit(device, graphics_queue, imm_command_buffer, imm_fence, [&](VkCommandBuffer cmd)
				{
					vkCmdFillBuffer(cmd, render_scene.meshlet_vis_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
				}
			);
			// clang-format on
		}

		// TODO: refactor if window resize
		auto screen_pixels = window_extent.width * window_extent.height;
		if (render_scene.oit_buffer.info.size < screen_pixels * sizeof(OITData))
		{
			render_scene.oit_buffer = create_buffer(
			    allocator,
			    screen_pixels * sizeof(OITData),
			    0,
			    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
			);

			fmt::println("oit buffer size: {}mb", static_cast<float>(render_scene.oit_buffer.info.size) / 1e6);

			// clang-format off
			immediate_submit(device, graphics_queue, imm_command_buffer, imm_fence, [&](VkCommandBuffer cmd)
				{
					vkCmdFillBuffer(cmd, render_scene.oit_buffer.buffer, 0, VK_WHOLE_SIZE, 0x3F800000);
				}
			);
			// clang-format on
		}
	}
}

// late & post_pass set in executecomputecull
void VulkanEngine::ready_mesh_cull(RenderScene::MeshPass& pass, CullData& cull_data, glm::mat4& proj, bool orthographic /*= false*/)
{
	auto projT = glm::transpose(proj);

	auto m0 = projT[0];
	auto m1 = projT[1];
	// auto m2 = projT[2];
	auto m3 = projT[3];

	//	m3, // near, if orthographic, m3 - m2
	//	m2, // far, if orthographic, m3 + m2
	//	m3 + m1, // bottom
	//	m3 - m1,
	//	m3 + m0, // left
	//	m3 - m0

	auto left_plane = m3 + m0;
	auto bottom_plane = m3 + m1;

	auto normalize_plane = [&](glm::vec4& plane)
	{
		float length = glm::length(glm::vec3(plane));
		plane /= length;
	};

	normalize_plane(left_plane);
	normalize_plane(bottom_plane);

	// TODO: fix
	// if (orthographic)
	//{
	//	cull_data.frustum_planes[0] = m3 - m2; // near
	//	cull_data.frustum_planes[1] = m3 + m2; // far
	//}

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
	cull_data.occlusion_enabled = CVAR_TOGGLE_OCCLUSION.get();

	cull_data.p00 = proj[0][0];
	cull_data.p11 = proj[1][1];
	cull_data.near = main_camera.far;
	cull_data.far = main_camera.near;

	cull_data.resolution = glm::vec2(depth_pyramid.extent.width, depth_pyramid.extent.height);
	cull_data.texture_lod = static_cast<float>(std::floor(std::log2(static_cast<float>(std::max(depth_pyramid.extent.width, depth_pyramid.extent.height)))) + 1);
	cull_data.lod_distance_factor = 2.0f / (cull_data.p11 * static_cast<float>(draw_extent.height));
	cull_data.lod_enabled = CVAR_TOGGLE_LOD.get();
	cull_data.task_submit = CVAR_TOGGLE_MESH_SHADING.get();
}

// late & post_pass set in executecomputecull
void VulkanEngine::ready_meshlet_cull(RenderScene::MeshPass& pass, ClusterCullData& cull_data, glm::mat4& proj, bool orthographic /*= false*/)
{
	auto projT = glm::transpose(proj);

	auto m0 = projT[0];
	auto m1 = projT[1];
	auto m2 = projT[2];
	auto m3 = projT[3];

	//	m3, // near, if orthographic, m3 - m2
	//	m2, // far, if orthographic, m3 + m2
	//	m3 + m1, // bottom
	//	m3 - m1,
	//	m3 + m0, // left
	//	m3 - m0

	auto left_plane = m3 + m0;
	auto bottom_plane = m3 + m1;

	auto normalize_plane = [&](glm::vec4& plane)
	{
		float length = glm::length(glm::vec3(plane));
		plane /= length;
	};

	normalize_plane(left_plane);
	normalize_plane(bottom_plane);

	// TODO: fix
	// if (orthographic)
	//{
	//	cull_data.frustum_planes[0] = m3 - m2; // near
	//	cull_data.frustum_planes[1] = m3 + m2; // far
	//}

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
	cull_data.occlusion_enabled = CVAR_TOGGLE_OCCLUSION.get();

	cull_data.p00 = proj[0][0];
	cull_data.p11 = proj[1][1];
	cull_data.near = main_camera.far;
	cull_data.far = main_camera.near;

	cull_data.resolution = glm::vec2(depth_pyramid.extent.width, depth_pyramid.extent.height);
	cull_data.texture_lod = static_cast<float>(std::floor(std::log2(static_cast<float>(std::max(depth_pyramid.extent.width, depth_pyramid.extent.height)))) + 1);
	cull_data.lod_distance_factor = 2.0f / (cull_data.p11 * static_cast<float>(draw_extent.height));
	cull_data.lod_enabled = CVAR_TOGGLE_LOD.get();
	cull_data.task_submit = CVAR_TOGGLE_MESH_SHADING.get();
}

void VulkanEngine::execute_compute_cull(VkCommandBuffer cmd, const RenderScene::MeshPass& pass, CullData& cull_data, bool late, uint32_t post_pass)
{
	ShaderPass current_pass = *shader_passes["mesh_cull"];
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &bindless_image_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_sampler_descriptor, 0, nullptr);

	cull_data.indices_buffer_address = get_buffer_address(device, render_scene.indices_buffer.buffer);
	cull_data.indices_buffer_address += pass.indices_offset * sizeof(uint32_t);

	cull_data.count = static_cast<uint32_t>(pass.unbatched_objects.size());
	cull_data.late = late ? 1 : 0;
	cull_data.post_pass = post_pass;

	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullData), &cull_data);
	vkCmdDispatch(cmd, static_cast<uint32_t>(std::ceil(pass.unbatched_objects.size() / 256.0f)), 1, 1);
}

void VulkanEngine::execute_compute_cull(VkCommandBuffer cmd, RenderScene::MeshPass& pass, ClusterCullData& cull_data, VkBuffer count_buffer, uint32_t offset, bool late, uint32_t post_pass)
{
	ShaderPass current_pass = *shader_passes["meshlet_cull"];
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &bindless_image_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_sampler_descriptor, 0, nullptr);

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

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

	ShadowCullPushConstants pc{};
	pc.object_buffer_address = get_buffer_address(device, render_scene.object_buffer.buffer);
	pc.mesh_buffer_address = get_buffer_address(device, render_scene.mesh_buffer.buffer);
	pc.indices_buffer_address = get_buffer_address(device, render_scene.indices_buffer.buffer);
	pc.draw_buffer_address = get_buffer_address(device, render_scene.draw_indirect_buffer.buffer);

	std::vector<RenderScene::MeshPass*> passes = { &render_scene.opaque_pass, &render_scene.mask_pass };
	for (const auto& pass : passes)
	{
		pc.count += static_cast<uint32_t>(pass->unbatched_objects.size());
	}
	pc.lod_enabled = CVAR_TOGGLE_LOD.get();

	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShadowCullPushConstants), &pc);
	vkCmdDispatch(cmd, static_cast<uint32_t>(std::ceil(pc.count / 256.0f)), 1, 1);
}

void VulkanEngine::render(VkCommandBuffer cmd, bool late, uint32_t post_pass, uint32_t query)
{
	vkCmdBeginQuery(cmd, get_current_frame().query_pool_pipelines, query, 0);

	// deferred
	VkClearColorValue clear_color_value{ 0.f, 0.f, 0.f, 1.0f };
	VkClearValue clear_value{ .color = clear_color_value };
	VkClearColorValue clear_color_value2{ 1.f, 1.f, 0.f, 1.0f };
	VkClearValue clear_value2{ .color = clear_color_value2 };

	std::vector<VkRenderingAttachmentInfo> rendering_attachment_infos{};
	bool visibility_rendering = CVAR_TOGGLE_VIS_BUFFER.get() && CVAR_TOGGLE_MESH_SHADING.get();
	if (visibility_rendering)
	{
		rendering_attachment_infos.push_back(late ? vkinit::attachment_info(visibility_buffer.view, nullptr) : vkinit::attachment_info(visibility_buffer.view, &clear_value));
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
	pc.debug_meshlets = CVAR_TOGGLE_MESH_SHADING.get() ? CVAR_TOGGLE_VIEW_MESHLETS.get() : 0;

	if (!CVAR_TOGGLE_MESH_SHADING.get())
	{
		ShaderPass current_pass = post_pass == 0 ? *shader_passes["geometry_vert"] : *shader_passes["geometry_vert_mask"];

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_tex_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_sampler_descriptor, 0, nullptr);

		vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GPUPushConstants), &pc);

		vkCmdBindIndexBuffer(cmd, render_scene.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);

		// reuse opaque section for rendering alphaClipped geometry; alphaClipped reserved for shadows
		vkCmdDrawIndexedIndirectCount(cmd, render_scene.draw_indirect_buffer.buffer, 2 * sizeof(uint32_t), render_scene.draw_indirect_buffer.buffer, 0, MAX_OPAQUE_DRAWS, sizeof(VkDrawIndexedIndirectCommand));

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
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_tex_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_sampler_descriptor, 0, nullptr);

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
	pc.debug_meshlets = CVAR_TOGGLE_MESH_SHADING.get() ? CVAR_TOGGLE_VIEW_MESHLETS.get() : 0;

	if (!CVAR_TOGGLE_MESH_SHADING.get())
	{
		ShaderPass current_pass = *shader_passes["mlab_vert"];

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_tex_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_sampler_descriptor, 0, nullptr);

		vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GPUPushConstants), &pc);

		vkCmdBindIndexBuffer(cmd, render_scene.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);

		// TODO: use a proper maxDrawCount, currently uses renderables size without checking for hardware limits
		vkCmdDrawIndexedIndirectCount(cmd, render_scene.draw_indirect_buffer.buffer, 2 * sizeof(uint32_t), render_scene.draw_indirect_buffer.buffer, 0, MAX_OPAQUE_DRAWS, sizeof(VkDrawIndexedIndirectCommand));

		stats.draw_count++;
	}
	else // mesh shading path
	{
		ShaderPass current_pass = *shader_passes["mlab_mesh"];

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_tex_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_sampler_descriptor, 0, nullptr);

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
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_tex_descriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_sampler_descriptor, 0, nullptr);

		vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ShadowPushConstants), &pc);

		vkCmdBindIndexBuffer(cmd, render_scene.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

		auto cascade_offset = cascade_idx * (2 * sizeof(uint32_t) + (MAX_OPAQUE_DRAWS + MAX_ALPHACLIP_DRAWS) * sizeof(VkDrawIndexedIndirectCommand));

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
		vkCmdDrawIndexedIndirectCount(cmd, render_scene.draw_indirect_buffer.buffer, 2 * sizeof(uint32_t) + cascade_offset, render_scene.draw_indirect_buffer.buffer, 0 + cascade_offset, MAX_OPAQUE_DRAWS, sizeof(VkDrawIndexedIndirectCommand));
		stats.draw_count++;

		if (CVAR_TOGGLE_MASK.get())
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
	vkutil::transition_image(
	    cmd,
	    depth_image.image,
	    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
	    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
	    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
	    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
	    VK_IMAGE_ASPECT_DEPTH_BIT
	);

	vkutil::transition_image(
	    cmd,
	    depth_pyramid.image,
	    VK_IMAGE_LAYOUT_UNDEFINED,
	    VK_IMAGE_LAYOUT_GENERAL,
	    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, // debugging in fragment shader
	    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
	    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	    VK_IMAGE_ASPECT_COLOR_BIT
	);

	ShaderPass current_pass = *shader_passes["depth_pyramid"];
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &bindless_image_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_sampler_descriptor, 0, nullptr);

	DepthPyramidPushConstants depth_pc{};

	uint32_t mip_levels = static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(std::max(depth_pyramid.extent.width, depth_pyramid.extent.height))))) + 1;

	for (uint32_t i = 0; i < mip_levels; i++)
	{
		int32_t workgroup_x = std::max(static_cast<int32_t>(depth_pyramid.extent.width) >> i, 1);
		int32_t workgroup_y = std::max(static_cast<int32_t>(depth_pyramid.extent.height) >> i, 1);
		depth_pc.image_size = { workgroup_x, workgroup_y };
		depth_pc.texture_id = i == 0 ? texture_cache.get_depth_image() : texture_cache.get_depth_pyramid_image();
		depth_pc.image_id = image_cache.get_depth_pyramid_image() + i;
		depth_pc.lod = i == 0 ? 0 : i - 1;

		vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DepthPyramidPushConstants), &depth_pc);
		vkCmdDispatch(cmd, (workgroup_x + 31) / 32, (workgroup_y + 31) / 32, 1);

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

	uint32_t cluster_x = (window_extent.width + CLUSTER_DIM - 1) / CLUSTER_DIM;
	uint32_t cluster_y = (window_extent.height + CLUSTER_DIM - 1) / CLUSTER_DIM;
	uint32_t cluster_z = CLUSTER_SLICE_COUNT;
	pc.cluster_size = glm::vec4(cluster_x, cluster_y, cluster_z, CLUSTER_DIM);
	pc.screen_size = glm::vec2(window_extent.width, window_extent.height);
	pc.near = main_camera.far; // reverse-z
	pc.far = main_camera.near;

	pc.light_cluster_buffer_address = get_buffer_address(device, light_cluster_buffer.buffer);

	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ClusterGridPushConstants), &pc);
	vkCmdDispatch(cmd, 1, 1, cluster_z / 2); // TODO: hardcoded to stay below maxComputeWorkGroupInvocations

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
	vkCmdDispatch(cmd, 27, 15, 24); // TODO: hardcoded
}