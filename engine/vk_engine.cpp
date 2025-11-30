#include "vk_engine.h"
#include <vk_initializers.h>
#include <vk_types.h>
#include <vk_images.h>
#include <vk_descriptors.h>
#include <vk_pipelines.h>
#include <vk_loader.h>
#include <vk_scene.h>
#include <cvars.h>

#include "tracy/Tracy.hpp"
#include "tracy/TracyVulkan.hpp"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <glm/gtx/transform.hpp>
#include "stb_image.h"

#include "glm/ext.hpp"
// for glm debug
#include "glm/gtx/string_cast.hpp"

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <SDL.h>
#include <SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include <thread>
#include <chrono>
#include <deque>
#include <memory>
#include <span>
#include <functional>
#include <cmath>
#include <utility>
#include <algorithm>
#include <random>

VulkanEngine* loaded_engine{};

VulkanEngine& VulkanEngine::get() { return *loaded_engine; }

constexpr bool USE_VALIDATION_LAYERS = true;

//#define SHADOW
//#define SINGLE

constexpr float LIGHT_FAR_PLANE{ 150.0f };
constexpr uint32_t SHADOW_MAP_SIZE{ 2048 };
constexpr int NUMBER_OF_CASCADES{ 4 };

AutoCVar_Int CVAR_SHADOW_NEAR{ "shadow.near", "pull back light frustum near plane", -20, -20, CVarFlags::EditSliderInt };
AutoCVar_Int CVAR_OCCLUSION{ "occlusion", "occlusion enabled", 1, 1, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_RENDER_PYRAMID{ "depth_pyramid.render", "render depth pyramid", 0, 0, CVarFlags::EditCheckbox };
AutoCVar_Int CVAR_DEPTH_PYRAMID_LOD{ "depth_pyramid.lod", "", 0, 0, CVarFlags::EditSliderInt };
AutoCVar_Int CVAR_DEBUG_LOD{ "LOD", "LOD enabled", 0, 0, CVarFlags::EditCheckbox};
AutoCVar_Int CVAR_SPHERE{ "visualize bounding spheres", "debug sphere", 0, 0, CVarFlags::EditCheckbox };

uint32_t nearest_pow2(uint32_t extent)
{
	return 1 << static_cast<uint32_t>(std::floor(std::log2(extent)));
}

// TODO: refactor in future
/*
void sort_transparency(const std::vector<RenderObject>& renderables, const Camera& cam, std::vector<size_t>& visible_indices)
{
	std::vector<float> distances{};
	std::vector<size_t> sorted{};

	for (size_t i = 0; i < visible_indices.size(); i++)
	{
		sorted.push_back(i);

		size_t idx = visible_indices[i];
		const RenderObject& obj = renderables[idx];
		glm::vec3 origin = glm::vec3(cam.get_view_matrix() * obj.transform * glm::vec4(obj.bounds.origin, 1.0));
		auto distance = glm::dot(origin, origin); // squared distance
		distances.push_back(distance);
	}

	std::sort(sorted.begin(), sorted.end(), [&](auto iA, auto iB) {
		float A = distances[iA];
		float B = distances[iB];

		return A > B; // render back to front
		});

	for (size_t i = 0; i < sorted.size(); i++)
	{
		auto idx = sorted[i];
		sorted[i] = visible_indices[idx];
	}

	visible_indices = std::move(sorted);
}
*/

void VulkanEngine::init()
{
	assert(loaded_engine == nullptr);
	loaded_engine = this;

	SDL_Init(SDL_INIT_VIDEO);

	SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

	window = SDL_CreateWindow(
		"Vulkan Engine",
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		window_extent.width,
		window_extent.height,
		window_flags
	);

	SDL_SetRelativeMouseMode(SDL_TRUE);

	init_vulkan();

	init_swapchain();

	init_commands();

	init_sync_structures();

	init_descriptors();

	init_pipelines();

	init_default_data();

	init_renderables();

	init_bindless();

	init_imgui();

	main_camera.position = glm::vec3(0, 0, 5);
	// (!) refactor? draw_extent set in init_default_data
	main_camera.near = 1000.0f;
	main_camera.far = 0.01f;
	main_camera.fov = 70.0f;
	main_camera.perspective = glm::perspective(glm::radians(main_camera.fov), static_cast<float>(draw_extent.width) / draw_extent.height, main_camera.near, main_camera.far);

	//init_precomputations();

	VkQueryPoolCreateInfo query_pool_info{};
	query_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
	query_pool_info.queryCount = static_cast<uint32_t>(100);
	VK_CHECK(vkCreateQueryPool(device, &query_pool_info, nullptr, &query_pool_timestamps));

	query_pool_info.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
	query_pool_info.queryCount = static_cast<uint32_t>(4);
	query_pool_info.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT;
	VK_CHECK(vkCreateQueryPool(device, &query_pool_info, nullptr, &query_pool_pipelines));

	is_initialized = true;
}

void VulkanEngine::cleanup()
{
	if (is_initialized) {

		vkDeviceWaitIdle(device);

		TracyVkDestroy(tracy_ctx);

		loaded_scenes.clear();

		for (const auto& info : sampler_cache.image_infos)
		{
			vkDestroySampler(device, info.sampler, nullptr);
		}

		for (int i = 0; i < FRAME_OVERLAP; i++)
		{
			vkDestroyCommandPool(device, frames[i].command_pool, nullptr);

			vkDestroyFence(device, frames[i].render_fence, nullptr);
			vkDestroySemaphore(device, frames[i].swapchain_semaphore, nullptr);
			vkDestroySemaphore(device, frames[i].render_semaphore, nullptr);

			destroy_buffer(frames[i].scene_buffer);

			frames[i].deletion_queue.flush();
		}
		
		destroy_buffer(render_scene.object_buffer);
		destroy_buffer(render_scene.mesh_buffer);

		std::vector<RenderScene::MeshPass*> passes = { &render_scene.forward_pass, &render_scene.transparent_pass };
		for (size_t i = 0; i < NUMBER_OF_CASCADES; i++)
		{
			passes.push_back(&render_scene.shadow_pass[i]);
		}

		for (auto* pass : passes)
		{
			auto p = *pass;
			destroy_buffer(p.draw_indirect_buffer);
			destroy_buffer(p.count_buffer);
			destroy_buffer(p.vis_buffer);
			destroy_buffer(p.debug_buffer);
			destroy_buffer(p.instance_buffer);
		}

		// (!) move destruction of combined vertex/idnex buffer here, away from loaded gltf

		for (const auto& [k, v] : shader_passes)
		{
			vkDestroyPipeline(device, v->pipeline, nullptr);
			vkDestroyPipelineLayout(device, v->layout, nullptr);
		}

		main_deletion_queue.flush();

		vkDestroyQueryPool(device, query_pool_timestamps, nullptr);
		vkDestroyQueryPool(device, query_pool_pipelines, nullptr);

		destroy_swapchain();

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyDevice(device, nullptr);

		vkb::destroy_debug_utils_messenger(instance, debug_messenger);
		vkDestroyInstance(instance, nullptr);

		SDL_DestroyWindow(window);
	}

	loaded_engine = nullptr;
}

void VulkanEngine::draw()
{
	ZoneScopedN("Draw");
	{
		ZoneScopedN("Wait render fence");
		VK_CHECK(vkWaitForFences(device, 1, &get_current_frame().render_fence, true, 1000000000));
	}
	VK_CHECK(vkResetFences(device, 1, &get_current_frame().render_fence));

	get_current_frame().deletion_queue.flush();

	SceneData* scene_uniform_data = static_cast<SceneData*>(get_current_frame().scene_buffer.info.pMappedData);
	*scene_uniform_data = scene_data;

	ready_mesh_draw();

	std::vector<RenderScene::MeshPass*> passes = { &render_scene.forward_pass };

#ifdef SHADOW
	for (size_t i = 0; i < NUMBER_OF_CASCADES; i++)
	{
		passes.push_back(&render_scene.shadow_pass[i]);
	}

	std::array<CullData, NUMBER_OF_CASCADES> shadow_cull_data{};
#endif

	CullData forward_cull_data{};

	{
		ZoneScopedN("Ready cull data");

#ifdef SHADOW
		for (size_t i = 0; i < NUMBER_OF_CASCADES; i++)
		{
			shadow_cull_data[i] = ready_cull_data(render_scene.shadow_pass[i], scene_data.shadow_transforms[i], true);
		}
#endif

		forward_cull_data = ready_cull_data(render_scene.forward_pass, scene_data.proj);
	}

	uint32_t swapchain_image_idx{};
	{
		ZoneScopedN("Acquire swapchain image");
		VK_CHECK(vkAcquireNextImageKHR(device, swapchain, 1000000000, get_current_frame().swapchain_semaphore, nullptr, &swapchain_image_idx));
	}

	VkCommandBuffer cmd = get_current_frame().main_command_buffer;

	VK_CHECK(vkResetCommandBuffer(cmd, 0));

	VkCommandBufferBeginInfo cmd_begin_info = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT); 

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));
	vkCmdResetQueryPool(cmd, query_pool_timestamps, 0, 100);
	vkCmdResetQueryPool(cmd, query_pool_pipelines, 0, 4);

	{
		//TracyVkZone(tracy_ctx, cmd, "Reset indirect buffers");
		
		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
			VK_PIPELINE_STAGE_2_CLEAR_BIT,
			VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
			VK_ACCESS_2_TRANSFER_WRITE_BIT
		);

		vkCmdFillBuffer(cmd, render_scene.forward_pass.count_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
	}

	{
		//TracyVkZone(tracy_ctx, get_current_frame().main_command_buffer, "Compute cull");

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
		); 

#ifdef SHADOW
		for (size_t i = 0; i < NUMBER_OF_CASCADES; i++)
		{
			execute_compute_cull(cmd, render_scene.shadow_pass[i], shadow_cull_data[i]);
		}
#endif
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_timestamps, 0);
		execute_compute_cull(cmd, render_scene.forward_pass, forward_cull_data, false);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_timestamps, 1);


		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
			VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT
		);

		vkutil::transition_image(
			cmd,
			depth_image.image,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, // last frame lighting pass?
			VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
			VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
			VK_IMAGE_ASPECT_DEPTH_BIT
		);

		vkutil::transition_image(
			cmd,
			draw_image.image,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			VK_PIPELINE_STAGE_2_BLIT_BIT,
			VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
			VK_ACCESS_2_TRANSFER_READ_BIT,
			VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
		); // from blit to swapchain prev frame

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_timestamps, 2);
		render(cmd, false, 0);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_timestamps, 3);

		build_depth_pyramid(cmd);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_PIPELINE_STAGE_2_CLEAR_BIT,
			VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT
		);

		vkCmdFillBuffer(cmd, render_scene.forward_pass.count_buffer.buffer, 0, VK_WHOLE_SIZE, 0);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
		);

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_timestamps, 4);
		execute_compute_cull(cmd, render_scene.forward_pass, forward_cull_data, true);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_timestamps, 5);

		vkutil::transition_buffer(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
			VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT
		);

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

		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_timestamps, 6);
		render(cmd, true, 1);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_timestamps, 7);

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

	}

	// (!) TODO: cull with AABB? ritter's?
#ifdef SHADOW
	{
		VkDependencyInfo info{};
		info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		info.imageMemoryBarrierCount = NUMBER_OF_CASCADES;

		std::vector<VkImageMemoryBarrier2> barriers{};

		for (size_t i = 0; i < NUMBER_OF_CASCADES; i++)
		{
			VkImageMemoryBarrier2 barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
			barrier.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
			barrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
			barrier.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
			barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
			barrier.subresourceRange = vkinit::image_subresource_range(VK_IMAGE_ASPECT_DEPTH_BIT);
			barrier.image = cascade_data[i].shadow_map.image;

			barriers.push_back(barrier);
		}

		info.pImageMemoryBarriers = barriers.data();
		vkCmdPipelineBarrier2(cmd, &info);

		for (size_t i = 0; i < NUMBER_OF_CASCADES; i++)
		{
			TracyVkZone(tracy_ctx, cmd, "CSM pass");
			shadow_pass(cmd, render_scene.shadow_pass[i], i);
		}

		for (size_t i = 0; i < NUMBER_OF_CASCADES; i++)
		{
			barriers[i].srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
			barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
			barriers[i].srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			barriers[i].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
			barriers[i].oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
			barriers[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}

		vkCmdPipelineBarrier2(cmd, &info);
	}
#endif
	
	{
		//TracyVkZone(tracy_ctx, get_current_frame().main_command_buffer, "Forward pass");
		forward_pass(cmd);
	}

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
		//TracyVkZone(tracy_ctx, get_current_frame().main_command_buffer, "Imgui pass")
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
	TracyVkCollect(tracy_ctx, get_current_frame().main_command_buffer);
	VK_CHECK(vkEndCommandBuffer(cmd));

	VkCommandBufferSubmitInfo cmd_info = vkinit::command_buffer_submit_info(cmd);

	VkSemaphoreSubmitInfo wait_info = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, get_current_frame().swapchain_semaphore);
	VkSemaphoreSubmitInfo submit_info = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, get_current_frame().render_semaphore); // all graphics bit?
	
	VkSubmitInfo2 submit = vkinit::submit_info(&cmd_info, &submit_info, &wait_info);

	{
		ZoneScopedN("Queue submission");
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
	FrameMark;
	frame_number++;

	std::array<uint64_t, 8> timestamp_results{};

	vkGetQueryPoolResults(
		device,
		query_pool_timestamps,
		0,
		timestamp_results.size(),
		timestamp_results.size() * sizeof(uint64_t),
		timestamp_results.data(),
		sizeof(uint64_t),
		VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
	);

	std::array<uint64_t, 2> pipeline_results{};

	vkGetQueryPoolResults(
		device,
		query_pool_pipelines,
		0,
		pipeline_results.size(),
		pipeline_results.size() * sizeof(uint64_t),
		pipeline_results.data(),
		sizeof(uint64_t),
		VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
	);

	auto cull_begin = static_cast<double>(timestamp_results[0]) * props.limits.timestampPeriod * 1e-6;
	auto cull_end   = static_cast<double>(timestamp_results[1]) * props.limits.timestampPeriod * 1e-6;
	stats.early_cull = cull_end - cull_begin;

	auto indirect_begin = static_cast<double>(timestamp_results[2]) * props.limits.timestampPeriod * 1e-6;
	auto indirect_end   = static_cast<double>(timestamp_results[3]) * props.limits.timestampPeriod * 1e-6;
	stats.early_indirect = indirect_end - indirect_begin;

	cull_begin = static_cast<double>(timestamp_results[4]) * props.limits.timestampPeriod * 1e-6;
	cull_end =   static_cast<double>(timestamp_results[5]) * props.limits.timestampPeriod * 1e-6;
	stats.late_cull = cull_end - cull_begin;

	indirect_begin = static_cast<double>(timestamp_results[6]) * props.limits.timestampPeriod * 1e-6;
	indirect_end =   static_cast<double>(timestamp_results[7]) * props.limits.timestampPeriod * 1e-6;
	stats.late_indirect = indirect_end - indirect_begin;

	stats.triangle_count = pipeline_results[0] + pipeline_results[1];
}

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

	int mip_level = int(std::floor(std::log2(std::max(prefiltered_image.extent.width, prefiltered_image.extent.height)))) + 1;
	pc.texture_id = bindless_texture.skybox;
	for (int mip = 0; mip < mip_level; mip++)
	{
		pc.image_id = bindless_image.prefiltered + mip; 
		pc.roughness = static_cast<float>(mip) / (mip_level - 1);
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

void VulkanEngine::run()
{
    SDL_Event e;
    bool bQuit = false;

	auto last_frame = std::chrono::system_clock::now();

    while (!bQuit) 
	{
		auto start = std::chrono::system_clock::now();
		auto deltatime = std::chrono::duration_cast<std::chrono::microseconds>(start - last_frame);
		stats.deltatime = deltatime.count() / 1000000.0f; // microseconds to seconds
		last_frame = start;

        // Handle events on queue
        while (SDL_PollEvent(&e) != 0) 
		{
            if (e.type == SDL_QUIT)
                bQuit = true;

            if (e.type == SDL_WINDOWEVENT) {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) 
                    stop_rendering = true;
                if (e.window.event == SDL_WINDOWEVENT_RESTORED) 
                    stop_rendering = false;
            }

			if (e.type == SDL_KEYDOWN)
			{
				if (e.key.repeat == 0 && e.key.keysym.sym == SDLK_SPACE)
				{
					stop_movement = !stop_movement;
					stop_movement ? SDL_SetRelativeMouseMode(SDL_FALSE) : SDL_SetRelativeMouseMode(SDL_TRUE);
				}
			}

			if (!stop_movement)
				main_camera.process_sdl_event(e);

			ImGui_ImplSDL2_ProcessEvent(&e);
        }

        if (stop_rendering) 
		{
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplSDL2_NewFrame();
		ImGui::NewFrame();
		
		//ImGui::ShowDemoWindow();
		CVarSystem::get()->draw_imgui_editor();

		{
			ImGui::Begin("Stats");
			ImGui::Text("frametime %f ms", stats.deltatime * 1000.0f);
			ImGui::Text("draws %i", stats.draw_count);
			ImGui::Text("scene update time %f ms", stats.scene_update_time);
			ImGui::Text("early cull %f ms", stats.early_cull);
			ImGui::Text("late  cull %f ms", stats.late_cull);
			ImGui::Text("early indirect %f ms", stats.early_indirect);
			ImGui::Text("late  indirect %f ms", stats.late_indirect);
			//ImGui::Text("triangles %.2fM", static_cast<double>(stats.triangle_count) * 1e-6);
			ImGui::Text("triangles %.2u", stats.triangle_count);

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

	SDL_Vulkan_CreateSurface(window, instance, &surface);

	// vulkan 1.3 features
	VkPhysicalDeviceVulkan13Features features13{};
	features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
	features13.dynamicRendering = true;
	features13.synchronization2 = true;

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

	// vulkan 1.0 features
	VkPhysicalDeviceFeatures features10{};
	features10.multiDrawIndirect = true;
	features10.pipelineStatisticsQuery = true;
	//features10.samplerAnisotropy = true;
	//features10.depthClamp = true;

	// use vkbootstrap to select a gpu. 
	// we want a gpu that can write to the SDL surface and supports vulkan 1.3 with the correct features
	vkb::PhysicalDeviceSelector selector{ vkb_inst };
	vkb::PhysicalDevice physicalDevice = selector
		.set_minimum_version(1, 3)
		.set_required_features(features10)
		.set_required_features_13(features13)
		.set_required_features_12(features12)
		.add_required_extension("VK_KHR_calibrated_timestamps")
		.set_surface(surface)
		.select()
		.value();

	// create the final vulkan device
	vkb::DeviceBuilder deviceBuilder{ physicalDevice };

	vkb::Device vkbDevice = deviceBuilder.build().value();

	// get the VkDevice handle used in the rest of a vulkan application
	device = vkbDevice.device;
	chosen_gpu = physicalDevice.physical_device;

	// use vkbootstrap to get a Graphics queue
	graphics_queue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
	graphics_queue_family = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

	VmaAllocatorCreateInfo allocator_info{};
	allocator_info.physicalDevice = chosen_gpu;
	allocator_info.device = device;
	allocator_info.instance = instance;
	allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT; // allows usage of GPU pointers
	vmaCreateAllocator(&allocator_info, &allocator);

	main_deletion_queue.push_function([&]() {
		vmaDestroyAllocator(allocator); 
		});

	vkGetPhysicalDeviceProperties(chosen_gpu, &props);
	assert(props.limits.timestampComputeAndGraphics);
}

void VulkanEngine::init_swapchain()
{
	create_swapchain(window_extent.width, window_extent.height);

	VkExtent3D draw_image_extent{ window_extent.width, window_extent.height, 1 };

	VkImageUsageFlags draw_image_flags{
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
		VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
		VK_IMAGE_USAGE_SAMPLED_BIT // for post FX sampling
	};

	// (!) refactor ping pong
	draw_image = create_image(draw_image_extent, VK_FORMAT_R16G16B16A16_SFLOAT, draw_image_flags, VK_IMAGE_ASPECT_COLOR_BIT);
	draw_image2 = create_image(draw_image_extent, VK_FORMAT_R16G16B16A16_SFLOAT, draw_image_flags, VK_IMAGE_ASPECT_COLOR_BIT);

	auto id = texture_cache.add_texture(draw_image.view);
	assert(id == 0); // (!) why am i checking this again?
	texture_cache.set_draw_image(id);

	id = texture_cache.add_texture(draw_image2.view);
	texture_cache.set_draw_image2(id);

	// depth image
	depth_image.format = VK_FORMAT_D32_SFLOAT;
	depth_image.extent = draw_image_extent;

	depth_image = create_image(draw_image_extent, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);

	id = texture_cache.add_texture(depth_image.view);
	texture_cache.set_depth_image(id);

	main_deletion_queue.push_function([&]() {
		vkDestroyImageView(device, draw_image.view, nullptr);
		vmaDestroyImage(allocator, draw_image.image, draw_image.allocation);
		vkDestroyImageView(device, depth_image.view, nullptr);
		vmaDestroyImage(allocator, depth_image.image, depth_image.allocation);

		vkDestroyImageView(device, draw_image2.view, nullptr);
		vmaDestroyImage(allocator, draw_image2.image, draw_image2.allocation);
	});
}

void VulkanEngine::init_commands()
{
	VkCommandPoolCreateInfo command_pool_info = vkinit::command_pool_create_info(
		graphics_queue_family,
		VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
	);

	for (int i = 0; i < FRAME_OVERLAP; i++) 
	{
		VK_CHECK(vkCreateCommandPool(device, &command_pool_info, nullptr, &frames[i].command_pool));

		VkCommandBufferAllocateInfo cmd_alloc_info = vkinit::command_buffer_allocate_info(
			frames[i].command_pool
		);

		VK_CHECK(vkAllocateCommandBuffers(device, &cmd_alloc_info, &frames[i].main_command_buffer));
	}

	auto p_timedomain = (PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR");
	auto p_timestamp = (PFN_vkGetCalibratedTimestampsEXT)vkGetInstanceProcAddr(instance, "vkGetCalibratedTimestampsKHR");
	tracy_ctx = TracyVkContextCalibrated(chosen_gpu, device, graphics_queue, frames[0].main_command_buffer, p_timedomain, p_timestamp);

	VK_CHECK(vkCreateCommandPool(device, &command_pool_info, nullptr, &imm_command_pool));
	VkCommandBufferAllocateInfo cmd_alloc_info = vkinit::command_buffer_allocate_info(
		imm_command_pool
	);
	VK_CHECK(vkAllocateCommandBuffers(device, &cmd_alloc_info, &imm_command_buffer));

	main_deletion_queue.push_function([&]() {
		vkDestroyCommandPool(device, imm_command_pool, nullptr);
	});
}

void VulkanEngine::init_sync_structures()
{
	VkFenceCreateInfo fence_info = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
	VkSemaphoreCreateInfo semaphore_info = vkinit::semaphore_create_info();

	for (size_t i = 0; i < FRAME_OVERLAP; i++)
	{
		VK_CHECK(vkCreateFence(device, &fence_info, nullptr, &frames[i].render_fence));

		VK_CHECK(vkCreateSemaphore(device, &semaphore_info, nullptr, &frames[i].swapchain_semaphore));
		VK_CHECK(vkCreateSemaphore(device, &semaphore_info, nullptr, &frames[i].render_semaphore));
	}

	VK_CHECK(vkCreateFence(device, &fence_info, nullptr, &imm_fence));

	main_deletion_queue.push_function([&]() {
		vkDestroyFence(device, imm_fence, nullptr);
	});
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height)
{
	vkb::SwapchainBuilder swapchainBuilder{ chosen_gpu, device, surface };

	swapchain_image_format = VK_FORMAT_B8G8R8A8_UNORM;
	//swapchain_image_format = VK_FORMAT_B8G8R8A8_SRGB;

	vkb::Swapchain vkbSwapchain = swapchainBuilder
		//.use_default_format_selection()
		.set_desired_format(VkSurfaceFormatKHR{ .format = swapchain_image_format, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR }) 
		//.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR) 
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

	for (size_t i = 0; i < swapchain_image_views.size(); i++) {

		vkDestroyImageView(device, swapchain_image_views[i], nullptr);
	}
}

void VulkanEngine::init_descriptors()
{
	//> building scene descriptor layout
	{
		DescriptorLayoutBuilder builder{};
		builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
		scene_descriptor_layout = builder.build(device);
	}

	std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> uniform_sizes = {
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 5},
	};

	DescriptorWriter writer{};
	for (int i = 0; i < FRAME_OVERLAP; i++)
	{
		frames[i].frame_descriptor_allocator.init(device, 1, uniform_sizes);

		main_deletion_queue.push_function([&, i]() {
			frames[i].frame_descriptor_allocator.destroy_pools(device);
		});

		frames[i].scene_buffer = create_buffer(sizeof(SceneData), VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
		);

		frames[i].scene_descriptor = frames[i].frame_descriptor_allocator.allocate(device, scene_descriptor_layout);

		writer.clear();
		writer.write_buffer(0, frames[i].scene_buffer.buffer, sizeof(SceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
		writer.update_set(device, frames[i].scene_descriptor);
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
		flags[0] = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT
			| VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
			//| VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;

		VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_info{};
		binding_flags_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
		binding_flags_info.bindingCount = 1;
		binding_flags_info.pBindingFlags = flags.data();

		bindless_tex_layout = builder.build(device, &binding_flags_info); // (!) update after bind req if included above?

		builder.clear();
		builder.add_binding(0, VK_DESCRIPTOR_TYPE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
		builder.bindings[0].descriptorCount = 10; // (!) validation layer not reporting if this is higher than pool maximum

		bindless_sampler_layout = builder.build(device, &binding_flags_info); // (!) update after bind req if included above?

		builder.clear();
		builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT);
		builder.bindings[0].descriptorCount = 1000;

		bindless_image_layout = builder.build(device, &binding_flags_info);
	}

	main_deletion_queue.push_function([&]() {
		global_descriptor_allocator.destroy_pools(device);
		vkDestroyDescriptorSetLayout(device, scene_descriptor_layout, nullptr);
		vkDestroyDescriptorSetLayout(device, bindless_tex_layout, nullptr);
		vkDestroyDescriptorSetLayout(device, bindless_sampler_layout, nullptr);
		vkDestroyDescriptorSetLayout(device, bindless_image_layout, nullptr);
	});
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& func)
{
	VK_CHECK(vkResetFences(device, 1, &imm_fence));
	VK_CHECK(vkResetCommandBuffer(imm_command_buffer, 0));

	VkCommandBuffer cmd = imm_command_buffer;

	VkCommandBufferBeginInfo cmd_begin_info = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	
	VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));

	func(cmd);

	VK_CHECK(vkEndCommandBuffer(cmd));

	VkCommandBufferSubmitInfo cmd_submit_info = vkinit::command_buffer_submit_info(cmd);

	VkSubmitInfo2 submit = vkinit::submit_info(&cmd_submit_info, nullptr, nullptr);

	VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit, imm_fence));

	VK_CHECK(vkWaitForFences(device, 1, &imm_fence, true, 9999999999)); 
}

void VulkanEngine::init_pipelines()
{
	std::vector<VkDescriptorSetLayout> descriptor_layouts{ bindless_image_layout, bindless_tex_layout, bindless_sampler_layout };
	VkPushConstantRange pc = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(IBLPushConstants) };

	//> IBL
	ComputePipelineBuilder compute_builder{};
	VkShaderModule module = shader_cache.add_shader(device, "equi_to_cube.comp.spv");
	compute_builder.set_shaders(module);
	std::unique_ptr<ShaderPass> equi_to_cube_pass = vkutil::build_shader(device, compute_builder, descriptor_layouts, &pc);

	module = shader_cache.add_shader(device, "irradiance.comp.spv");
	compute_builder.set_shaders(module);
	std::unique_ptr<ShaderPass> irradiance_pass = vkutil::build_shader(device, compute_builder, descriptor_layouts, &pc);

	module = shader_cache.add_shader(device, "prefiltered.comp.spv");
	compute_builder.set_shaders(module);
	std::unique_ptr<ShaderPass> prefiltered_pass = vkutil::build_shader(device, compute_builder, descriptor_layouts, &pc);

	module = shader_cache.add_shader(device, "brdf.comp.spv");
	compute_builder.set_shaders(module);
	std::unique_ptr<ShaderPass> brdf_pass = vkutil::build_shader(device, compute_builder, descriptor_layouts, &pc);

	module = shader_cache.add_shader(device, "depth_pyramid.comp.spv");
	compute_builder.set_shaders(module);
	pc = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DepthPyramidPushConstants) };
	std::unique_ptr<ShaderPass> depth_pyramid_pass = vkutil::build_shader(device, compute_builder, descriptor_layouts, &pc);

	//> INDIRECT CULL
	module = shader_cache.add_shader(device, "indirect_cull.comp.spv");
	compute_builder.set_shaders(module);
	pc = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullData) };
	std::unique_ptr<ShaderPass> cull_pass = vkutil::build_shader(device, compute_builder, descriptor_layouts, &pc);

	//> GRAPHICS PIPELINE
	descriptor_layouts.clear();
	descriptor_layouts = { scene_descriptor_layout, bindless_tex_layout, bindless_sampler_layout };
	VkShaderModule frag_module{};

	//> SHADOW 
	PipelineBuilder builder{};
	builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
	builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
	builder.set_multisampling_none();
	builder.disable_blending();

	module = shader_cache.add_shader(device, "depth.vert.spv");
	builder.set_shaders(module);
	builder.set_cull_mode(VK_CULL_MODE_FRONT_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	builder.enable_depth(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
	builder.set_color_attachment_format(VK_FORMAT_UNDEFINED);
	builder.set_depth_format(depth_image.format);
	builder.dynamic_state.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS);
	builder.rasterization.depthBiasEnable = VK_TRUE;
	pc = { VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ShadowPushConstants) };
	std::unique_ptr<ShaderPass> shadow_pass = vkutil::build_shader(device, builder, descriptor_layouts, &pc);

	//> DOUBLE SIDED SHADOW
	frag_module = shader_cache.add_shader(device, "depth.frag.spv");
	builder.set_shaders(module, frag_module);
	builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	std::unique_ptr<ShaderPass> shadow_flat_pass = vkutil::build_shader(device, builder, descriptor_layouts, &pc);

	//> GEOMETRY + LIGHTING
	//> DOUBLE SIDED MASK
	module = shader_cache.add_shader(device, "mesh_pbr.vert.spv");
	frag_module = shader_cache.add_shader(device, "mesh_pbr_clip.frag.spv");
	builder.set_shaders(module, frag_module);
	builder.set_color_attachment_format(draw_image.format); 
	builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	builder.disable_blending();
	builder.enable_depth(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
	builder.dynamic_state.pop_back(); // remove depth bias dynamic state
	builder.rasterization.depthBiasEnable = VK_FALSE;
	//pc = { VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants) };
	pc = { VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GPUPushConstants) }; // (!) gpu driven
	std::unique_ptr<ShaderPass> textured_lit_clip_pass = vkutil::build_shader(device, builder, descriptor_layouts, &pc);

	//> DOUBLE SIDED LIT
	//frag_module = shader_cache.add_shader(device, "mesh_pbr.frag.spv");
	frag_module = shader_cache.add_shader(device, "basic_mesh.frag.spv");
	builder.set_shaders(module, frag_module);
	std::unique_ptr<ShaderPass> textured_lit2_pass = vkutil::build_shader(device, builder, descriptor_layouts, &pc);

	//> NON-DOUBLE SIDED LIT
	builder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	std::unique_ptr<ShaderPass> textured_lit_pass = vkutil::build_shader(device, builder, descriptor_layouts, &pc);

	//> DOUBLE SIDED TRANSPARENT BACK FIRST
	frag_module = shader_cache.add_shader(device, "mesh_pbr_transparent.frag.spv");
	builder.set_shaders(module, frag_module);
	builder.enable_blending_alphablend();
	builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	builder.enable_depth(false, VK_COMPARE_OP_GREATER_OR_EQUAL);
	//pc = { VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants) }; // (!) uncomment if not gpu driven
	std::unique_ptr<ShaderPass> blend_pass = vkutil::build_shader(device, builder, descriptor_layouts, &pc);

	//> SKYBOX
	module = shader_cache.add_shader(device, "skybox.vert.spv");
	frag_module = shader_cache.add_shader(device, "skybox.frag.spv");
	builder.set_shaders(module, frag_module);
	builder.disable_blending();
	builder.enable_depth(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
	pc = { VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SkyboxPushConstants) };
	std::unique_ptr<ShaderPass> skybox_pass = vkutil::build_shader(device, builder, descriptor_layouts, &pc);

	//> DEBUG
	module = shader_cache.add_shader(device, "full_screen.vert.spv");
	frag_module = shader_cache.add_shader(device, "debug.frag.spv");
	builder.set_shaders(module, frag_module);
	builder.disable_depth();
	builder.set_depth_format(VK_FORMAT_UNDEFINED);
	pc = { VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(DebugPushConstants) };
	std::unique_ptr<ShaderPass> debug_pass = vkutil::build_shader(device, builder, descriptor_layouts, &pc);

	//> POST FX
	module = shader_cache.add_shader(device, "full_screen.vert.spv");
	frag_module = shader_cache.add_shader(device, "tonemap.frag.spv");
	builder.set_shaders(module, frag_module);
	builder.disable_depth();
	builder.set_depth_format(VK_FORMAT_UNDEFINED);
	pc = { VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PostFXPushConstants) };
	std::unique_ptr<ShaderPass> tonemap_pass = vkutil::build_shader(device, builder, descriptor_layouts, &pc);

	shader_passes["equi_to_cube"] = std::move(equi_to_cube_pass);
	shader_passes["irradiance"] = std::move(irradiance_pass);
	shader_passes["prefiltered"] = std::move(prefiltered_pass);
	shader_passes["brdf"] = std::move(brdf_pass);
	shader_passes["shadow"] = std::move(shadow_pass);
	shader_passes["shadow_flat"] = std::move(shadow_flat_pass);
	shader_passes["textured_lit_clip"] = std::move(textured_lit_clip_pass);
	shader_passes["textured_lit"] = std::move(textured_lit_pass);
	shader_passes["textured_lit2"] = std::move(textured_lit2_pass);
	shader_passes["skybox"] = std::move(skybox_pass);
	shader_passes["tonemap"] = std::move(tonemap_pass);
	shader_passes["blend"] = std::move(blend_pass);
	shader_passes["cull"] = std::move(cull_pass);
	shader_passes["debug"] = std::move(debug_pass);
	shader_passes["depth_pyramid"] = std::move(depth_pyramid_pass);

	for (const auto& [k, v] : shader_cache.data)
	{
		vkDestroyShaderModule(device, v, nullptr);
	}
}

AllocatedBuffer VulkanEngine::create_buffer(size_t alloc_size, VmaAllocationCreateFlags flags, VkBufferUsageFlags usage)
{
	VkBufferCreateInfo buffer_info{};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.size = alloc_size;
	buffer_info.usage = usage;

	VmaAllocationCreateInfo alloc_info{};
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
	alloc_info.flags = flags;

	AllocatedBuffer new_buffer{};

	VK_CHECK(vmaCreateBuffer(allocator, &buffer_info, &alloc_info, &new_buffer.buffer, &new_buffer.allocation, &new_buffer.info));

	return new_buffer;
}

AllocatedBuffer VulkanEngine::reallocate_buffer(size_t alloc_size, AllocatedBuffer old_buffer, VmaAllocationCreateFlags flags, VkBufferUsageFlags usage)
{
	AllocatedBuffer new_buffer{};
	new_buffer = create_buffer(alloc_size, flags, usage);

	get_current_frame().deletion_queue.push_function([=]() {
		destroy_buffer(old_buffer); // (!) or destroy directly?
		});

	return new_buffer;
}

void VulkanEngine::destroy_buffer(const AllocatedBuffer& buffer)
{
	vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation);
}

GPUMeshBuffers VulkanEngine::upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices)
{
	const size_t vertex_buffer_size = vertices.size() * sizeof(Vertex);
	const size_t index_buffer_size = indices.size() * sizeof(uint32_t);

	GPUMeshBuffers mesh_buffer{};

	mesh_buffer.vertex_buffer = create_buffer(vertex_buffer_size, 0,
		VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
	);

	VkBufferDeviceAddressInfo address_info{};
	address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	address_info.buffer = mesh_buffer.vertex_buffer.buffer;

	mesh_buffer.vertex_buffer_address= vkGetBufferDeviceAddress(device, &address_info);

	mesh_buffer.index_buffer = create_buffer(index_buffer_size, 0,
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
	);

	AllocatedBuffer staging = create_buffer(vertex_buffer_size + index_buffer_size,
		VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT
	);

	void* data = staging.info.pMappedData;
	memcpy(data, vertices.data(), vertex_buffer_size);
	memcpy(static_cast<char*>(data) + vertex_buffer_size, indices.data(), index_buffer_size);

	immediate_submit([&](VkCommandBuffer cmd) {
		VkBufferCopy vertex_copy{};
		vertex_copy.dstOffset = 0;
		vertex_copy.srcOffset = 0;
		vertex_copy.size = vertex_buffer_size;

		vkCmdCopyBuffer(cmd, staging.buffer, mesh_buffer.vertex_buffer.buffer, 1, &vertex_copy);

		VkBufferCopy index_copy{};
		index_copy.dstOffset = 0;
		index_copy.srcOffset = vertex_buffer_size;
		index_copy.size = index_buffer_size;

		vkCmdCopyBuffer(cmd, staging.buffer, mesh_buffer.index_buffer.buffer, 1, &index_copy);
	});

	destroy_buffer(staging);

	return mesh_buffer;
}

AllocatedImage VulkanEngine::create_image(VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags /*= 0*/, bool mipmapped /*= false*/)
{
	AllocatedImage new_image{};
	new_image.extent = extent;
	new_image.format = format;

	VkImageCreateInfo img_info = vkinit::image_create_info(format, usage, extent);
	if (mipmapped)
	{
		img_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(extent.width, extent.height)))) + 1;
		img_info.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT; 
	}

	VmaAllocationCreateInfo alloc_info{};
	alloc_info.flags = flags;
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
	alloc_info.requiredFlags = VkMemoryPropertyFlagBits(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	VK_CHECK(vmaCreateImage(allocator, &img_info, &alloc_info, &new_image.image, &new_image.allocation, nullptr));

	VkImageViewCreateInfo img_view_info = vkinit::imageview_create_info(format, new_image.image, aspect);

	VK_CHECK(vkCreateImageView(device, &img_view_info, nullptr, &new_image.view));

	return new_image;
}

// currently used for HDR, png and jpg, NOT ktx2
AllocatedImage VulkanEngine::create_image(void* data, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags /*= 0*/, bool mipmapped /*= false*/)
{
	size_t data_size = extent.depth * extent.width * extent.height * 4; // 4 is # of channels
	if (format == VK_FORMAT_R32G32B32A32_SFLOAT) // (!) review - hdr?
		data_size *= sizeof(float);
	AllocatedBuffer upload_buffer = create_buffer(data_size, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
	
	memcpy(upload_buffer.info.pMappedData, data, data_size);

	// dst_bit to account for copy from staging buffer
	AllocatedImage new_image = create_image(extent, format, usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT, aspect, flags, mipmapped);

	immediate_submit([&](VkCommandBuffer cmd) {
		vkutil::transition_image(
			cmd, new_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			0,
			VK_PIPELINE_STAGE_2_TRANSFER_BIT,
			0,
			VK_ACCESS_2_TRANSFER_WRITE_BIT
		);

		VkBufferImageCopy copy_region{};
		copy_region.bufferOffset = 0;
		copy_region.bufferRowLength = 0;
		copy_region.bufferImageHeight = 0;

		copy_region.imageSubresource.aspectMask = aspect;
		copy_region.imageSubresource.mipLevel = 0;
		copy_region.imageSubresource.baseArrayLayer = 0;
		copy_region.imageSubresource.layerCount = 1;

		copy_region.imageExtent = extent;

		vkCmdCopyBufferToImage(cmd, upload_buffer.buffer, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

		if (mipmapped)
		{
			vkutil::generate_mipmaps(cmd, new_image.image, VkExtent2D{ new_image.extent.width, new_image.extent.height });
		}
		else // only works for textures to be sampled in fragment shader
		{
			vkutil::transition_image(
				cmd, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_2_TRANSFER_BIT,
				VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
				VK_ACCESS_2_TRANSFER_WRITE_BIT,
				VK_ACCESS_2_SHADER_READ_BIT
			);
		}
		});

	destroy_buffer(upload_buffer);

	return new_image;
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
		img_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(extent.width, extent.height)))) + 1;
		img_info.usage |= (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT); // (!) prefiltered cubemap wont need these
	}

	VmaAllocationCreateInfo alloc_info{};
	alloc_info.flags = flags;
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
	alloc_info.requiredFlags = VkMemoryPropertyFlagBits(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	VK_CHECK(vmaCreateImage(allocator, &img_info, &alloc_info, &new_image.image, &new_image.allocation, nullptr));

	VkImageViewCreateInfo img_view_info{ vkinit::imageview_create_info(format, new_image.image, VK_IMAGE_ASPECT_COLOR_BIT) };
	img_view_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;

	VK_CHECK(vkCreateImageView(device, &img_view_info, nullptr, &new_image.view));

	return new_image;
}

void VulkanEngine::destroy_image(const AllocatedImage& image)
{
	if (image.view == nullptr)
		fmt::println("was null");
	vkDestroyImageView(device, image.view, nullptr);
	vmaDestroyImage(allocator, image.image, image.allocation);
}

void VulkanEngine::init_default_data()
{
	// (!) refactor prob necessary after implementing window/swapchain resize
	draw_extent.width = draw_image.extent.width;
	draw_extent.height = draw_image.extent.height;

	//> default textures

	uint32_t magenta_color = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
	std::array<uint32_t, 16 * 16 > pixels{}; //for 16x16 checkerboard texture
	for (int x = 0; x < 16; x++) 
	{
		for (int y = 0; y < 16; y++) 
		{
			pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta_color : 0;
		}
	}

	error_image = create_image(static_cast<void*>(pixels.data()), VkExtent3D{ 16, 16, 1 },
		VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT);

	bindless_texture.checkerboard = texture_cache.add_texture(error_image.view);

	//> default samplers
	VkSampler sampler{};
	VkSamplerCreateInfo sampler_info{};
	sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sampler_info.magFilter = VK_FILTER_LINEAR;
	sampler_info.minFilter = VK_FILTER_LINEAR;

	sampler_info.maxLod = VK_LOD_CLAMP_NONE;
	sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

	//sampler_info.anisotropyEnable = VK_TRUE;
	//sampler_info.maxAnisotropy = 16.0f;

	vkCreateSampler(device, &sampler_info, nullptr, &sampler);
	sampler_cache.add_sampler(sampler);

	//sampler_info.anisotropyEnable = VK_FALSE;
	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

	vkCreateSampler(device, &sampler_info, nullptr, &sampler);
	sampler_cache.add_sampler(sampler);

	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK; // reverse depth

	vkCreateSampler(device, &sampler_info, nullptr, &sampler);
	sampler_cache.add_sampler(sampler);

	sampler_info.magFilter = VK_FILTER_NEAREST;
	sampler_info.minFilter = VK_FILTER_NEAREST;
	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler_info.borderColor = {};

	vkCreateSampler(device, &sampler_info, nullptr, &sampler);
	sampler_cache.add_sampler(sampler);

	sampler_info.magFilter = VK_FILTER_LINEAR;
	sampler_info.minFilter = VK_FILTER_LINEAR;
	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

	VkSamplerReductionModeCreateInfo reduction_info{};
	reduction_info.sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO;
	reduction_info.reductionMode = VK_SAMPLER_REDUCTION_MODE_MIN;

	sampler_info.pNext = &reduction_info;

	vkCreateSampler(device, &sampler_info, nullptr, &sampler);
	sampler_cache.add_sampler(sampler); // id = 4

	sampler_info.pNext = nullptr;
	sampler_info.magFilter = VK_FILTER_NEAREST;
	sampler_info.minFilter = VK_FILTER_NEAREST;
	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK; // reverse depth

	vkCreateSampler(device, &sampler_info, nullptr, &sampler);
	sampler_cache.add_sampler(sampler); // id = 5

	//> CSM
	float far = LIGHT_FAR_PLANE; 
	float near = main_camera.far; 
	size_t m = cascade_data.size();
	float range = far - near;
	float ratio = far / near;
	float lambda = 0.5;

	for (size_t idx = 0; idx < m; idx++)
	{
		float i = idx + 1.0f;
		float log = near * std::powf(ratio, i / m);
		float uniform = near + range * i / m;
		float split = lambda * log + (1.0f - lambda) * uniform; // in world units
		cascade_data[idx].split_ratio = (split - near) / range;
		scene_data.cascade_splits[static_cast<int>(idx)] = (cascade_data[idx].split_ratio * range + near) * -1.0f;

		cascade_data[idx].shadow_map = create_image(VkExtent3D{SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1}, VK_FORMAT_D32_SFLOAT,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT
		);

		bindless_texture.shadow = texture_cache.add_texture(cascade_data[idx].shadow_map.view);
	}
	bindless_texture.shadow -= (static_cast<uint8_t>(cascade_data.size()) - 1); // dirty adj

	main_deletion_queue.push_function([&]() {
		destroy_image(error_image);

		for (size_t i = 0; i < cascade_data.size(); i++)
		{
			destroy_image(cascade_data[i].shadow_map);
		}
	});

	//> init scene
	render_scene.init();

	//> create depth pyramid
	VkExtent3D depth_pyramid_extent{};
	depth_pyramid_extent.width = nearest_pow2(draw_extent.width);
	depth_pyramid_extent.height = nearest_pow2(draw_extent.height);
	depth_pyramid_extent.depth = 1;

	depth_pyramid = create_image(depth_pyramid_extent, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, true);
	
	// sampling in occlusion culling
	auto id = texture_cache.add_texture(depth_pyramid.view);
	texture_cache.set_depth_pyramid_image(id);

	uint32_t mip_levels = static_cast<uint32_t>(std::floor(std::log2(std::max(depth_pyramid_extent.width, depth_pyramid_extent.height)))) + 1;

	std::vector<VkImageView> pyramid_views(mip_levels);
	VkImageViewCreateInfo img_view_info = vkinit::imageview_create_info(VK_FORMAT_R32_SFLOAT, depth_pyramid.image, VK_IMAGE_ASPECT_COLOR_BIT);
	img_view_info.subresourceRange.levelCount = 1;
	img_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
	img_view_info.subresourceRange.baseMipLevel = 0;
	vkCreateImageView(device, &img_view_info, nullptr, &pyramid_views[0]);
	id = image_cache.add_texture(pyramid_views[0]);
	image_cache.set_depth_pyramid_image(id);

	for (int mip = 1; mip < mip_levels; mip++)
	{
		img_view_info.subresourceRange.baseMipLevel = mip;
		vkCreateImageView(device, &img_view_info, nullptr, &pyramid_views[mip]);
		image_cache.add_texture(pyramid_views[mip]);
	}

	main_deletion_queue.push_function([&, pyramid_views]() {
		destroy_image(depth_pyramid);
		for (size_t i = 0; i < pyramid_views.size(); i++)
		{
			vkDestroyImageView(device, pyramid_views[i], nullptr);
		}
	});
}

void VulkanEngine::init_renderables()
{
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

	cubemap_image = create_cubemap(ibl_extent, VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT, 0, true
	);

	irradiance_image = create_cubemap({64, 64, 1}, VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT
	);

	prefiltered_image = create_cubemap({ 512, 512, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT, 0, true
	);

	brdflut_image = create_image({ 128, 128, 1 }, VK_FORMAT_R16G16_SFLOAT, 
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT
	);

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

	main_deletion_queue.push_function([&, temporary_views]() {
		for (int mip = 0; mip < temporary_views.size(); mip++)
		{
			vkDestroyImageView(device, temporary_views[mip], nullptr);
		}
	});

	main_deletion_queue.push_function([&]() {
		destroy_image(equirectangular_image);
		destroy_image(cubemap_image);
		destroy_image(irradiance_image);
		destroy_image(prefiltered_image);
		destroy_image(brdflut_image);
		});

	//std::string asset_path = "../../assets/ABeautifulGame.glb";
	//std::string asset_path = "../../assets/sphere.gltf";
	//std::string asset_path = "../../assets/khronos_sponza/Sponza.gltf";
	//std::string asset_path = "../../assets/bistro_interior_wine_ktx2/BistroInterior_WineFixed.gltf";
	//std::string asset_path = "../../assets/bistro_exterior_ktx2/BistroExteriorFixed.gltf";
	std::string asset_path = "../../assets/suzanne.gltf";
	//std::string asset_path = "../../assets/AlphaBlendModeTest.glb";
	auto start{ std::chrono::system_clock::now() };
	auto asset_file = load_gltf(this, asset_path);
	auto end{ std::chrono::system_clock::now() };
	auto elapsed{ std::chrono::duration_cast<std::chrono::microseconds>(end - start) };
	float ret = elapsed.count() / 1000.0f;
	fmt::println("load gltf: {}ms", ret);
	assert(asset_file.has_value());
	loaded_scenes["DamagedHelmet"] = *asset_file;
	render_scene.combined_mesh_buffer = loaded_scenes["DamagedHelmet"]->combined_mesh_buffer;

	std::array<glm::mat4, 3> t{};
	t[0] = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, 4));// * glm::scale(glm::mat4(1.0f), glm::vec3(2.0));
	t[1] = glm::translate(glm::mat4(1.0f), glm::vec3(-0.75, -0.2, 4));// * glm::scale(glm::mat4(1.0f), glm::vec3(2.0));
	t[2] = glm::translate(glm::mat4(1.0f), glm::vec3(0.75, 0.2, 4));// * glm::scale(glm::mat4(1.0f), glm::vec3(2.0));
	for (const auto& n : loaded_scenes["DamagedHelmet"]->top_nodes)
	{
		register_object(n.get(), t[0]);
#ifndef SINGLE
		register_object(n.get(), t[1]); 
		register_object(n.get(), t[2]);
#endif
	}

	std::mt19937 mt(42);
	auto draw_radius = 20.0f;
	auto draw_count = 5000;

	for (size_t i = 0; i < draw_count; i++)
	{
		const float x = static_cast<float>(mt()) / mt.max() * draw_radius - draw_radius * 0.5f;
		const float y = static_cast<float>(mt()) / mt.max() * draw_radius - draw_radius * 0.5f;
		const float z = static_cast<float>(mt()) / mt.max() * -draw_radius;

		glm::mat4 t = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
		glm::vec3 axis = glm::normalize(glm::vec3(static_cast<float>(mt()) / mt.max(), static_cast<float>(mt()) / mt.max(), static_cast<float>(mt()) / mt.max()));
		glm::mat4 r = glm::rotate(glm::mat4(1.0f), glm::radians(static_cast<float>(mt()) / mt.max() * 360.0f), axis);
		glm::mat4 s = glm::scale(glm::mat4(1.0f), glm::vec3(static_cast<float>(mt()) / mt.max()) + 1.0f);
		const auto transform = t * r * s;

		for (const auto& n : loaded_scenes["DamagedHelmet"]->top_nodes)
		{
#ifndef SINGLE
			register_object(n.get(), transform);
#endif
		}
	}
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
	write.descriptorCount = static_cast<uint32_t>(texture_cache.image_infos.size()); // (!) validation layer does not report if smaller count than req used; fragment sample simply returns black
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

void VulkanEngine::register_object(Node* node, const glm::mat4& top_matrix)
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

			if (found)
			{
				obj.primitive_id.handle = static_cast<uint32_t>(handle + i);
			}
			else
			{
				obj.primitive_id.handle = static_cast<uint32_t>(render_scene.primitives.size());
				render_scene.primitives.emplace_back(DrawPrimitive{ s.bounds.origin, s.bounds.radius, s.mesh_lods, s.lod_count });
			}

			obj.material_buffer_address = node->mesh->material_buffer_address;
			obj.material = &material_cache.data[s.material];
			obj.material_id = s.material_id;
			obj.transform = node_matrix;

			uint32_t handle = static_cast<uint32_t>(render_scene.renderables.size());
			render_scene.renderables.push_back(obj);

			if (s.pass == MaterialPass::Blend)
			{
				//ctx.transparent_objects.push_back(obj);
			}
			else // OPAQUE and MASK
			{
				render_scene.forward_pass.unbatched_objects.push_back(handle);
#ifdef SHADOW
				for (size_t i = 0; i < NUMBER_OF_CASCADES; i++)
				{
					render_scene.shadow_pass[i].unbatched_objects.push_back(handle);
				}
#endif
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

	main_camera.update(stats.deltatime);

	scene_data.view = main_camera.get_view_matrix();
	scene_data.proj = main_camera.perspective;
	scene_data.viewproj = scene_data.proj * scene_data.view;
	//scene_data.sunlight_dir = glm::vec4(7.75, 12.5, 12.5, 1.);
	scene_data.sunlight_dir = glm::vec4(0.001, 12.0, 0.0, 1.);
	//scene_data.sunlight_dir = glm::vec4(0.0, 12.0, 12.0, 1.);
	scene_data.sunlight_color = glm::vec4(1);

	update_cascade();
	for (size_t i = 0; i < cascade_data.size(); i++)
	{
		scene_data.shadow_transforms[i] = cascade_data[i].viewproj;
	}
	scene_data.camera_pos = glm::vec4(main_camera.position, 1.0);

	// (!) uncomment if registering objects here
	//render_scene.renderables.clear();
	//render_scene.unbatched_objects.clear();

	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	stats.scene_update_time = elapsed.count() / 1000.0f; // milliseconds
}

uint32_t TextureCache::add_texture(const VkImageView& view)
{
	for (size_t i = 0; i < image_infos.size(); i++)
	{
		if (image_infos[i].imageView == view)
			return static_cast<uint32_t>(i);
	}

	uint32_t id = static_cast<uint32_t>(image_infos.size());

	image_infos.emplace_back(VkDescriptorImageInfo{ 0, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL });

	return id;
}

void SamplerCache::add_sampler(const VkSampler& sampler)
{
	image_infos.emplace_back(VkDescriptorImageInfo{ .sampler = sampler });
}

uint32_t ImageCache::add_texture(const VkImageView& view)
{
	for (size_t i = 0; i < image_infos.size(); i++)
	{
		if (image_infos[i].imageView == view)
			return static_cast<uint32_t>(i);
	}

	uint32_t id = static_cast<uint32_t>(image_infos.size());

	image_infos.emplace_back(VkDescriptorImageInfo{ 0, view, VK_IMAGE_LAYOUT_GENERAL });

	return id;
}

VkShaderModule ShaderCache::add_shader(VkDevice device, const char* path)
{
	std::string shader_path{ "../../shaders/" };
	shader_path += path;
	auto it = data.find(shader_path);

	if (it == data.end())
	{
		VkShaderModule module{};
		vkutil::load_shader_module(shader_path.c_str(), device, &module);
		data[shader_path] = module;
	}

	return data[shader_path];
}

uint32_t MaterialCache::add_material(ShaderPass* forward, ShaderPass* shadow)
{
	// (!) highly inefficient due to linear search, refactor
	for (size_t i = 0; i < data.size(); i++)
	{
		if (data[i].forward_pass == forward && data[i].shadow_pass == shadow)
			return static_cast<uint32_t>(i);
	}

	data.emplace_back(Material{ forward, shadow });

	return static_cast<uint32_t>(data.size()) - 1;
}

void VulkanEngine::forward_pass(VkCommandBuffer cmd)
{
	//vkutil::transition_image(
	//	cmd,
	//	draw_image.image,
	//	VK_IMAGE_LAYOUT_UNDEFINED,
	//	VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	//	VK_PIPELINE_STAGE_2_BLIT_BIT,
	//	VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
	//	VK_ACCESS_2_TRANSFER_READ_BIT,
	//	VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
	//); // commented as already performed prior to 2 pass occlusion culling

	//vkutil::transition_image(
	//	cmd,
	//	depth_image.image,
	//	VK_IMAGE_LAYOUT_UNDEFINED,
	//	VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
	//	VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, // depth reduction read
	//	VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
	//	VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
	//	VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
	//	VK_IMAGE_ASPECT_DEPTH_BIT
	//);

	VkClearColorValue clear_color_value{ 0.0f, 0.0f, 0.0f, 1.0f };
	VkClearValue clear_value{ .color = clear_color_value };
	VkRenderingAttachmentInfo color_attachment = vkinit::attachment_info(draw_image.view, &clear_value);
	VkRenderingAttachmentInfo depth_attachment = vkinit::depth_attachment_info(depth_image.view);
	VkRenderingInfo render_info = vkinit::rendering_info(draw_extent, &color_attachment, &depth_attachment);

	//vkCmdBeginRendering(cmd, &render_info);

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

	ShaderPass current_pass = *shader_passes["textured_lit"];

	//vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
	//vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_tex_descriptor, 0, nullptr);
	//vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_sampler_descriptor, 0, nullptr);

	////> gpu driven
	//GPUPushConstants gpc{}; // (!) clean up
	//gpc.material_buffer_address = render_scene.renderables[0].material_buffer_address; // (!) refactor this

	//VkBufferDeviceAddressInfo address_info{};
	//address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	//address_info.buffer = render_scene.object_buffer.buffer;
	//gpc.object_buffer_address = vkGetBufferDeviceAddress(device, &address_info);

	//gpc.vertex_buffer_address = render_scene.combined_mesh_buffer.vertex_buffer_address;

	//vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GPUPushConstants), &gpc);

	//vkCmdBindIndexBuffer(cmd, render_scene.combined_mesh_buffer.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

	//for (size_t i = 0; i < render_scene.forward_pass.multibatches.size(); i++)
	//{
	//	const auto& multibatch = render_scene.forward_pass.multibatches[i];
	//	const auto& pipeline = multibatch.pipeline;

	//	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pipeline);
	//	vkCmdDrawIndexedIndirectCount(cmd, render_scene.forward_pass.draw_indirect_buffer.buffer, multibatch.offset * sizeof(VkDrawIndexedIndirectCommand),
	//		render_scene.forward_pass.count_buffer.buffer, i * sizeof(uint32_t),
	//		multibatch.max_draw_count, sizeof(VkDrawIndexedIndirectCommand)
	//	);
	//	stats.draw_count++;
	//}

	//vkCmdEndRendering(cmd);

	color_attachment = vkinit::attachment_info(draw_image.view, nullptr);
	render_info = vkinit::rendering_info(draw_extent, &color_attachment, nullptr);

	vkCmdBeginRendering(cmd, &render_info);

	current_pass = *shader_passes["debug"];
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_sampler_descriptor, 0, nullptr);
	DebugPushConstants pc{};
	pc.texture_id = texture_cache.get_depth_pyramid_image();
	//pc.texture_id = texture_cache.get_depth_image();
	pc.lod = CVAR_DEPTH_PYRAMID_LOD.get();
	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(DebugPushConstants), &pc);
	if (CVAR_RENDER_PYRAMID.get())
		vkCmdDraw(cmd, 3, 1, 0, 0);
	stats.draw_count++;

	//> skybox
	//current_pass = *shader_passes["skybox"];
	//vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
	//vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
	//vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_tex_descriptor, 0, nullptr);
	//vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_sampler_descriptor, 0, nullptr);
	//SkyboxPushConstants pc{};
	//auto proj = main_camera.perspective;
	//auto view_no_translation = glm::mat3(main_camera.get_view_matrix());
	//auto view = glm::mat4(view_no_translation);
	//pc.inverse_viewproj = glm::inverse(view) * glm::inverse(proj); // go in reverse order
	//pc.texture_id = bindless_texture.skybox;
	//vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SkyboxPushConstants), &pc);
	//vkCmdDraw(cmd, 3, 1, 0, 0);
	//stats.draw_count++;

	vkCmdEndRendering(cmd);

	//vkutil::transition_image(
	//	cmd,
	//	draw_image.image,
	//	VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	//	VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	//	VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
	//	VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
	//	VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
	//	VK_ACCESS_2_SHADER_READ_BIT
	//);

	////> post fx
	//color_attachment = vkinit::attachment_info(draw_image2.view, &clear_value);
	////depth_attachment = vkinit::depth_attachment_info(depth_image.view);
	//render_info = vkinit::rendering_info(draw_extent, &color_attachment, nullptr);

	//vkutil::transition_image(
	//	cmd,
	//	draw_image2.image,
	//	VK_IMAGE_LAYOUT_UNDEFINED,
	//	VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	//	VK_PIPELINE_STAGE_2_BLIT_BIT,
	//	VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
	//	VK_ACCESS_2_TRANSFER_READ_BIT,
	//	VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
	//);

	//vkCmdBeginRendering(cmd, &render_info);

	//viewport.x = 0;
	//viewport.y = static_cast<float>(draw_extent.height);
	//viewport.width = static_cast<float>(draw_extent.width);
	//viewport.height = -static_cast<float>(draw_extent.height);
	//viewport.minDepth = 0.0f;
	//viewport.maxDepth = 1.0f;
	//vkCmdSetViewport(cmd, 0, 1, &viewport);

	//scissor.offset.x = 0;
	//scissor.offset.y = 0;
	//scissor.extent.width = draw_extent.width;
	//scissor.extent.height = draw_extent.height;
	//vkCmdSetScissor(cmd, 0, 1, &scissor);

	//current_pass = *shader_passes["tonemap"];

	//vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
	//vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
	//vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_tex_descriptor, 0, nullptr);
	//vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_sampler_descriptor, 0, nullptr);
	//PostFXPushConstants fx_pc{};
	//fx_pc.texture_id = texture_cache.get_draw_image();
	//vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PostFXPushConstants), &fx_pc);
	//vkCmdDraw(cmd, 3, 1, 0, 0);
	//stats.draw_count++;

	//vkCmdEndRendering(cmd);

	vkutil::transition_image(
		cmd,
		//draw_image2.image,
		draw_image.image,
		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		VK_PIPELINE_STAGE_2_BLIT_BIT,
		VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
		VK_ACCESS_2_TRANSFER_READ_BIT
	);
}

void VulkanEngine::shadow_pass(VkCommandBuffer cmd, RenderScene::MeshPass& pass, size_t cascade_idx)
{
	CascadeData& cascade = cascade_data[cascade_idx];

	VkRenderingAttachmentInfo depth_attachment = vkinit::depth_attachment_info(cascade.shadow_map.view);
	VkExtent2D shadow_extent = VkExtent2D{ cascade.shadow_map.extent.width, cascade.shadow_map.extent.height };
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
	vkCmdSetDepthBias(cmd, -depth_bias, 0.0f, -slope_scaled_depth_bias); // (!) modify for double sided geometry? no back face culling to assist with artifacts

	ShaderPass current_pass = *shader_passes["shadow"];
	// (!) ugly - refactor current_pass.layout and current_pass? 
	// (!) these are not reset between dynamic rendering instances - verify
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_sampler_descriptor, 0, nullptr);

	//> gpu driven
	ShadowPushConstants pc{}; // (!) clean up
	pc.viewproj = cascade.viewproj;

	VkBufferDeviceAddressInfo address_info{};
	address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	pc.material_buffer_address = render_scene.renderables[0].material_buffer_address; // (!) refactor this

	address_info.buffer = render_scene.object_buffer.buffer;
	pc.object_buffer_address = vkGetBufferDeviceAddress(device, &address_info);

	pc.vertex_buffer_address = render_scene.combined_mesh_buffer.vertex_buffer_address;
	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ShadowPushConstants), &pc);

	vkCmdBindIndexBuffer(cmd, render_scene.combined_mesh_buffer.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
	for (size_t i = 0; i < pass.multibatches.size(); i++)
	{
		const auto& multibatch = pass.multibatches[i];
		const auto& pipeline = multibatch.pipeline;

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pipeline);

		vkCmdDrawIndexedIndirectCount(cmd, pass.draw_indirect_buffer.buffer, multibatch.offset * sizeof(VkDrawIndexedIndirectCommand),
			pass.count_buffer.buffer, i * sizeof(uint32_t),
			multibatch.max_draw_count, sizeof(VkDrawIndexedIndirectCommand)
		);

		stats.draw_count++;
	}
	vkCmdEndRendering(cmd);
};

void VulkanEngine::update_cascade()
{
	auto light_dir = glm::normalize(glm::vec3(scene_data.sunlight_dir));

	glm::mat4 view = main_camera.get_view_matrix();
	// (!) refactor draw_extent?
	glm::mat4 proj = glm::perspective(glm::radians(main_camera.fov), static_cast<float>(draw_extent.width) / draw_extent.height, LIGHT_FAR_PLANE, main_camera.far);
	glm::mat4 inv_viewproj = glm::inverse(proj * view);
	
	glm::mat4 light_view = glm::lookAt(glm::vec3(0.0), -light_dir, glm::vec3(0.0, 1.0, 0.0));

	std::array<glm::vec3, 8> frustum_corners{
		glm::vec3(-1,  1,  1),
		glm::vec3( 1,  1,  1),
		glm::vec3(-1, -1,  1),
		glm::vec3( 1, -1,  1),
		glm::vec3(-1,  1,  0),
		glm::vec3( 1,  1,  0),
		glm::vec3(-1, -1,  0),
		glm::vec3( 1, -1,  0), // reverse depth order, flipping z values will be incorrect
	};

	for (size_t i = 0; i < frustum_corners.size(); i++)
	{
		glm::vec4 corner = inv_viewproj * glm::vec4(frustum_corners[i], 1.0);
		frustum_corners[i] = glm::vec3(corner / corner.w);
	}
	
	float last_split = 0.0;
	for (size_t i = 0; i < cascade_data.size(); i++)
	{
		float current_split = cascade_data[i].split_ratio;

		std::array<glm::vec3, 8> transformed_corners = frustum_corners;

		for (size_t i = 0; i < 4; i++)
		{
			glm::vec3 distance = transformed_corners[i + 4] - transformed_corners[i];
			transformed_corners[i] = transformed_corners[i] + distance * last_split;
			transformed_corners[i + 4] = transformed_corners[i] + distance * current_split;
		}
		last_split = current_split;

		// (!) TODO: try ritter's
		glm::vec3 center{};
		for (size_t i = 0; i < transformed_corners.size(); i++)
		{
			center += transformed_corners[i];
		}
		center /= 8.0f;

		float radius{};
		for (size_t i = 0; i < transformed_corners.size(); i++)
		{
			float dist = glm::length(transformed_corners[i] - center);
			radius = std::max(dist, radius);
		}

		// stable csm via snapping projection matrix - https://github.com/TheRealMJP/Shadows/blob/master/Shadows/SetupShadows.hlsl
		// stable csm via snapping frustum center, less robust - https://alextardif.com/shadowmapping.html
		glm::mat4 shadow_view = glm::lookAt(center + radius * light_dir, center, glm::vec3(0, 1, 0));
		glm::mat4 shadow_proj = glm::ortho(-radius, radius, -radius, radius, radius * 2.0f, 0.0f + CVAR_SHADOW_NEAR.get());
		glm::vec2 shadow_origin = glm::vec2(0.0);

		shadow_origin = shadow_proj * shadow_view * glm::vec4(shadow_origin, 0.0, 1.0);
		shadow_origin *= (SHADOW_MAP_SIZE / 2.0f);

		glm::vec2 rounded_origin = glm::round(shadow_origin);
		glm::vec2 offset = rounded_origin - shadow_origin;
		offset *= (2.0f / SHADOW_MAP_SIZE);

		shadow_proj[3][0] += offset.x;
		shadow_proj[3][1] += offset.y;

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
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls

	// Setup Platform/Renderer backends
	ImGui_ImplSDL2_InitForVulkan(window);
	ImGui_ImplVulkan_InitInfo init_info{};
	init_info.ApiVersion = VK_API_VERSION_1_3; // (!) hardcoded
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

	ImGui_ImplVulkan_Init(&init_info);

	main_deletion_queue.push_function([&, imgui_pool]() {
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplSDL2_Shutdown();
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

// Update Object Buffer
// *Instancing* ForEach RenderPass: Pass Object -> sort -> Indirect Batch -> Multi Batch -> Count Buffer -> Instance Buffer -> gInstance Buffer -> Indirect Buffer
// *No instancing* ForEach RenderPass: Pass Object -> sort -> Indirect Batch -> Multi Batch -> Visibility Buffer -> Count Buffer -> Indirect Buffer
void VulkanEngine::ready_mesh_draw()
{
	ZoneScopedN("Ready mesh draw");

	if (render_scene.object_buffer.info.size < render_scene.renderables.size() * sizeof(ObjectData))
	{
		fmt::println("object_buffer");
		render_scene.object_buffer = reallocate_buffer(
			render_scene.renderables.size() * sizeof(ObjectData),
			render_scene.object_buffer,
			VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
			VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
		);

		render_scene.build_object_buffer();
	}

	if (render_scene.mesh_buffer.info.size < render_scene.primitives.size() * sizeof(DrawPrimitive))
	{
		fmt::println("mesh_buffer");
		render_scene.mesh_buffer = reallocate_buffer(
			render_scene.primitives.size() * sizeof(DrawPrimitive),
			render_scene.mesh_buffer,
			VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
			VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT // ssbo usage?
		);

		render_scene.build_mesh_buffer();
	}

	std::vector<RenderScene::MeshPass*> passes = { &render_scene.forward_pass, &render_scene.transparent_pass };
	for (size_t i = 0; i < NUMBER_OF_CASCADES; i++)
	{
		passes.push_back(&render_scene.shadow_pass[i]);
	}
	passes.push_back(&render_scene.forward_pass);
	passes.push_back(&render_scene.transparent_pass);

	for (size_t i = 0; i < passes.size(); i++)
	{
		auto& pass = *passes[i];

		if (pass.pass_objects.size() != pass.unbatched_objects.size())
		{
			fmt::println("pass_object, indirect_batch");
			render_scene.build_pass_objects(pass);
			render_scene.sort_objects(pass);
			render_scene.build_indirect_batch(pass);
			render_scene.build_multi_batch(pass);
		}

		if (pass.instance_buffer.info.size < pass.pass_objects.size() * sizeof(GPUInstance))
		{
			fmt::println("instance buffer");

			pass.instance_buffer = reallocate_buffer(
				pass.pass_objects.size() * sizeof(GPUInstance),
				pass.instance_buffer,
				VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
				VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
			);

			render_scene.build_instance_buffer(pass);
		}

		if (pass.vis_buffer.info.size < pass.unbatched_objects.size())
		{
			fmt::println("visibility buffer");
			pass.vis_buffer = reallocate_buffer(
				pass.unbatched_objects.size() * sizeof(uint32_t),
				pass.vis_buffer,
				0,
				VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
			);

			pass.debug_buffer = reallocate_buffer(
				pass.unbatched_objects.size() * sizeof(float),
				pass.debug_buffer,
				0,
				VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
			);

			// test
			immediate_submit([&](VkCommandBuffer cmd) {
				vkCmdFillBuffer(cmd, pass.vis_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
				});
		}

		if (pass.count_buffer.info.size < pass.multibatches.size() * sizeof(uint32_t))
		{
			fmt::println("count_buffer");
			pass.count_buffer = reallocate_buffer(
				pass.multibatches.size() * sizeof(uint32_t),
				pass.count_buffer,
				0,
				VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
			);
		}

		if (pass.draw_indirect_buffer.info.size < pass.pass_objects.size() * sizeof(GPUIndirect))
		{
			fmt::println("draw_indirect_buffer");
			pass.draw_indirect_buffer = reallocate_buffer(
				pass.pass_objects.size() * sizeof(GPUIndirect),
				pass.draw_indirect_buffer,
				0,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
			);
		}
	}
}

CullData VulkanEngine::ready_cull_data(RenderScene::MeshPass& pass, glm::mat4& proj, bool orthographic /*= false*/)
{
	CullData cull_data{};

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
		
	auto normalize_plane = [&](glm::vec4& plane) {
		float length = glm::length(glm::vec3(plane));
		plane /= length;
	};

	normalize_plane(left_plane);
	normalize_plane(bottom_plane);

	// (!) TODO: fix
	//if (orthographic)
	//{
	//	cull_data.frustum_planes[0] = m3 - m2; // near
	//	cull_data.frustum_planes[1] = m3 + m2; // far
	//}

	cull_data.view = scene_data.view;
	cull_data.frustum_planes = glm::vec4(left_plane.x, left_plane.z, bottom_plane.y, bottom_plane.z);

	VkBufferDeviceAddressInfo address_info{};
	address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	address_info.buffer = render_scene.object_buffer.buffer;
	cull_data.object_buffer_address = vkGetBufferDeviceAddress(device, &address_info);

	address_info.buffer = render_scene.mesh_buffer.buffer;
	cull_data.mesh_buffer_address = vkGetBufferDeviceAddress(device, &address_info);

	address_info.buffer = pass.instance_buffer.buffer;
	cull_data.instance_buffer_address = vkGetBufferDeviceAddress(device, &address_info);

	address_info.buffer = pass.draw_indirect_buffer.buffer;
	cull_data.draw_indirect_address = vkGetBufferDeviceAddress(device, &address_info);

	address_info.buffer = pass.count_buffer.buffer;
	cull_data.count_buffer_address = vkGetBufferDeviceAddress(device, &address_info);

	address_info.buffer = pass.vis_buffer.buffer;
	cull_data.vis_buffer_address = vkGetBufferDeviceAddress(device, &address_info);

	address_info.buffer = pass.debug_buffer.buffer;
	cull_data.debug_buffer_address = vkGetBufferDeviceAddress(device, &address_info);

	cull_data.count = static_cast<uint32_t>(pass.pass_objects.size());
	cull_data.texture_id = texture_cache.get_depth_pyramid_image();
	cull_data.occlusion = CVAR_OCCLUSION.get();
	
	cull_data.p00 = proj[0][0];
	cull_data.p11 = proj[1][1];
	cull_data.near = main_camera.far;
	cull_data.far = main_camera.near;

	cull_data.resolution = glm::vec2(depth_pyramid.extent.width, depth_pyramid.extent.height);
	cull_data.texture_lod = std::floor(std::log2(std::max(depth_pyramid.extent.width, depth_pyramid.extent.height))) + 1;
	cull_data.lod_distance_factor = 2.0 / (cull_data.p11 * static_cast<float>(draw_extent.height));
	cull_data.debug_lod = CVAR_DEBUG_LOD.get();

	return cull_data;
}

void VulkanEngine::execute_compute_cull(VkCommandBuffer cmd, RenderScene::MeshPass& pass, CullData& cull_data, bool late)
{
	ShaderPass current_pass = *shader_passes["cull"];
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.pipeline);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 0, 1, &bindless_image_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 1, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, current_pass.layout, 2, 1, &bindless_sampler_descriptor, 0, nullptr);

	cull_data.late = late ? 1 : 0;

	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullData), &cull_data);
	vkCmdDispatch(cmd, static_cast<uint32_t>(std::ceil(pass.pass_objects.size() / 256.0)), 1, 1);
}

void VulkanEngine::render(VkCommandBuffer cmd, bool late, uint32_t query)
{
	vkCmdBeginQuery(cmd, query_pool_pipelines, query, 0);

	VkClearColorValue clear_color_value{ 0.0f, 0.0f, 0.0f, 1.0f };
	VkClearValue clear_value{ .color = clear_color_value };
	VkRenderingAttachmentInfo color_attachment = late ? vkinit::attachment_info(draw_image.view, nullptr) : vkinit::attachment_info(draw_image.view, &clear_value);
	VkRenderingAttachmentInfo depth_attachment = vkinit::depth_attachment_info(depth_image.view);
	depth_attachment.loadOp = late ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
	VkRenderingInfo render_info = vkinit::rendering_info(draw_extent, &color_attachment, &depth_attachment);

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

	auto depth_bias = 0.f;
	auto slope_scaled_depth_bias = 0.f;
	vkCmdSetDepthBias(cmd, -depth_bias, 0.0f, -slope_scaled_depth_bias); // (!) modify for double sided geometry? no back face culling to assist with artifacts

	ShaderPass current_pass = *shader_passes["textured_lit"];

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_sampler_descriptor, 0, nullptr);

	GPUPushConstants pc{};

	VkBufferDeviceAddressInfo address_info{};
	address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	address_info.buffer = render_scene.object_buffer.buffer;

	pc.material_buffer_address = 0;
	pc.object_buffer_address = vkGetBufferDeviceAddress(device, &address_info);
	pc.vertex_buffer_address = render_scene.combined_mesh_buffer.vertex_buffer_address;
	pc.sphere = CVAR_SPHERE.get();

	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GPUPushConstants), &pc);

	vkCmdBindIndexBuffer(cmd, render_scene.combined_mesh_buffer.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

	for (size_t i = 0; i < render_scene.forward_pass.multibatches.size(); i++)
	{
		const auto& multibatch = render_scene.forward_pass.multibatches[i];
		const auto& pipeline = multibatch.pipeline;

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline); 
		vkCmdDrawIndexedIndirectCount(cmd, render_scene.forward_pass.draw_indirect_buffer.buffer, multibatch.offset * sizeof(VkDrawIndexedIndirectCommand),
			render_scene.forward_pass.count_buffer.buffer, i * sizeof(uint32_t),
			multibatch.max_draw_count, sizeof(VkDrawIndexedIndirectCommand)
		);
		stats.draw_count++;
	}

	vkCmdEndRendering(cmd);
	vkCmdEndQuery(cmd, query_pool_pipelines, query);
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

	uint32_t mip_levels = static_cast<uint32_t>(std::floor(std::log2(std::max(depth_pyramid.extent.width, depth_pyramid.extent.height)))) + 1;

	for (size_t i = 0; i < mip_levels; i++)
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