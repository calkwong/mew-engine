#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_initializers.h>
#include <vk_types.h>
#include <vk_images.h>
#include <vk_descriptors.h>
#include <vk_pipelines.h>
#include <vk_loader.h>

#define VMA_IMPLEMENTATION

//#ifndef VMA_DEBUG_LOG_FORMAT 
//
//#define VMA_DEBUG_LOG_FORMAT(format, ...) do { \
//	printf((format), __VA_ARGS__); \
//	printf("\n"); \
//} while(false)
//
//#endif 
//
//#ifndef VMA_DEBUG_LOG 
//#define VMA_DEBUG_LOG(str)   VMA_DEBUG_LOG_FORMAT("%s", (str)) 
//#endif 


#include "vk_mem_alloc.h"

#include <glm/gtx/transform.hpp>
#include "stb_image.h"

// for glm debug
#include "glm/ext.hpp"
#include "glm/gtx/string_cast.hpp"

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

static void check_vk_result(VkResult err)
{
	if (err == 0)
		return;
	fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
	if (err < 0)
		abort();
}

#include <thread>

VulkanEngine* loaded_engine{};

VulkanEngine& VulkanEngine::get() { return *loaded_engine; }

constexpr bool USE_VALIDATION_LAYERS = true;

uint32_t equi_id{};
uint32_t cube_id{};
uint32_t irradiance_id{};
uint32_t prefiltered_id{};
uint32_t brdflut_id{};

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
	main_camera.perspective = glm::perspective(glm::radians(70.0f), static_cast<float>(draw_extent.width) / draw_extent.height, 45.0f, 0.01f);

	init_precomputations();

	is_initialized = true;
}

void VulkanEngine::cleanup()
{
	if (is_initialized) {

		vkDeviceWaitIdle(device);
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

			//frames[i].frame_descriptor.destroy_pools(device); // (!) validation layer should report - must be destroyed after pipeline/pipeline layout
			frames[i].deletion_queue.flush(); // (!) will this clash with final frame still in flight
		}
		
		for (const auto& [k, v] : shader_passes)
		{
			vkDestroyPipeline(device, v->pipeline, nullptr);
			vkDestroyPipelineLayout(device, v->layout, nullptr);
		}

		main_deletion_queue.flush();

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
	update_scene();
	VK_CHECK(vkWaitForFences(device, 1, &get_current_frame().render_fence, true, 1000000000));
	VK_CHECK(vkResetFences(device, 1, &get_current_frame().render_fence));

	get_current_frame().deletion_queue.flush();

	SceneData* scene_uniform_data = static_cast<SceneData*>(get_current_frame().scene_buffer.info.pMappedData);
	fmt::println("shadow transform: {}", glm::to_string(scene_data.shadow_transform));
	*scene_uniform_data = scene_data;

	uint32_t swapchain_image_idx{};
	VK_CHECK(vkAcquireNextImageKHR(device, swapchain, 1000000000, get_current_frame().swapchain_semaphore, nullptr, &swapchain_image_idx));

	VkCommandBuffer cmd = get_current_frame().main_command_buffer;

	VK_CHECK(vkResetCommandBuffer(cmd, 0));

	VkCommandBufferBeginInfo cmd_begin_info = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT); // (!)

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));

	shadow_pass(cmd);
	forward_pass(cmd);

	vkutil::transition_image(
		cmd,
		swapchain_images[swapchain_image_idx],
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		0,
		VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		0,
		VK_ACCESS_2_TRANSFER_WRITE_BIT
	);

	vkutil::copy_image(cmd, draw_image2.image, swapchain_images[swapchain_image_idx], draw_extent, swapchain_extent);

	vkutil::transition_image(
		cmd,
		swapchain_images[swapchain_image_idx],
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		VK_ACCESS_2_TRANSFER_WRITE_BIT,
		VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
	);
	
	draw_imgui(cmd, swapchain_image_views[swapchain_image_idx]);

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

	VK_CHECK(vkEndCommandBuffer(cmd));

	VkCommandBufferSubmitInfo cmd_info = vkinit::command_buffer_submit_info(cmd);

	VkSemaphoreSubmitInfo wait_info = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, get_current_frame().swapchain_semaphore);
	VkSemaphoreSubmitInfo submit_info = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, get_current_frame().render_semaphore); // all graphics bit?
	
	VkSubmitInfo2 submit = vkinit::submit_info(&cmd_info, &submit_info, &wait_info);

	VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit, get_current_frame().render_fence));

	VkPresentInfoKHR present_info{};
	present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	present_info.waitSemaphoreCount = 1;
	present_info.pWaitSemaphores = &get_current_frame().render_semaphore;
	present_info.swapchainCount = 1;
	present_info.pSwapchains = &swapchain;
	present_info.pImageIndices = &swapchain_image_idx;

	VK_CHECK(vkQueuePresentKHR(graphics_queue, &present_info));

	frame_number++;
}

void VulkanEngine::init_precomputations()
{
	//> draw
	VkCommandBuffer cmd = imm_command_buffer;
	VK_CHECK(vkResetFences(device, 1, &imm_fence));

	VK_CHECK(vkResetCommandBuffer(cmd, 0));

	VkCommandBufferBeginInfo cmd_begin_info = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT); // (!)

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
	vkCmdDispatch(cmd, std::ceil(cubemap_image.extent.width / 16.0), std::ceil(cubemap_image.extent.height / 16.0), 1);

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
	vkCmdDispatch(cmd, std::ceil(irradiance_image.extent.width / 8.0), std::ceil(irradiance_image.extent.height / 8.0), 1);

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
		vkCmdDispatch(cmd, std::ceil((prefiltered_image.extent.width >> mip) / 8.0), std::ceil((prefiltered_image.extent.height >> mip) / 8.0), 1);
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
	vkCmdDispatch(cmd, std::ceil(brdflut_image.extent.width / 8.0), std::ceil(brdflut_image.extent.height / 8.0), 1);

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

    while (!bQuit) {
		auto start = std::chrono::system_clock::now();
		auto deltatime = std::chrono::duration_cast<std::chrono::microseconds>(start - last_frame);
		stats.deltatime = deltatime.count() / 1000000.0f; // microseconds to seconds
		last_frame = start;

        // Handle events on queue
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT)
                bQuit = true;

            if (e.type == SDL_WINDOWEVENT) {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
                    stop_rendering = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
                    stop_rendering = false;
                }
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

        if (stop_rendering) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplSDL2_NewFrame();
		ImGui::NewFrame();
		
		ImGui::ShowDemoWindow();

		ImGui::Render();

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
	// bindless textures
	features12.descriptorBindingPartiallyBound = true;
	features12.descriptorBindingVariableDescriptorCount = true;
	features12.runtimeDescriptorArray = true;
	features12.shaderSampledImageArrayNonUniformIndexing = true;

	// use vkbootstrap to select a gpu. 
	// we want a gpu that can write to the SDL surface and supports vulkan 1.3 with the correct features
	vkb::PhysicalDeviceSelector selector{ vkb_inst };
	vkb::PhysicalDevice physicalDevice = selector
		.set_minimum_version(1, 3)
		.set_required_features_13(features13)
		.set_required_features_12(features12)
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
	texture_cache.set_draw_image(id);

	id = texture_cache.add_texture(draw_image2.view);
	texture_cache.set_draw_image2(id);

	// depth image
	depth_image.format = VK_FORMAT_D32_SFLOAT;
	depth_image.extent = draw_image_extent;

	depth_image = create_image(draw_image_extent, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);

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

	//swapchain_image_format = VK_FORMAT_B8G8R8A8_UNORM;
	swapchain_image_format = VK_FORMAT_B8G8R8A8_SRGB;

	vkb::Swapchain vkbSwapchain = swapchainBuilder
		//.use_default_format_selection()
		.set_desired_format(VkSurfaceFormatKHR{ .format = swapchain_image_format, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR }) // (!)
		.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR) 
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
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1} 
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
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 20 },
		{ VK_DESCRIPTOR_TYPE_SAMPLER, 10 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1} // imgui
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
		builder.bindings[0].descriptorCount = 20;

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
	ComputePipelineBuilder compute_builder{};

	//>
	ShaderEffect equi_to_cube{
		.layouts = { bindless_image_layout, bindless_tex_layout, bindless_sampler_layout },
		.pc = {
			{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(IBLPushConstants) }
		}
	};
	equi_to_cube.build_effect(device, "../../shaders/equi_to_cube.comp.spv");
	std::unique_ptr<ShaderPass> equi_to_cube_pass = vkutil::build_shader(device, &equi_to_cube, compute_builder);

	ShaderEffect irradiance = equi_to_cube;
	irradiance.build_effect(device, "../../shaders/irradiance.comp.spv");
	std::unique_ptr<ShaderPass> irradiance_pass = vkutil::build_shader(device, &irradiance, compute_builder);

	ShaderEffect prefiltered{
		.layouts = { bindless_image_layout, bindless_tex_layout, bindless_sampler_layout },
		.pc = {
			{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(IBLPushConstants) }
		}
	};
	prefiltered.build_effect(device, "../../shaders/prefiltered.comp.spv");
	std::unique_ptr<ShaderPass> prefiltered_pass = vkutil::build_shader(device, &prefiltered, compute_builder);

	ShaderEffect brdflut{
		.layouts = { bindless_image_layout },
		.pc = {
			{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(IBLPushConstants) }
		}
	};
	brdflut.build_effect(device, "../../shaders/brdf.comp.spv");
	std::unique_ptr<ShaderPass> brdf_pass = vkutil::build_shader(device, &brdflut, compute_builder);

	//> 
	PipelineBuilder builder{};
	builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
	builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
	builder.set_cull_mode(VK_CULL_MODE_FRONT_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	builder.set_multisampling_none();
	builder.disable_blending();
	builder.enable_depth(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
	builder.set_color_attachment_format(VK_FORMAT_UNDEFINED);
	builder.set_depth_format(depth_image.format);

	builder.dynamic_state.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS);
	builder.rasterization.depthBiasEnable = VK_TRUE;

	ShaderEffect shadow{
		.pc = {
			{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPushConstants) }
		}
	};
	shadow.build_effect(device, "../../shaders/depth.vert.spv");
	std::unique_ptr<ShaderPass> shadow_pass = vkutil::build_shader(device, &shadow, builder);

	//>
	builder.set_color_attachment_format(draw_image.format); 
	builder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	builder.disable_blending();
	builder.enable_depth(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
	builder.dynamic_state.pop_back();
	builder.rasterization.depthBiasEnable = VK_FALSE;

	ShaderEffect textured_lit{
		.layouts = { scene_descriptor_layout, bindless_tex_layout, bindless_sampler_layout },
		.pc = { 
			{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants) }
		}
	};
	textured_lit.build_effect(device, "../../shaders/mesh_pbr.vert.spv", "../../shaders/mesh_pbr.frag.spv");

	std::unique_ptr<ShaderPass> textured_lit_pass = vkutil::build_shader(device, &textured_lit, builder);

	// transparent
	//builder.enable_blending_alphablend();
	//builder.enable_depth(false, VK_COMPARE_OP_GREATER_OR_EQUAL);
	//textured_lit.build_effect(device, "../../shaders/mesh_pbr.vert.spv", "../../shaders/mesh_pbr.frag.spv"); // rebuild as above destroyed shadermodule
	//std::unique_ptr<ShaderPass> blend_pass = vkutil::build_shader(device, &textured_lit, builder);

	//>
	ShaderEffect skybox{
		.layouts = { bindless_tex_layout, bindless_sampler_layout },
		.pc = {
			{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SkyboxPushConstants)}
		}
	};
	skybox.build_effect(device, "../../shaders/skybox.vert.spv", "../../shaders/skybox.frag.spv");
	std::unique_ptr<ShaderPass> skybox_pass = vkutil::build_shader(device, &skybox, builder);

	//>
	builder.disable_depth();
	builder.set_depth_format(VK_FORMAT_UNDEFINED);

	ShaderEffect tonemap{
		.layouts = { bindless_tex_layout, bindless_sampler_layout },
		.pc = {
			{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PostFXPushConstants)}
		}
	};
	tonemap.build_effect(device, "../../shaders/full_screen.vert.spv", "../../shaders/tonemap.frag.spv");
	std::unique_ptr<ShaderPass> tonemap_pass = vkutil::build_shader(device, &tonemap, builder);

	shader_passes["equi_to_cube"] = std::move(equi_to_cube_pass);
	shader_passes["irradiance"] = std::move(irradiance_pass);
	shader_passes["prefiltered"] = std::move(prefiltered_pass);
	shader_passes["brdf"] = std::move(brdf_pass);
	shader_passes["shadow"] = std::move(shadow_pass);
	shader_passes["textured_lit"] = std::move(textured_lit_pass);
	shader_passes["skybox"] = std::move(skybox_pass);
	shader_passes["tonemap"] = std::move(tonemap_pass);
	//shader_passes["blend"] = std::move(blend_pass);
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

AllocatedImage VulkanEngine::create_image(void* data, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags /*= 0*/, bool mipmapped /*= false*/)
{
	size_t data_size = extent.depth * extent.width * extent.height * 4; // 4 is # of channels
	if (format == VK_FORMAT_R32G32B32A32_SFLOAT)
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
	uint32_t white_color = glm::packUnorm4x8(glm::vec4(1));
	white_image = create_image(static_cast<void*>(&white_color), VkExtent3D{ 1, 1, 1 },
		VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_SAMPLED_BIT, 
		VK_IMAGE_ASPECT_COLOR_BIT);

	uint32_t black_color = glm::packUnorm4x8(glm::vec4(0));
	black_image = create_image(static_cast<void*>(&black_color), VkExtent3D{ 1, 1, 1 },
		VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT);

	uint32_t metal_rough_color = glm::packUnorm4x8(glm::vec4(0, 0.5, 0, 0));
	default_mr_image = create_image(static_cast<void*>(&metal_rough_color), VkExtent3D{ 1, 1, 1 },
		VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT);

	uint32_t normal_color = glm::packUnorm4x8(glm::vec4(0.5, 0.5, 1, 0));
	default_normal_image = create_image(static_cast<void*>(&normal_color), VkExtent3D{ 1, 1, 1 },
		VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT);

	uint32_t magenta_color = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
	std::array<uint32_t, 16 * 16 > pixels{}; //for 16x16 checkerboard texture
	for (int x = 0; x < 16; x++) 
	{
		for (int y = 0; y < 16; y++) 
		{
			pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta_color : black_color;
		}
	}

	error_image = create_image(static_cast<void*>(pixels.data()), VkExtent3D{ 16, 16, 1 },
		VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT);

	bindless_texture.white = texture_cache.add_texture(white_image.view); 
	bindless_texture.black = texture_cache.add_texture(black_image.view);
	bindless_texture.metal_roughness = texture_cache.add_texture(default_mr_image.view);
	bindless_texture.normal = texture_cache.add_texture(default_normal_image.view);
	bindless_texture.checkerboard = texture_cache.add_texture(error_image.view);

	//> default samplers
	VkSampler sampler{};
	VkSamplerCreateInfo sampler_info{};
	sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sampler_info.magFilter = VK_FILTER_LINEAR;
	sampler_info.minFilter = VK_FILTER_LINEAR;

	sampler_info.maxLod = VK_LOD_CLAMP_NONE;
	sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

	vkCreateSampler(device, &sampler_info, nullptr, &sampler);
	sampler_cache.add_sampler(sampler);

	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

	vkCreateSampler(device, &sampler_info, nullptr, &sampler);
	sampler_cache.add_sampler(sampler);

	sampler_info.magFilter = VK_FILTER_NEAREST;
	sampler_info.minFilter = VK_FILTER_NEAREST;
	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

	vkCreateSampler(device, &sampler_info, nullptr, &sampler);
	sampler_cache.add_sampler(sampler);

	//> shadow map
	shadow_map = create_image(VkExtent3D{ 4096, 4096, 1 }, VK_FORMAT_D32_SFLOAT, 
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT
	);

	bindless_texture.shadow = texture_cache.add_texture(shadow_map.view);

	main_deletion_queue.push_function([&]() {
		destroy_image(white_image);
		destroy_image(black_image);
		destroy_image(default_mr_image);
		destroy_image(default_normal_image);
		destroy_image(error_image);
		destroy_image(shadow_map);
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

	scene_data.brdf_id = bindless_texture.brdf;
	scene_data.irradiance_id = bindless_texture.irradiance;
	scene_data.prefiltered_id = bindless_texture.prefiltered;
	scene_data.shadow_id = bindless_texture.shadow;

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

	//std::string asset_path = "../../assets/DamagedHelmet/GLTF-Embedded/DamagedHelmet.gltf";
	//std::string asset_path = "../../assets/ABeautifulGame.glb";
	//std::string asset_path = "../../assets/sphere.gltf";
	std::string asset_path = "../../assets/oaktree.gltf";
	auto start{ std::chrono::system_clock::now() };
	auto asset_file = load_gltf(this, asset_path, true);
	auto end{ std::chrono::system_clock::now() };
	auto elapsed{ std::chrono::duration_cast<std::chrono::microseconds>(end - start) };
	float ret = elapsed.count() / 1000.0f;
	fmt::println("load gltf: {}ms", ret);
	assert(asset_file.has_value());
	loaded_scenes["DamagedHelmet"] = *asset_file;
	asset_path = "../../assets/terrain_gridlines.gltf";
	asset_file = load_gltf(this, asset_path, true);
	loaded_scenes["terrain"] = *asset_file;
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
	variable_desc_counts[0] = sampler_cache.image_infos.size();
	bindless_sampler_descriptor = global_descriptor_allocator.allocate(device, bindless_sampler_layout, &variable_desc_info);
	variable_desc_counts[0] = image_cache.image_infos.size();
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

void VulkanEngine::register_object(Node& node, const glm::mat4& top_matrix, DrawContext& ctx)
{
	glm::mat4 node_matrix = top_matrix * node.world_transform;
	if (node.mesh != nullptr)
	{
		for (auto& s : node.mesh->surfaces)
		{
			RenderObject obj{};
			obj.index_count = s.count;
			obj.first_index = s.start_index;
			obj.index_buffer = node.mesh->mesh_buffer.index_buffer.buffer;
			obj.vertex_buffer_address = node.mesh->mesh_buffer.vertex_buffer_address;
			obj.material_buffer_address = node.mesh->material_buffer_address;
			obj.material = s.material;
			obj.material_id = s.material_id;
			obj.bounds = s.bounds;
			obj.transform = node_matrix;

			if (s.pass == MaterialPass::MainColor)
				ctx.opaque_objects.push_back(obj);
			else if (s.pass == MaterialPass::Transparent)
				ctx.transparent_objects.push_back(obj);
		}
	}

	for (auto& c : node.children)
		register_object(*c, top_matrix, ctx);
}

void VulkanEngine::update_scene()
{
	main_camera.update(stats.deltatime);

	scene_data.view = main_camera.get_view_matrix();
	scene_data.proj = main_camera.perspective;
	scene_data.viewproj = scene_data.proj * scene_data.view;
	scene_data.sunlight_dir = glm::vec4(7.75, 12.5, 12.5, 1.);
	scene_data.sunlight_color = glm::vec4(1);

	update_cascade();
	auto light_dir = glm::normalize(glm::vec3(scene_data.sunlight_dir));
	glm::mat4 shadow_view = glm::lookAt(cascade_data.center + cascade_data.radius * light_dir, cascade_data.center, glm::vec3(0, 1, 0));
	auto extent = cascade_data.radius;
	auto shadow_proj = glm::ortho(-extent, extent, -extent, extent, extent * 2.0f, 0.0f);
	cascade_data.viewproj = shadow_proj * shadow_view;
	scene_data.shadow_transform = cascade_data.viewproj;
	scene_data.camera_pos = main_camera.position;

	main_draw_context.opaque_objects.clear();
	if (main_draw_context.transparent_objects.size() != 0)
		fmt::println("transparent_objects not empty"); // (!) debug
	main_draw_context.transparent_objects.clear();

	for (auto& n : loaded_scenes["DamagedHelmet"]->top_nodes)
	{
		register_object(*n, glm::mat4(1.0), main_draw_context);
	}

	for (auto& n : loaded_scenes["terrain"]->top_nodes)
	{
		register_object(*n, glm::mat4(1.0), main_draw_context);
	}
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

void VulkanEngine::forward_pass(VkCommandBuffer cmd)
{
	vkutil::transition_image(
		cmd,
		draw_image.image,
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		VK_ACCESS_2_TRANSFER_READ_BIT,
		VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
	);

	vkutil::transition_image(
		cmd,
		depth_image.image,
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
		VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
		VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
		VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
		VK_IMAGE_ASPECT_DEPTH_BIT
	);

	VkClearColorValue clear_color_value{ 0.0f, 0.0f, 0.0f, 1.0f };
	VkClearValue clear_value{ .color = clear_color_value };
	VkRenderingAttachmentInfo color_attachment = vkinit::attachment_info(draw_image.view, &clear_value);
	VkRenderingAttachmentInfo depth_attachment = vkinit::depth_attachment_info(depth_image.view);
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

	ShaderPass current_pass = *shader_passes["textured_lit"];

	//vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_sampler_descriptor, 0, nullptr);

	// (!) capture clause for current_pass?
	auto draw = [&](const RenderObject& obj) {
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, obj.material->pipeline);
		vkCmdBindIndexBuffer(cmd, obj.index_buffer, 0, VK_INDEX_TYPE_UINT32);

		PushConstants pc{};
		pc.world_transform = obj.transform;
		pc.vertex_buffer_address = obj.vertex_buffer_address;
		pc.material_buffer_address = obj.material_buffer_address;
		pc.material_id = obj.material_id;

		vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pc);
		vkCmdDrawIndexed(cmd, obj.index_count, 1, obj.first_index, 0, 0);
	};

	for (auto& obj : main_draw_context.opaque_objects)
	{
		draw(obj);
	}

	for (auto& obj : main_draw_context.transparent_objects)
	{
		//draw(obj);
	}

	//> skybox
	current_pass = *shader_passes["skybox"];
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_sampler_descriptor, 0, nullptr);
	SkyboxPushConstants pc{};
	auto proj = main_camera.perspective;
	auto view_no_translation = glm::mat3(main_camera.get_view_matrix());
	auto view = glm::mat4(view_no_translation);
	pc.inverse_viewproj = glm::inverse(view) * glm::inverse(proj); // go in reverse order
	pc.texture_id = bindless_texture.skybox;
	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SkyboxPushConstants), &pc);
	vkCmdDraw(cmd, 3, 1, 0, 0);

	vkCmdEndRendering(cmd);

	vkutil::transition_image(
		cmd,
		draw_image.image,
		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
		VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
		VK_ACCESS_2_SHADER_READ_BIT
	);

	//> post fx
	color_attachment = vkinit::attachment_info(draw_image2.view, &clear_value);
	//depth_attachment = vkinit::depth_attachment_info(depth_image.view);
	render_info = vkinit::rendering_info(draw_extent, &color_attachment, nullptr);

	vkutil::transition_image(
		cmd,
		draw_image2.image,
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		VK_ACCESS_2_TRANSFER_READ_BIT,
		VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
	);

	vkCmdBeginRendering(cmd, &render_info);

	viewport.x = 0;
	viewport.y = static_cast<float>(draw_extent.height);
	viewport.width = static_cast<float>(draw_extent.width);
	viewport.height = -static_cast<float>(draw_extent.height);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	vkCmdSetViewport(cmd, 0, 1, &viewport);

	scissor.offset.x = 0;
	scissor.offset.y = 0;
	scissor.extent.width = draw_extent.width;
	scissor.extent.height = draw_extent.height;
	vkCmdSetScissor(cmd, 0, 1, &scissor);


	current_pass = *shader_passes["tonemap"];

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_sampler_descriptor, 0, nullptr);
	PostFXPushConstants fx_pc{};
	fx_pc.texture_id = texture_cache.get_draw_image();
	//fx_pc.texture_id = bindless_texture.shadow;
	vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PostFXPushConstants), &fx_pc);
	vkCmdDraw(cmd, 3, 1, 0, 0);

	vkCmdEndRendering(cmd);

	vkutil::transition_image(
		cmd,
		draw_image2.image,
		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
		VK_ACCESS_2_TRANSFER_WRITE_BIT
	);
}

void VulkanEngine::shadow_pass(VkCommandBuffer cmd)
{
	vkutil::transition_image(
		cmd,
		shadow_map.image,
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
		VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
		VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
		VK_ACCESS_2_SHADER_READ_BIT,
		VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
		VK_IMAGE_ASPECT_DEPTH_BIT
	);

	VkRenderingAttachmentInfo depth_attachment = vkinit::depth_attachment_info(shadow_map.view);
	VkExtent2D shadow_extent = VkExtent2D{ shadow_map.extent.width, shadow_map.extent.height };
	VkRenderingInfo render_info = vkinit::rendering_info(shadow_extent, nullptr, &depth_attachment);
	//render_info.colorAttachmentCount = 0; // implemented into above

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

	ShaderPass current_pass = *shader_passes["shadow"];

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);

	auto draw = [&](const RenderObject& obj) {
		vkCmdBindIndexBuffer(cmd, obj.index_buffer, 0, VK_INDEX_TYPE_UINT32);

		ShadowPushConstants pc{};
		pc.model = obj.transform;
		pc.viewproj = cascade_data.viewproj;
		pc.vertex_buffer_address = obj.vertex_buffer_address;

		vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPushConstants), &pc);
		vkCmdDrawIndexed(cmd, obj.index_count, 1, obj.first_index, 0, 0);
	};
	
	for (auto& obj : main_draw_context.opaque_objects)
	{
		draw(obj);
	}

	vkCmdEndRendering(cmd);

	vkutil::transition_image(
		cmd,
		shadow_map.image,
		VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
		VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
		VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		VK_ACCESS_2_SHADER_READ_BIT,
		VK_IMAGE_ASPECT_DEPTH_BIT
	);

};

void VulkanEngine::update_cascade()
{
	std::array<glm::vec3, 8> frustum_corners{
		glm::vec3(-1,  1,  0),
		glm::vec3( 1,  1,  0),
		glm::vec3(-1, -1,  0),
		glm::vec3( 1, -1,  0),
		glm::vec3(-1,  1,  1),
		glm::vec3( 1,  1,  1),
		glm::vec3(-1, -1,  1),
		glm::vec3( 1, -1,  1),
	};

	glm::mat4 view = main_camera.get_view_matrix();
	// (!) hardcoded near plane, fix
	glm::mat4 proj = glm::perspective(glm::radians(70.0f), static_cast<float>(draw_extent.width) / draw_extent.height, 30.f, 0.01f);
	glm::mat4 inv_viewproj = glm::inverse(proj * view);

	for (size_t i = 0; i < frustum_corners.size(); i++)
	{
		glm::vec4 corner = inv_viewproj * glm::vec4(frustum_corners[i], 1.0);
		frustum_corners[i] = glm::vec3(corner / corner.w);
	}

	glm::vec3 center{};
	for (size_t i = 0; i < frustum_corners.size(); i++)
	{
		center += frustum_corners[i];
	}
	center /= 8.0f;

	float radius{};
	for (size_t i = 0; i < frustum_corners.size(); i++)
	{
		radius = std::max(radius, glm::length(frustum_corners[i] - center));
	}
	cascade_data.radius = radius;

	// texel snapping - https://alextardif.com/shadowmapping.html
	float texels_per_unit = 4096.0 / (2.0 * radius); // (!) hardcoded
	glm::mat4 light_aligned_view = glm::lookAt(glm::vec3(0), -glm::normalize(glm::vec3(scene_data.sunlight_dir)), glm::vec3(0, 1, 0));
	glm::vec4 new_center = light_aligned_view * glm::vec4(center, 1.0);
	new_center.x = std::floor(new_center.x * texels_per_unit) / texels_per_unit;
	new_center.y = std::floor(new_center.y * texels_per_unit) / texels_per_unit;
	new_center = glm::inverse(light_aligned_view) * new_center;
	cascade_data.center = new_center;
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
	//VkClearColorValue clear_color_value{ 0.0f, 0.0f, 0.0f, 1.0f };
	//VkClearValue clear_value{ .color = clear_color_value };
	VkRenderingAttachmentInfo color_attachment = vkinit::attachment_info(swapchain_view, nullptr);
	VkRenderingInfo render_info = vkinit::rendering_info(swapchain_extent, &color_attachment, nullptr);
	
	vkCmdBeginRendering(cmd, &render_info);

	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
	
	vkCmdEndRendering(cmd);
}