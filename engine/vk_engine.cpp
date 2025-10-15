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
#include "vk_mem_alloc.h"

#include <glm/gtx/transform.hpp>
#include "stb_image.h"

#include <thread>

VulkanEngine* loaded_engine{};

VulkanEngine& VulkanEngine::get() { return *loaded_engine; }

constexpr bool USE_VALIDATION_LAYERS = true;


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

	init_bindless_textures();

	main_camera.position = glm::vec3(0, 0, 5);

	is_initialized = true;
}

void VulkanEngine::cleanup()
{
	if (is_initialized) {

		vkDeviceWaitIdle(device);
		loaded_scenes.clear();

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

	uint32_t swapchain_image_idx{};
	VK_CHECK(vkAcquireNextImageKHR(device, swapchain, 1000000000, get_current_frame().swapchain_semaphore, nullptr, &swapchain_image_idx));

	VkCommandBuffer cmd = get_current_frame().main_command_buffer;

	VK_CHECK(vkResetCommandBuffer(cmd, 0));

	VkCommandBufferBeginInfo cmd_begin_info = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT); // (!)

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));

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

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 0, 1, &get_current_frame().scene_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 1, 1, &bindless_tex_descriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pass.layout, 2, 1, &bindless_sampler_descriptor, 0, nullptr);

	// (!) capture clause for current_pass?
	auto draw = [&](const RenderObject& obj) {
		vkCmdBindIndexBuffer(cmd, obj.index_buffer, 0, VK_INDEX_TYPE_UINT32);

		PushConstants pc{};
		pc.world_transform = obj.transform;
		pc.vertex_buffer_address = obj.vertex_buffer_address;
		pc.material_buffer_address = loaded_scenes["DamagedHelmet"]->material_buffer_address;
		pc.material_id = obj.material_id;

		vkCmdPushConstants(cmd, current_pass.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pc);
		vkCmdDrawIndexed(cmd, obj.index_count, 1, obj.first_index, 0, 0);
	};

	for (auto& obj : main_draw_context.opaque_objects)
	{
		draw(obj);
	}

	vkCmdEndRendering(cmd);

	vkutil::transition_image(
		cmd,
		draw_image.image,
		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
		VK_ACCESS_2_TRANSFER_READ_BIT
	);

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

	vkutil::copy_image(cmd, draw_image.image, swapchain_images[swapchain_image_idx], draw_extent, swapchain_extent);

	vkutil::transition_image(
		cmd,
		swapchain_images[swapchain_image_idx],
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
		VK_ACCESS_2_TRANSFER_WRITE_BIT,
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
        }

        if (stop_rendering) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        draw();

		//auto end = std::chrono::system_clock::now();

		//auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
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

	// draw image
	VkExtent3D draw_image_extent{ window_extent.width, window_extent.height, 1 };

	draw_image.format = VK_FORMAT_R16G16B16A16_SFLOAT;
	draw_image.extent = draw_image_extent;

	VkImageUsageFlags draw_image_flags{
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
		VK_IMAGE_USAGE_TRANSFER_SRC_BIT
	};

	VkImageCreateInfo img_create_info = vkinit::image_create_info(draw_image.format, draw_image_flags, draw_image.extent);

	VmaAllocationCreateInfo vma_allocation_info{};
	vma_allocation_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	vma_allocation_info.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	vmaCreateImage(allocator, &img_create_info, &vma_allocation_info, &draw_image.image, &draw_image.allocation, nullptr);

	VkImageViewCreateInfo imgview_create_info = vkinit::imageview_create_info(draw_image.format, draw_image.image, VK_IMAGE_ASPECT_COLOR_BIT);
	VK_CHECK(vkCreateImageView(device, &imgview_create_info, nullptr, &draw_image.view));

	// depth image
	depth_image.format = VK_FORMAT_D32_SFLOAT;
	depth_image.extent = draw_image_extent;

	VkImageUsageFlagBits depth_image_flags{
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
	};

	img_create_info = vkinit::image_create_info(depth_image.format, depth_image_flags, depth_image.extent);

	vmaCreateImage(allocator, &img_create_info, &vma_allocation_info, &depth_image.image, &depth_image.allocation, nullptr);
	
	imgview_create_info = vkinit::imageview_create_info(depth_image.format, depth_image.image, VK_IMAGE_ASPECT_DEPTH_BIT);
	VK_CHECK(vkCreateImageView(device, &imgview_create_info, nullptr, &depth_image.view));

	main_deletion_queue.push_function([&]() {
		vkDestroyImageView(device, draw_image.view, nullptr);
		vmaDestroyImage(allocator, draw_image.image, draw_image.allocation);
		vkDestroyImageView(device, depth_image.view, nullptr);
		vmaDestroyImage(allocator, depth_image.image, depth_image.allocation);
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

	swapchain_image_format = VK_FORMAT_B8G8R8A8_UNORM;
	//swapchain_image_format = VK_FORMAT_B8G8R8A8_SRGB;

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
		{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
		{VK_DESCRIPTOR_TYPE_SAMPLER, 10}
	};

	global_descriptor_allocator.init(device, 1, sizes);

	//> building bindless layouts
	{
		DescriptorLayoutBuilder builder{};
		builder.add_binding(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT);
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
		builder.add_binding(0, VK_DESCRIPTOR_TYPE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
		builder.bindings[0].descriptorCount = 10; // (!) validation layer not reporting if this is higher than pool maximum

		bindless_sampler_layout = builder.build(device, &binding_flags_info); // (!) update after bind req if included above?
	}

	main_deletion_queue.push_function([&]() {
		global_descriptor_allocator.destroy_pools(device);
		vkDestroyDescriptorSetLayout(device, scene_descriptor_layout, nullptr);
		vkDestroyDescriptorSetLayout(device, bindless_tex_layout, nullptr);
		vkDestroyDescriptorSetLayout(device, bindless_sampler_layout, nullptr);
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
	// layout and shaders set later
	PipelineBuilder builder{};
	builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
	builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
	builder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	builder.set_multisampling_none();
	builder.disable_blending();
	builder.disable_depth(); // still need to set depth format?
	builder.set_depth_format(VK_FORMAT_UNDEFINED);
	builder.set_color_attachment_format(draw_image.format);

	//>
	ShaderEffect equi_to_cube{
		.layouts = { bindless_tex_layout, bindless_sampler_layout },
		.pc = {
			{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(uint32_t) * 2 }
		}
	};

	equi_to_cube.build_effect(device, "../../shaders/full_screen.vert.spv", "../../shaders/equi_to_cube.frag.spv");

	std::unique_ptr<ShaderPass> equi_to_cube_pass = vkutil::build_shader(device, &equi_to_cube, builder);

	//>
	ShaderEffect textured_lit{
		.layouts = { scene_descriptor_layout, bindless_tex_layout, bindless_sampler_layout },
		.pc = { 
			{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants) }
		}
	};
	textured_lit.build_effect(device, "../../shaders/mesh_pbr.vert.spv", "../../shaders/mesh_pbr.frag.spv");

	builder.enable_depth(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
	builder.set_depth_format(depth_image.format);
	std::unique_ptr<ShaderPass> textured_lit_pass = vkutil::build_shader(device, &textured_lit, builder);

	//>
	ShaderEffect skybox{
		.layouts = { bindless_tex_layout, bindless_sampler_layout },
		.pc = {
			{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::mat4) + sizeof(uint32_t)}
		}
	};
	skybox.build_effect(device, "../../shaders/skybox.vert.spv", "../../shaders/skybox.frag.spv");
	std::unique_ptr<ShaderPass> skybox_pass = vkutil::build_shader(device, &skybox, builder);

	shader_passes["equi_to_cube"] = std::move(equi_to_cube_pass);
	shader_passes["textured_lit"] = std::move(textured_lit_pass);
	shader_passes["skybox"] = std::move(skybox_pass);
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

	VkSamplerCreateInfo sampler_info{};
	sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sampler_info.magFilter = VK_FILTER_LINEAR;
	sampler_info.minFilter = VK_FILTER_LINEAR;

	sampler_info.maxLod = VK_LOD_CLAMP_NONE;
	sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

	vkCreateSampler(device, &sampler_info, nullptr, &default_linear_sampler);

	sampler_info.magFilter = VK_FILTER_NEAREST;
	sampler_info.minFilter = VK_FILTER_NEAREST;

	vkCreateSampler(device, &sampler_info, nullptr, &default_nearest_sampler);

	texture_cache.add_texture(white_image.view); 
	texture_cache.add_texture(black_image.view);
	texture_cache.add_texture(default_mr_image.view);
	texture_cache.add_texture(default_normal_image.view);
	texture_cache.add_texture(error_image.view);

	main_deletion_queue.push_function([&]() {
		destroy_image(white_image);
		destroy_image(black_image);
		destroy_image(default_mr_image);
		destroy_image(default_normal_image);
		destroy_image(error_image);
		vkDestroySampler(device, default_linear_sampler, nullptr);
		vkDestroySampler(device, default_nearest_sampler, nullptr);
	});
}

void VulkanEngine::init_renderables()
{
	std::string asset_path = "../../assets/DamagedHelmet/GLTF-Embedded/DamagedHelmet.gltf";
	//std::string asset_path = "../../assets/sphere.gltf";
	auto asset_file = load_gltf(this, asset_path, true);
	assert(asset_file.has_value());
	loaded_scenes["DamagedHelmet"] = *asset_file;
}

void VulkanEngine::init_bindless_textures()
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
	bindless_sampler_descriptor = global_descriptor_allocator.allocate(device, bindless_sampler_layout, &variable_desc_info);


	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = bindless_tex_descriptor;
	write.dstBinding = 0;
	write.descriptorCount = static_cast<uint32_t>(texture_cache.image_infos.size()); // (!) validation layer does not report if smaller count than req used; fragment sample simply returns black
	write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
	write.pImageInfo = texture_cache.image_infos.data();

	vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

	// samplers
	variable_desc_counts[0] = 1;
	write.dstSet = bindless_sampler_descriptor;
	write.descriptorCount = 1;
	write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
	VkDescriptorImageInfo image_info{};
	image_info.sampler = default_linear_sampler;
	write.pImageInfo = &image_info;

	vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
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
			obj.material_id = s.material_id;
			obj.bounds = s.bounds;
			obj.transform = node_matrix;

			ctx.opaque_objects.push_back(obj);
		}
	}

	for (auto& c : node.children)
		register_object(*c, top_matrix, ctx);
}

void VulkanEngine::update_scene()
{
	main_camera.update(stats.deltatime);

	// (!) update scene data; any sync needed? 
	SceneData* scene_uniform_data = static_cast<SceneData*>(get_current_frame().scene_buffer.info.pMappedData);

	SceneData updated_data{};
	updated_data.view = main_camera.get_view_matrix();
	updated_data.proj = glm::perspective(glm::radians(60.0f), static_cast<float>(draw_extent.width) / draw_extent.height, 10000.0f, 0.1f);
	updated_data.viewproj = updated_data.proj * updated_data.view;
	updated_data.camera_pos = main_camera.position;

	*scene_uniform_data = updated_data;

	main_draw_context.opaque_objects.clear();
	for (auto& n : loaded_scenes["DamagedHelmet"]->top_nodes)
	{
		register_object(*n, glm::mat4(1.0), main_draw_context);
	}

}

uint32_t TextureCache::add_texture(const VkImageView& view)
{
	for (size_t i = 0; i < image_infos.size(); i++)
	{
		if (image_infos[i].imageView == view) // TODO: implement sampler check
			return static_cast<uint32_t>(i);
	}

	uint32_t id = static_cast<uint32_t>(image_infos.size());

	image_infos.emplace_back(VkDescriptorImageInfo{ 0, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL });

	return id;
}