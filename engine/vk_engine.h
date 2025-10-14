#pragma once

#include <vk_types.h>
#include <vk_initializers.h>
#include <vk_descriptors.h>
#include <camera.h>
#include <vk_loader.h>

#include "VkBootstrap.h"

constexpr unsigned int FRAME_OVERLAP = 2;


struct DeletionQueue
{
	std::deque<std::function<void()>> deletors{};

	void push_function(std::function<void()>&& function) 
	{ 
		deletors.push_back(function); 
	}
	
	void flush()
	{
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++)
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

	VkSemaphore swapchain_semaphore{};
	VkSemaphore render_semaphore{};
	VkFence render_fence{};

	DescriptorAllocatorGrowable frame_descriptor_allocator{};
	AllocatedBuffer scene_buffer{};
	VkDescriptorSet scene_ds{};

	DeletionQueue deletion_queue{};

};

struct EngineStats
{
	//float frameTime{};
	//int triangleCount{};
	//int drawCallCount{};
	//float sceneUpdateTime{};
	//float meshDrawTime{};

	float deltatime{};
	float last_frame{};
};

// destruction of textures handled by gltf (not internally); does not support dynamic objs
struct TextureCache
{
	std::vector<VkDescriptorImageInfo> image_infos{};
	const uint32_t PLACEHOLDERS{ 5 };

	uint32_t add_texture(const VkImageView& view, VkSampler sampler);
};

// consider moving to vk_scene
// should there be material_buffer_address? or is that in pass object? currently handled by push constant
struct RenderObject
{
	uint32_t index_count{};
	uint32_t first_index{};
	VkBuffer index_buffer{};

	uint32_t material_id{};
	Bounds bounds{};

	glm::mat4 transform{};
	VkDeviceAddress vertex_buffer_address{};
};

class VulkanEngine
{
public:
	bool is_initialized{ false };
	int frame_number{ 0 };
	bool stop_rendering{ false };
	bool stop_movement{ false };
	VkExtent2D window_extent{ 1700, 900 };

	VkInstance instance{}; // vulkan library handle
	VkDebugUtilsMessengerEXT debug_messenger{}; // vulkan debug output handle
	VkPhysicalDevice chosen_gpu{};
	VkDevice device{};
	VkSurfaceKHR surface{}; // vulkan window surface

	struct SDL_Window* window{};

	VkSwapchainKHR swapchain{};
	VkFormat swapchain_image_format{};

	std::vector<VkImage> swapchain_images{};
	std::vector<VkImageView> swapchain_image_views{};
	VkExtent2D swapchain_extent{};

	FrameData frames[FRAME_OVERLAP]{};
	FrameData& get_current_frame() { return frames[frame_number % FRAME_OVERLAP]; };
	DeletionQueue main_deletion_queue{};

	VmaAllocator allocator{};

	AllocatedImage draw_image{};
	VkExtent2D draw_extent{};

	AllocatedImage depth_image{};
	VkExtent2D depth_extent{};

	AllocatedImage white_image{};
	AllocatedImage black_image{};
	AllocatedImage default_mr_image{};
	AllocatedImage default_normal_image{};
	AllocatedImage error_image{};
	VkSampler default_linear_sampler{};
	VkSampler default_nearest_sampler{};

	DescriptorAllocatorGrowable global_descriptor_allocator{};

	VkFence imm_fence{};
	VkCommandBuffer imm_command_buffer{};
	VkCommandPool imm_command_pool{};

	VkQueue graphics_queue{};
	uint32_t graphics_queue_family{};

	DrawContext main_draw_context{};
	Camera main_camera{};
	EngineStats stats{};

	TextureCache texture_cache{};

	std::unordered_map<std::string, std::shared_ptr<LoadedGLTF>> loaded_scenes{};

	// (!) temp, refactor and move elsewhere?
	VkDescriptorSetLayout scene_ds_layout{};
	VkPipeline pbr_pipeline{};
	VkPipelineLayout pbr_pipeline_layout{};
	VkDescriptorSet pbr_ds{};
	VkDescriptorSetLayout pbr_ds_layout{};

	// (!) currently exceeds hardware min size 128 bytes
	struct PBRPushConstants
	{
		glm::mat4 world_transform{};
		//glm::mat4 viewproj{};
		//glm::vec4 camera_position{};
		VkDeviceAddress vertex_buffer_address{};
		VkDeviceAddress material_buffer_address{};
		uint32_t material_id{};
	};
	static VulkanEngine& get();

	void init();
	void cleanup();
	void draw();
	void run();
	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& func);
	AllocatedBuffer create_buffer(size_t alloc_size, VmaAllocationCreateFlags flags, VkBufferUsageFlags usage);
	void destroy_buffer(const AllocatedBuffer& buffer);
	GPUMeshBuffers upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices);
	// view has access to all mip and layers
	AllocatedImage create_image(VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags = 0, bool mipmapped = false); // does not currently handle priority
	AllocatedImage create_image(void* data, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags = 0, bool mipmapped = false);
	void destroy_image(const AllocatedImage& image);

	void register_object(Node& node, const glm::mat4& top_matrix, DrawContext& ctx);
	void update_scene();

private:
	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_sync_structures();
	void init_descriptors();
	void init_pipelines();
	void init_default_data();
	void init_renderables();
	void init_bindless_textures(); // MAYBE: split this up and have partial writes/updates 

	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();
};