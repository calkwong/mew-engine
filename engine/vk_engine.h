#pragma once

#include <vk_types.h>
#include <vk_descriptors.h>
#include <vk_loader.h>
#include <vk_scene.h>
#include <camera.h>

#include "VkBootstrap.h"
#include "tracy/TracyVulkan.hpp"
#include <vulkan/vulkan.h>

#include <span>
#include <functional>
#include <vector>
#include <array>
#include <deque>
#include <string>

constexpr unsigned int FRAME_OVERLAP = 2;

// (!) hardware min size 128 bytes
struct PushConstants
{
	glm::mat4 world_transform{};
	VkDeviceAddress vertex_buffer_address{};
	VkDeviceAddress material_buffer_address{};
	uint32_t material_id{};
	uint32_t debug_idx{};
};

struct GPUPushConstants
{
	VkDeviceAddress material_buffer_address{};
	VkDeviceAddress object_buffer_address{};
	VkDeviceAddress instance_buffer_address{};
	VkDeviceAddress vertex_buffer_address{};
};

struct IBLPushConstants
{
	uint32_t texture_id{};
	uint32_t image_id{};
	float roughness{};
};

//struct ShadowPushConstants
//{
//	glm::mat4 model{};
//	glm::mat4 viewproj{};
//	VkDeviceAddress vertex_buffer_address{};
//	VkDeviceAddress material_buffer_address{};
//	uint32_t material_id;
//};

struct ShadowPushConstants
{
	glm::mat4 viewproj{};
	VkDeviceAddress material_buffer_address{};
	VkDeviceAddress object_buffer_address{};
	VkDeviceAddress instance_buffer_address{};
	VkDeviceAddress vertex_buffer_address{};
};

struct SkyboxPushConstants
{
	glm::mat4 inverse_viewproj{};
	uint32_t texture_id{};
};

struct PostFXPushConstants
{
	uint32_t texture_id{};
};

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
	VkDescriptorSet scene_descriptor{};

	DeletionQueue deletion_queue{};
};

struct EngineStats
{
	int draw_count{};
	float scene_update_time{};
	float deltatime{};
};

// destruction of textures handled by gltf (not internally); does not support dynamic objs
struct TextureCache
{
	std::vector<VkDescriptorImageInfo> image_infos{};

	uint32_t add_texture(const VkImageView& view);

	// (!) refactor
	void set_draw_image(uint32_t id) { draw_id = id; };
	void set_draw_image2(uint32_t id) { draw_id2 = id; };
	uint32_t get_draw_image() { return draw_id; };
	uint32_t get_draw_image2() { return draw_id2; };

private:
	uint32_t draw_id{};
	uint32_t draw_id2{};
};

struct SamplerCache
{
	std::vector<VkDescriptorImageInfo> image_infos{};

	// (!) currently performs no checking
	void add_sampler(const VkSampler& sampler);
};

struct ImageCache
{
	std::vector<VkDescriptorImageInfo> image_infos{};

	uint32_t add_texture(const VkImageView& view);
};

struct ShaderCache
{
	std::unordered_map<std::string, VkShaderModule> data{};

	VkShaderModule add_shader(VkDevice device, const char* path);
};

struct ShaderPass;

struct MaterialCache
{
	std::vector<Material> data{};

	uint32_t add_material(ShaderPass* forward, ShaderPass* shadow);
};

// to refactor
struct BindlessTexture
{
	uint8_t checkerboard{};
	uint8_t equi{};
	uint8_t skybox{};
	uint8_t irradiance{};
	uint8_t prefiltered{};
	uint8_t brdf{};
	uint8_t shadow{};
};

struct BindlessImage
{
	uint8_t skybox{};
	uint8_t irradiance{};
	uint8_t prefiltered{};
	uint8_t brdf{};
};

class VulkanEngine
{
public:
	bool is_initialized{ false };
	uint32_t frame_number{ 0 };
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
	AllocatedImage draw_image2{};
	VkExtent2D draw_extent{};

	AllocatedImage depth_image{};
	VkExtent2D depth_extent{};

	VkExtent3D ibl_extent{};

	AllocatedImage white_image{};
	AllocatedImage black_image{};
	AllocatedImage default_mr_image{};
	AllocatedImage default_normal_image{};
	AllocatedImage error_image{};
	AllocatedImage offscreen_image{};
	AllocatedImage equirectangular_image{};
	AllocatedImage cubemap_image{};
	AllocatedImage irradiance_image{};
	AllocatedImage prefiltered_image{};
	AllocatedImage brdflut_image{};

	AllocatedImage shadow_map{};


	VkSampler default_linear_sampler{};
	VkSampler default_cube_sampler{};
	VkSampler default_nearest_sampler{};

	DescriptorAllocatorGrowable global_descriptor_allocator{};

	VkFence imm_fence{};
	VkCommandBuffer imm_command_buffer{};
	VkCommandPool imm_command_pool{};

	VkQueue graphics_queue{};
	uint32_t graphics_queue_family{};

	Camera main_camera{};
	EngineStats stats{};

	SamplerCache sampler_cache{}; 
	TextureCache texture_cache{};
	ImageCache image_cache{};
	ShaderCache shader_cache{};
	MaterialCache material_cache{};

	std::unordered_map<std::string, std::shared_ptr<LoadedGLTF>> loaded_scenes{};

	VkDescriptorSetLayout scene_descriptor_layout{};
	VkDescriptorSetLayout bindless_tex_layout{};
	VkDescriptorSetLayout bindless_sampler_layout{};
	VkDescriptorSetLayout bindless_image_layout{};

	VkDescriptorSet bindless_tex_descriptor{};
	VkDescriptorSet bindless_sampler_descriptor{};
	VkDescriptorSet bindless_image_descriptor{};

	std::unordered_map<std::string, std::unique_ptr<ShaderPass>> shader_passes{};

	BindlessTexture bindless_texture{};
	BindlessImage bindless_image{};

	SceneData scene_data{};
	std::array<CascadeData, 4> cascade_data{};

	tracy::VkCtx* tracy_ctx{};
	RenderScene render_scene{};

	static VulkanEngine& get();

	void init();
	void cleanup();
	void draw();
	void run();
	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& func);
	AllocatedBuffer create_buffer(size_t alloc_size, VmaAllocationCreateFlags flags, VkBufferUsageFlags usage);
	AllocatedBuffer reallocate_buffer(size_t alloc_size, AllocatedBuffer old_buffer, VmaAllocationCreateFlags flags, VkBufferUsageFlags usage);
	void destroy_buffer(const AllocatedBuffer& buffer);
	GPUMeshBuffers upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices);
	// view has access to all mip and layers
	AllocatedImage create_image(VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags = 0, bool mipmapped = false); // does not currently handle priority
	AllocatedImage create_image(void* data, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags = 0, bool mipmapped = false);
	AllocatedImage create_cubemap(VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags = 0, bool mipmapped = false); 
	void destroy_image(const AllocatedImage& image);

	void update_scene();

	void register_object(Node* node, const glm::mat4& top_matrix);
	void forward_pass(VkCommandBuffer cmd);
	void shadow_pass(VkCommandBuffer cmd, RenderScene::MeshPass& pass, size_t cascade_idx);
	void update_cascade();
	void draw_imgui(VkCommandBuffer cmd, VkImageView swapchain_view);
	void ready_mesh_draw();
	CullData ready_cull_data(RenderScene::MeshPass& pass, glm::mat4& viewproj, bool orthographic = false);
	void execute_compute_cull(VkCommandBuffer cmd, RenderScene::MeshPass& pass, CullData& cull_data);
	void execute_compact_indirect(VkCommandBuffer cmd, RenderScene::MeshPass& pass);

private:
	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_sync_structures();
	void init_descriptors();
	void init_pipelines();
	void init_default_data();
	void init_renderables();
	void init_bindless(); 
	void init_precomputations();
	void init_imgui();

	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();
};