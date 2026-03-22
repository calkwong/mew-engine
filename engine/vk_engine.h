#pragma once

#include "cache.h"
#include "camera.h"
#include "vk_descriptors.h"
#include "vk_loader.h"
#include "vk_pipelines.h"
#include "vk_scene.h"

#include <VkBootstrap.h>
#include <ranges>
// #include <tracy/TracyVulkan.hpp>

#include <array>
#include <deque>
#include <functional>
#include <string>
#include <vector>
#include <memory>

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
		for (auto& deletor : std::ranges::reverse_view(deletors))
		{
			deletor();
		}

		deletors.clear();
	}
};

struct FrameData
{
	VkCommandPool command_pool{};
	VkCommandBuffer main_command_buffer{};

	VkQueryPool query_pool_timestamps{};
	VkQueryPool query_pool_pipelines{};

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
	unsigned int triangle_count{};
	double scene_update_time{};
	double deltatime{};
	double frame_avg{};
	double early_cull{};
	double late_cull{};
	double mask_cull{};
	double early_indirect{};
	double late_indirect{};
	double mask_indirect{};
	double deferred_shading{};
	double light_culling{};
	double transparent_cull{};
	double transparent_render{};
	double shadow_cull{};
	double shadow_render{};
	double taa_resolve{};
	unsigned int cascade0{};
	unsigned int cascade1{};
	unsigned int cascade2{};
	unsigned int cascade3{};
};

struct SDL_Window;

class VulkanEngine
{
public:
	bool is_initialized{ false };
	uint32_t frame_number{ 0 };
	bool stop_rendering{ false };
	bool stop_movement{ false };
	bool freeze_camera{ false };
	bool first_frame{ true };
	bool render_imgui{ true };
	glm::mat4 last_view{};
	glm::mat4 last_proj{};

	VkExtent2D window_extent{ 1700, 900 };

	VkInstance instance{}; // vulkan library handle
	VkDebugUtilsMessengerEXT debug_messenger{}; // vulkan debug output handle
	VkPhysicalDevice chosen_gpu{};
	VkDevice device{};
	VkSurfaceKHR surface{}; // vulkan window surface

	SDL_Window* window{};

	VkSwapchainKHR swapchain{};
	VkFormat swapchain_image_format{};

	std::vector<VkImage> swapchain_images{};
	std::vector<VkImageView> swapchain_image_views{};
	VkExtent2D swapchain_extent{};

	FrameData frames[FRAME_OVERLAP]{};
	FrameData& get_current_frame() { return frames[frame_number % FRAME_OVERLAP]; }
	FrameData& get_last_frame() { return frames[(frame_number - 1) % FRAME_OVERLAP]; }
	DeletionQueue main_deletion_queue{};

	VmaAllocator allocator{};

	AllocatedImage draw_image{};
	AllocatedImage visibility_buffer{};
	AllocatedImage velocity_buffer{};
	std::array<AllocatedImage, 2> accumulation_buffers{};
	std::vector<AllocatedImage> gbuffers{};
	VkExtent2D draw_extent{};

	AllocatedImage depth_image{};
	AllocatedImage depth_pyramid{};

	VkExtent3D ibl_extent{};

	AllocatedImage white_image{};
	AllocatedImage black_image{};
	AllocatedImage default_mr_image{};
	AllocatedImage default_normal_image{};
	AllocatedImage error_image{};
	AllocatedImage offscreen_image{};

	// GI
	AllocatedImage hdri{};
	AllocatedImage hdri_cubemap{};
	AllocatedImage irradiance_cubemap{}; // for SH reference
	AllocatedImage prefiltered_envmap{};
	AllocatedImage brdf_lut{};

	AllocatedImage shadow_map{};

	AllocatedBuffer light_buffer{};
	AllocatedBuffer light_cluster_buffer{};
	AllocatedBuffer light_index_buffer{};
	AllocatedBuffer light_grid_buffer{};
	AllocatedBuffer light_count_buffer{};

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

	std::unique_ptr<LoadedGLTF> loaded_scene{};

	VkDescriptorSetLayout scene_descriptor_layout{};
	VkDescriptorSetLayout bindless_tex_layout{};
	VkDescriptorSetLayout bindless_sampler_layout{};
	VkDescriptorSetLayout bindless_image_layout{};

	VkDescriptorSet bindless_tex_descriptor{};
	VkDescriptorSet bindless_sampler_descriptor{};
	VkDescriptorSet bindless_image_descriptor{};

	std::unordered_map<std::string, std::unique_ptr<ShaderPass>> shader_passes{};

	SceneData scene_data{};
	std::array<CascadeData, 4> cascade_data{};
	std::array<float, 4> jx{};
	std::array<float, 4> jy{};
	std::vector<VkImageMemoryBarrier2> image_barriers{};
	std::vector<VkMemoryBarrier2> buffer_barriers{};

	// tracy::VkCtx* tracy_ctx{};
	RenderScene render_scene{};

	VkPhysicalDeviceProperties device_properties{};

	static VulkanEngine& get();

	void init(const std::string& file_path);
	void cleanup();
	void draw();
	void run();

	void update_scene();

	void register_object(const Node* node, const glm::mat4& top_matrix);
	void execute_deferred_shading(VkCommandBuffer cmd, VkImageView view);
	void execute_taa_resolve(VkCommandBuffer cmd, VkImageView view);
	void update_cascade();
	void draw_imgui(VkCommandBuffer cmd, VkImageView swapchain_view);
	void upload_buffers();
	void ready_mesh_cull(RenderScene::MeshPass& pass, CullData& cull_data, glm::mat4& proj, bool orthographic = false);
	void ready_meshlet_cull(RenderScene::MeshPass& pass, ClusterCullData& cull_data, glm::mat4& proj, bool orthographic = false);
	void execute_compute_cull(VkCommandBuffer cmd, const RenderScene::MeshPass& pass, CullData& cull_data, bool late, uint32_t post_pass);
	void execute_compute_cull(VkCommandBuffer cmd, RenderScene::MeshPass& pass, ClusterCullData& cull_data, VkBuffer count_buffer, uint32_t offset, bool late, uint32_t post_pass);
	void execute_shadow_cull(VkCommandBuffer cmd);
	void render(VkCommandBuffer cmd, bool late, uint32_t post_pass, uint32_t query);
	void render_transparent(VkCommandBuffer cmd, uint32_t query);
	void render_shadows(VkCommandBuffer cmd, uint32_t cascade_idx, uint32_t query);
	void build_depth_pyramid(VkCommandBuffer cmd);
	void execute_light_culling(VkCommandBuffer cmd);

private:
	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_sync_structures();
	void init_descriptors();
	void init_pipelines();
	void init_default_data();
	void init_renderables(const std::string& file_path);
	void init_bindless();
	void init_gi();
	void init_imgui();
	void build_cluster_grid();

	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();
};

// TODO: move these to math/utility haeder
uint32_t nearest_pow2(uint32_t extent);
float Halton(uint32_t i, uint32_t b);
uint32_t get_groupcount(uint32_t size, uint32_t threads);
float size_in_bytes(uint64_t size);
void decompose_transform(const glm::mat4& m, glm::vec3& translation, glm::vec3& scale, glm::vec4& rotation);