#pragma once

#include "cache.h"
#include "inputs.h"
#include "vk_descriptors.h"
#include "vk_loader.h"
#include "vk_pipelines.h"
#include "vk_scene.h"

#include <VkBootstrap.h>
#include <ranges>

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

	VkSemaphore image_acquired_semaphore{};
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
	double deltatime{};
	double cpu_time{};
	double gpu_time{};
	double early_cull{};
	double late_cull{};
	double mask_cull{};
	double hiz{};
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
struct CullData;
struct ClusterCullData;

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
	bool reload_shaders{ false };
	glm::mat4 last_view{};
	glm::mat4 last_proj{};

	VkExtent2D window_extent{ 1700, 900 };

	VkInstance instance{}; // vulkan library handle
	VkDebugUtilsMessengerEXT debug_messenger{}; // vulkan debug output handle
	VkPhysicalDevice chosen_gpu{};
	VkSurfaceKHR surface{}; // vulkan window surface
	SDL_Window* window{};

	VkQueue graphics_queue{};
	uint32_t graphics_queue_family{};
	VkPhysicalDeviceProperties device_properties{};
	VkDevice device{};

	VkSwapchainKHR swapchain{};
	VkFormat swapchain_image_format{};

	std::vector<VkImage> swapchain_images{};
	std::vector<VkImageView> swapchain_image_views{};
	VkExtent2D swapchain_extent{};
	VkExtent2D draw_extent{};

	FrameData frames[FRAME_OVERLAP]{};
	FrameData& get_current_frame() { return frames[frame_number % FRAME_OVERLAP]; }
	std::vector<VkSemaphore> render_done_semaphores{};
	DeletionQueue main_deletion_queue{};

	DescriptorAllocatorGrowable global_descriptor_allocator{};
	VkDescriptorSetLayout scene_descriptor_layout{};
	VkDescriptorSetLayout bindless_tex_layout{};
	VkDescriptorSetLayout bindless_sampler_layout{};
	VkDescriptorSetLayout bindless_image_layout{};
	VkDescriptorSetLayout rasterizer_ordered_buf_layout{};
	VkDescriptorSetLayout as_layout{};

	VmaAllocator allocator{};

	VkFence imm_fence{};
	VkCommandBuffer imm_command_buffer{};
	VkCommandPool imm_command_pool{};

	std::unique_ptr<LoadedGLTF> loaded_scene{};

	Camera main_camera{};
	SceneData scene_data{};
	std::array<CascadeData, 4> cascade_data{};
	std::array<glm::vec2, 8> jitter_offset{};
	EngineStats stats{};

	SamplerCache sampler_cache{};
	TextureCache texture_cache{};
	ImageCache image_cache{};
	ShaderCache shader_cache{};

	VkDescriptorSet bindless_tex_descriptor{};
	VkDescriptorSet bindless_sampler_descriptor{};
	VkDescriptorSet bindless_image_descriptor{};
	VkDescriptorSet rasterizer_ordered_buf_descriptor{};
	VkDescriptorSet as_descriptor{};

	std::unordered_map<std::string, std::unique_ptr<ShaderPass>> shader_passes{};

	std::vector<VkImageMemoryBarrier2> image_barriers{};
	std::vector<VkMemoryBarrier2> buffer_barriers{};

	AllocatedImage draw_image{};
	AllocatedImage depth_image{};
	AllocatedImage visibility_buffer{};
	AllocatedImage velocity_buffer{};
	std::array<AllocatedImage, 2> accumulation_buffers{};
	std::vector<AllocatedImage> gbuffers{};
	AllocatedImage depth_pyramid{};
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

	AllocatedBuffer blas_buffer{};
	AllocatedBuffer tlas_buffer{};
	AllocatedBuffer tlas_instance_buffer{};

	VkAccelerationStructureKHR tlas_as{};

	RenderScene render_scene{};

	// tracy::VkCtx* tracy_ctx{};

	static VulkanEngine& get();

	void init(int argc, char** argv);
	void cleanup();
	void draw();
	void run();

	void update_scene();
	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& func) const;
	void register_object(const Node* node, const glm::mat4& top_matrix);
	void resolve_taa(VkCommandBuffer cmd);
	void update_cascade();
	void draw_imgui(VkCommandBuffer cmd, VkImageView swapchain_view);
	void upload_buffers();
	void ready_mesh_cull(RenderScene::MeshPass& pass, CullData& cull_data, glm::mat4& proj);
	void ready_meshlet_cull(RenderScene::MeshPass& pass, ClusterCullData& cull_data, glm::mat4& proj);
	void execute_compact_dispatch(VkCommandBuffer cmd);
	void execute_compute_cull(VkCommandBuffer cmd, const RenderScene::MeshPass& pass, CullData& cull_data, bool late, uint32_t post_pass);
	void execute_compute_cull(VkCommandBuffer cmd, RenderScene::MeshPass& pass, ClusterCullData& cull_data, VkBuffer count_buffer, uint32_t offset, bool late, uint32_t post_pass);
	void execute_shadow_cull(VkCommandBuffer cmd);
	void render(VkCommandBuffer cmd, bool late, uint32_t post_pass, uint32_t query);
	void render_transparent(VkCommandBuffer cmd, uint32_t query);
	void render_shadows(VkCommandBuffer cmd, uint32_t cascade_idx, uint32_t query);
	void execute_spd(VkCommandBuffer cmd);
	void build_depth_pyramid(VkCommandBuffer cmd);
	void execute_light_culling(VkCommandBuffer cmd);
	void execute_shading(VkCommandBuffer cmd);
	void create_acceleration_structures();

private:
	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_sync_structures();
	void init_descriptors();
	void init_shaders();
	void init_pipelines();
	void init_resources();
	void init_renderables(int argc, char** argv);
	void update_descriptors();
	void execute_baked_gi();
	void init_imgui();
	void build_cluster_grid();

	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();
};
