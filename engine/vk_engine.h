#pragma once

#include "vk_types.h"
#include "vk_descriptors.h"
#include "vk_loader.h"
#include "vk_scene.h"
#include "camera.h"

#include <VkBootstrap.h>
#include <tracy/TracyVulkan.hpp>
#include <vulkan/vulkan.h>

#include <span>
#include <functional>
#include <vector>
#include <array>
#include <deque>
#include <string>

constexpr unsigned int FRAME_OVERLAP = 2;

struct IBLPushConstants
{
	uint32_t texture_id{};
	uint32_t image_id{};
	float roughness{};
};

struct GPUPushConstants // temporarily shared by vertex and mesh shading path
{
	VkDeviceAddress object_buffer_address{};
	VkDeviceAddress vertex_buffer_address{};
	VkDeviceAddress meshtask_buffer_address{};
	VkDeviceAddress meshlet_buffer_address{};
	VkDeviceAddress meshlet_indices_buffer_address{};
	VkDeviceAddress cluster_indices_address{};
	VkDeviceAddress material_buffer_address{};
	VkDeviceAddress oit_buffer_address{};
	uint32_t debug_meshlets;
};

struct DeferredPushConstants
{
	glm::vec4 cluster_size{}; // xyz are cluster data structure dimensions, w is a single cluster's dimension
	glm::vec2 screen_size{};
	VkDeviceAddress light_buffer_address{};
	VkDeviceAddress light_index_buffer_address{};
	VkDeviceAddress light_grid_buffer_address{};
	VkDeviceAddress oit_buffer_address{};
	uint32_t depth_id{};
	uint32_t albedo_id{};
	uint32_t normal_id{};
	uint32_t world_pos_id{};
	uint32_t shadow_id{};
	uint32_t light_culling{}; // for toggling light culling between naive and proper implementation
	float near{};
	float scale{}; 
	float bias{};  
	uint32_t debug_meshlets{};
	uint32_t resolve_transparent{};
	uint32_t shadows{};
	uint32_t pcf{};
	uint32_t debug_shadowmap{};
	uint32_t debug_cascades{};
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
	VkDeviceAddress vertex_buffer_address{};
};

struct SkyboxPushConstants
{
	glm::mat4 inverse_viewproj{};
	uint32_t texture_id{};
};

struct DebugPushConstants
{
	uint32_t texture_id{};
	uint32_t lod{}; // depth pyramid lod
};

struct DepthPyramidPushConstants
{
	std::array<int32_t, 2> image_size{};
	uint32_t texture_id{};
	uint32_t image_id{};
	uint32_t lod{};
};

struct ClusterGridPushConstants
{
	glm::mat4 inverse_proj{};
	glm::vec4 cluster_size{};
	glm::vec2 screen_size{};
	float near{};
	float far{};
	VkDeviceAddress light_cluster_buffer_address{};
};

struct LightCullingPushConstants
{
	glm::mat4 view{};
	glm::mat4 light_rot{};
	VkDeviceAddress light_cluster_buffer_address{};
	VkDeviceAddress light_buffer_address{};
	VkDeviceAddress light_index_buffer_address{};
	VkDeviceAddress light_grid_buffer_address{};
	VkDeviceAddress light_count_buffer_address{};
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
	unsigned int cascade0{};
	unsigned int cascade1{};
	unsigned int cascade2{};
	unsigned int cascade3{};
};

// destruction of textures handled by gltf (not internally); does not support dynamic objs
struct TextureCache
{
	std::vector<VkDescriptorImageInfo> image_infos{};

	uint32_t add_texture(const VkImageView& view);

	// TODO: refactor
	void set_draw_image(uint32_t id) { draw_id = id; };
	void set_gbuffers(uint32_t id) { gbuffer_id = id; };
	void set_depth_image(uint32_t id) { depth_id = id; };
	void set_depth_pyramid_image(uint32_t id) { depth_pyramid_id = id; };
	void set_shadowmap(uint32_t id) { shadowmap_id = id; };
	uint32_t get_draw_image() { return draw_id; };
	uint32_t get_first_gbuffer() { return gbuffer_id; };
	uint32_t get_depth_image() { return depth_id; };
	uint32_t get_depth_pyramid_image() { return depth_pyramid_id; };
	uint32_t get_shadowmap() { return shadowmap_id; };

private:
	uint32_t draw_id{};
	uint32_t gbuffer_id{};
	uint32_t depth_id{};
	uint32_t depth_pyramid_id{};
	uint32_t shadowmap_id{};
};

struct SamplerCache
{
	std::vector<VkDescriptorImageInfo> image_infos{};

	// TODO: perform cache checking
	void add_sampler(const VkSampler& sampler);
};

struct ImageCache
{
	std::vector<VkDescriptorImageInfo> image_infos{};

	uint32_t add_texture(const VkImageView& view);

	void set_depth_pyramid_image(uint32_t id) { depth_pyramid_id = id; };
	uint32_t get_depth_pyramid_image() { return depth_pyramid_id; };

private:
	uint32_t depth_pyramid_id{};
};

struct ShaderCache
{
	//std::unordered_map<std::string, VkShaderModule> data{};
	std::unordered_map<std::string, ShaderProgram> data{};

	ShaderProgram& operator[](std::string key);

	void add_shader(VkDevice device, const char* path, VkShaderStageFlagBits stage);
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
	bool freeze_camera{ false };
	glm::mat4 last_view{};
	glm::mat4 last_proj{};

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
	FrameData& get_last_frame() { return frames[(frame_number - 1) % FRAME_OVERLAP]; };
	DeletionQueue main_deletion_queue{};

	VmaAllocator allocator{};

	AllocatedImage draw_image{};
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
	AllocatedImage equirectangular_image{};
	AllocatedImage cubemap_image{};
	AllocatedImage irradiance_image{};
	AllocatedImage prefiltered_image{};
	AllocatedImage brdflut_image{};

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

	VkPhysicalDeviceProperties props{};

	static VulkanEngine& get();

	void init(std::vector<std::string> file_paths);
	void cleanup();
	void draw();
	void run();
	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& func);
	AllocatedBuffer create_buffer(size_t alloc_size, VmaAllocationCreateFlags flags, VkBufferUsageFlags usage);
	AllocatedBuffer reallocate_buffer(size_t alloc_size, AllocatedBuffer old_buffer, VmaAllocationCreateFlags flags, VkBufferUsageFlags usage);
	void destroy_buffer(const AllocatedBuffer& buffer);
	GPUMeshBuffers upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices);
	AllocatedBuffer upload_buffer(void* data, size_t data_size);

	// view has access to all mip and layers
	AllocatedImage create_image(VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags = 0, bool mipmapped = false); // does not currently handle priority
	AllocatedImage create_image(void* data, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags = 0, bool mipmapped = false);
	AllocatedImage create_cubemap(VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VmaAllocationCreateFlags flags = 0, bool mipmapped = false); 
	void destroy_image(const AllocatedImage& image);

	void update_scene();

	void register_object(Node* node, const glm::mat4& top_matrix);
	void execute_debug_pass(VkCommandBuffer cmd);
	void execute_deferred_shading(VkCommandBuffer cmd);
	void update_cascade();
	void draw_imgui(VkCommandBuffer cmd, VkImageView swapchain_view);
	void ready_mesh_draw();
	void ready_mesh_cull(RenderScene::MeshPass& pass, CullData& cull_data, glm::mat4& proj, bool orthographic = false);
	void ready_meshlet_cull(RenderScene::MeshPass& pass, ClusterCullData& cull_data, glm::mat4& proj, bool orthographic = false);
	void ready_shadow_cull(RenderScene::MeshPass& pass, CullData& cull_data, glm::mat4& proj, bool orthographic = false);
	void execute_compute_cull(VkCommandBuffer cmd, RenderScene::MeshPass& pass, CullData& cull_data, bool late, uint32_t post_pass);
	void execute_compute_cull(VkCommandBuffer cmd, RenderScene::MeshPass& pass, ClusterCullData& cull_data, VkBuffer count_buffer, uint32_t offset, bool late, uint32_t post_pass);
	void execute_shadow_cull(VkCommandBuffer cmd, CullData& cull_data);
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
	void init_renderables(std::vector<std::string>& file_paths);
	void init_bindless(); 
	void init_precomputations();
	void init_imgui();
	void build_cluster_grid();

	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();
};