#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

// destruction of textures handled by gltf (not internally); does not support dynamic objs
struct TextureCache
{
	std::vector<VkDescriptorImageInfo> image_infos{};

	uint32_t add_texture(const VkImageView& view);

	void set_draw_image(uint32_t id);
	void set_gbuffers(uint32_t id); // deferred
	void set_visibility_buffer(uint32_t id); // visibility
	void set_depth_image(uint32_t id);
	void set_depth_pyramid_image(uint32_t id);
	void set_shadowmap(uint32_t id);
	void set_accumulation_buffer(uint32_t id);
	void set_hdri(uint32_t id);
	uint32_t get_draw_image() const;
	uint32_t get_first_gbuffer() const; // deferred
	uint32_t get_visibility_buffer() const; // visibility
	uint32_t get_depth_image() const;
	uint32_t get_depth_pyramid_image() const;
	uint32_t get_shadowmap() const;
	uint32_t get_accumulation_buffer(uint32_t flip) const;
	uint32_t get_hdri() const;

private:
	uint32_t draw_id{};
	uint32_t gbuffer_id{};
	uint32_t visibility_id{};
	uint32_t depth_id{};
	uint32_t depth_pyramid_id{};
	uint32_t shadowmap_id{};
	uint32_t accum_id{};
	uint32_t hdri_id{};

	// pingpong
	int accum_pp{-1};
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

	void set_depth_pyramid_image(uint32_t id);
	void set_hdri(uint32_t id);
	void set_draw_image(uint32_t id);
	uint32_t get_depth_pyramid_image() const;
	uint32_t get_hdri() const;
	uint32_t get_draw_image() const;

private:
	uint32_t depth_pyramid_id{};
	uint32_t hdri_id{};
	uint32_t draw_id{};
};

struct ShaderProgram;

struct ShaderCache
{
	std::unordered_map<std::string, std::unique_ptr<ShaderProgram>> data{};

	ShaderProgram* operator[](const std::string& key);

	void add_shader(VkDevice device, const char* path, VkShaderStageFlagBits stage);
};