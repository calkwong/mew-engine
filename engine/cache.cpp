#include "common.h"
#include "cache.h"
#include "vk_pipelines.h"

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

void TextureCache::set_draw_image(uint32_t id)
{
	draw_id = id;
}

void TextureCache::set_gbuffers(uint32_t id)
{
	gbuffer_id = id;
}

void TextureCache::set_visibility_buffer(uint32_t id)
{
	visibility_id = id;
}

void TextureCache::set_depth_image(uint32_t id)
{
	depth_id = id;
}

void TextureCache::set_depth_pyramid_image(uint32_t id)
{
	depth_pyramid_id = id;
}

void TextureCache::set_shadowmap(uint32_t id)
{
	shadowmap_id = id;
}
void TextureCache::set_accumulation_buffer(uint32_t id)
{
	accum_id = id;
}

void TextureCache::set_hdri(uint32_t id)
{
	hdri_id = id;
}

uint32_t TextureCache::get_hdri() const
{
	return hdri_id;
}

uint32_t TextureCache::get_accumulation_buffer(uint32_t flip) const
{
	// accum_pp = flip ? accum_pp * -1 : accum_pp;
	// accum_id = flip ? accum_id + accum_pp : accum_id;
	// accum_pp *= -1;
	// accum_id += accum_pp;
	return accum_id + flip;
}

uint32_t TextureCache::get_draw_image() const
{
	return draw_id;
}

uint32_t TextureCache::get_first_gbuffer() const
{
	return gbuffer_id;
}

uint32_t TextureCache::get_visibility_buffer() const
{
	return visibility_id;
}

uint32_t TextureCache::get_depth_image() const
{
	return depth_id;
}

uint32_t TextureCache::get_depth_pyramid_image() const
{
	return depth_pyramid_id;
}

uint32_t TextureCache::get_shadowmap() const
{
	return shadowmap_id;
}

uint32_t TextureCache::add_texture(const VkImageView& view)
{
	for (size_t i = 0; i < image_infos.size(); i++)
	{
		if (image_infos[i].imageView == view)
			return static_cast<uint32_t>(i);
	}

	image_infos.emplace_back(VkDescriptorImageInfo{ 0, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL });

	return static_cast<uint32_t>(image_infos.size() - 1);
}

void SamplerCache::add_sampler(const VkSampler& sampler)
{
	image_infos.emplace_back(VkDescriptorImageInfo{ .sampler = sampler });
}

void ImageCache::set_depth_pyramid_image(uint32_t id)
{
	depth_pyramid_id = id;
}

void ImageCache::set_hdri(uint32_t id)
{
	hdri_id = id;
}
void ImageCache::set_draw_image(uint32_t id)
{
	draw_id = id;
}

uint32_t ImageCache::get_draw_image() const
{
	return draw_id;
}

uint32_t ImageCache::get_hdri() const
{
	return hdri_id;
}

uint32_t ImageCache::get_depth_pyramid_image() const
{
	return depth_pyramid_id;
}

uint32_t ImageCache::add_texture(const VkImageView& view)
{
	for (size_t i = 0; i < image_infos.size(); i++)
	{
		if (image_infos[i].imageView == view)
			return static_cast<uint32_t>(i);
	}

	image_infos.emplace_back(VkDescriptorImageInfo{ 0, view, VK_IMAGE_LAYOUT_GENERAL });

	return static_cast<uint32_t>(image_infos.size() - 1);
}

ShaderProgram* ShaderCache::operator[](const std::string& key)
{
	return data[key].get();
}

void ShaderCache::add_shader(VkDevice device, const char* path, VkShaderStageFlagBits stage)
{
	const auto it = data.find(path);

	std::string shader_path{ "shaders/compiled/" };
	shader_path += path;
	shader_path += ".spv";

	if (it == data.end())
	{
		VkShaderModule module{};
		vkutil::load_shader_module(shader_path.c_str(), device, &module);

		data[path] = std::make_unique<ShaderProgram>(module, stage);
	}
}