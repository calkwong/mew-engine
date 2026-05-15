#include "common.h"
#include "cache.h"
#include "vk_pipelines.h"

#include <filesystem>
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

// note: assuming FIF == 2; pass in framenumber % FIF
uint32_t TextureCache::get_accumulation_buffer(uint32_t offset) const
{
    return accum_id + offset;
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

    image_infos.emplace_back(VkDescriptorImageInfo{ 0, view, VK_IMAGE_LAYOUT_GENERAL });

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

uint32_t ImageCache::get_accumulation_buffer(uint32_t offset) const
{
    return accum_id + offset;
}

void ImageCache::set_accumulation_buffer(uint32_t id)
{
    accum_id = id;
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

void ShaderCache::add_shader(VkDevice device, const char* path, size_t push_constant_size)
{
    const auto it = data.find(path);

    std::string shader_path{ "shaders/compiled/" };
    shader_path += path;
    shader_path += ".spv";

    if (it == data.end())
    {
        VkShaderModule module{};
        vkutil::load_shader_module(shader_path.c_str(), device, &module);
        auto time = std::filesystem::last_write_time(shader_path);
        data[path] = std::make_unique<ShaderProgram>(module, static_cast<uint32_t>(push_constant_size), path, time);
    }
}

void ImageCache::set_irradiance(uint32_t id)
{
    irradiance_id = id;
}

void ImageCache::set_prefiltered(uint32_t id)
{
    prefiltered_id = id;
}

void ImageCache::set_brdf(uint32_t id)
{
    brdf_id = id;
}

uint32_t ImageCache::get_irradiance() const
{
    return irradiance_id;
}

uint32_t ImageCache::get_prefiltered() const
{
    return prefiltered_id;
}

uint32_t ImageCache::get_brdf() const
{
    return brdf_id;
}
