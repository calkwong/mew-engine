#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

struct TextureCache
{
    std::vector<uint8_t> textures{};

    uint32_t add_texture();

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
    uint32_t get_accumulation_buffer(uint32_t offset) const;
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
};

struct ImageCache
{
    std::vector<uint8_t> textures{};

    uint32_t add_texture();

    void set_depth_pyramid_image(uint32_t id);
    void set_hdri(uint32_t id);
    void set_draw_image(uint32_t id);
    void set_accumulation_buffer(uint32_t id);
    void set_irradiance(uint32_t id);
    void set_prefiltered(uint32_t id);
    void set_brdf(uint32_t id);
    uint32_t get_depth_pyramid_image() const;
    uint32_t get_hdri() const;
    uint32_t get_draw_image() const;
    uint32_t get_accumulation_buffer(uint32_t offset) const;
    uint32_t get_irradiance() const;
    uint32_t get_prefiltered() const;
    uint32_t get_brdf() const;

private:
    uint32_t depth_pyramid_id{};
    uint32_t hdri_id{}; // this is skybox we load/store to
    uint32_t draw_id{};
    uint32_t accum_id{};
    uint32_t irradiance_id{};
    uint32_t prefiltered_id{};
    uint32_t brdf_id{};
};

struct ShaderProgram;

struct ShaderCache
{
    std::unordered_map<std::string, std::unique_ptr<ShaderProgram>> data{};

    ShaderProgram* operator[](const std::string& key);

    void add_shader(VkDevice device, const char* path);
};
