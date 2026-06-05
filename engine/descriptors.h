#pragma once

#include "common.h"
#include "resources.h"

#include <cstdint>
#include <vector>

class SamplerHeapManager
{
public:
    void write_sampler_heap(VkDevice device, void* p_heap);
    void build_desc_set_bindings(std::vector<VkDescriptorSetAndBindingMappingEXT>& mappings);

    uint32_t sampler_descriptor_size{};

private:
    void get_sampler_descriptor(VkDevice device, VkFilter filter, VkSamplerMipmapMode mipmap, VkSamplerAddressMode address, void* descriptor, VkSamplerReductionModeCreateInfo* reduce = 0);
};

class ResourceHeapManager
{
public:
    uint32_t add_uav(AllocatedImage& image, VkImageViewType type, VkImageAspectFlags aspect, uint32_t mip = 0);
    uint32_t add_srv(AllocatedImage& image, VkImageViewType type, VkImageAspectFlags aspect);
    uint32_t add_buffer(AllocatedBuffer& buffer, VkDescriptorType type);
    uint32_t add_acceleration_structure(VkAccelerationStructureKHR as, VkDeviceSize size);

    void update_buffer(uint32_t handle, AllocatedBuffer& buffer, VkDeviceSize size);
    void update_uav(uint32_t handle, AllocatedImage& image, uint32_t mip = 0);
    void update_srv(uint32_t handle, AllocatedImage& image);

    uint32_t set_texture_offset();
    void write_resource_heap(VkDevice device, void* p_heap, bool rebuild = false);
    void build_desc_set_bindings(std::vector<VkDescriptorSetAndBindingMappingEXT>& mappings);

    uint32_t buffer_descriptor_size{};
    uint32_t image_descriptor_size{};

private:
    struct BufferInfo
    {
        VkBuffer buffer{};
        VkDeviceSize size{};
        VkDescriptorType type{};
    };

    struct ImageInfo
    {
        VkImage image{};
        VkFormat format{};
        VkImageViewType type{};
        VkImageAspectFlags aspect{};
        uint32_t mip = 0;
    };

    struct ASInfo
    {
        VkAccelerationStructureKHR as{};
        VkDeviceSize size{};
    };

    void get_image_descriptor(VkDevice device, void* descriptor, ImageInfo& img_info, VkDescriptorType descriptor_type);
    void get_as_descriptor(VkDevice device, void* descriptor, ASInfo& as_info);
    void get_buffer_descriptor(VkDevice device, void* descriptor, BufferInfo& buf_info);

    uint32_t texture_offset{};

    std::vector<BufferInfo> buffer_infos{};
    std::vector<ASInfo> as_infos{};
    std::vector<ImageInfo> uav_infos{};
    std::vector<ImageInfo> srv_infos{};
};

void write_buffer_descriptor(VkDevice device, void* descriptor, VkDeviceAddress buf_addr, VkDeviceSize buf_size, VkDescriptorType type, uint32_t buffer_descriptor_size);
