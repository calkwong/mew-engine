#pragma once

#include "common.h"

#include <cstdint>
#include <vector>

class SamplerHeapManager
{
    void init_samplers(VkDevice device, void* p_heap);

    uint32_t sampler_descriptor_size{};

private:
    void get_sampler_descriptor(VkDevice device, VkFilter filter, VkSamplerMipmapMode mipmap, VkSamplerAddressMode address, void* descriptor, VkSamplerReductionModeCreateInfo* reduce = 0);
    void build_desc_set_bindings(std::vector<VkDescriptorSetAndBindingMappingEXT>& mappings);
};

class ResourceHeapManager
{
public:
    uint32_t add_uav(VkImage image, VkFormat format, VkImageViewType type, VkImageAspectFlags aspect, uint32_t mip);
    uint32_t add_srv(VkImage image, VkFormat format, VkImageViewType type, VkImageAspectFlags aspect, uint32_t mip);
    uint32_t add_buffer(VkBuffer buffer, VkDeviceSize size, VkDescriptorType type);
    uint32_t add_acceleration_structure(VkAccelerationStructureKHR as, VkDeviceSize size);
    void write_resource_heap(VkDevice device, void* p_heap);
    void set_srv_rebuild_size();

    void update_buffer(uint32_t handle, VkBuffer buffer, VkDeviceSize size);
    void update_uav(uint32_t handle, VkImage image, uint32_t mip);
    void update_srv(uint32_t handle, VkImage image, uint32_t mip);
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

    uint32_t srv_rebuild_size{};

    std::vector<BufferInfo> buffer_infos{};
    std::vector<ASInfo> as_infos{};
    std::vector<ImageInfo> uav_infos{};
    std::vector<ImageInfo> srv_infos{};
};
