#include "descriptors.h"
#include "common.h"
#include "resources.h"

#include <cstdint>
#include <vector>

// implicitly initializes hardcoded sampler infos
void SamplerHeapManager::write_sampler_heap(VkDevice device, void* p_heap)
{
    VkSamplerReductionModeCreateInfo reduction_info{};
    reduction_info.sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO;
    reduction_info.reductionMode = VK_SAMPLER_REDUCTION_MODE_MIN;

    // 0: linear
    // 1: cube map sampling
    // 2: shadow map sampler
    // 3: hiz
    // 4: nearest clamp to border
    // 5: nearest clamp to edge
    // 6: linear clamp to edge
    // 7: nearest repeat

    get_sampler_descriptor(device, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, static_cast<uint8_t*>(p_heap) + 0 * sampler_descriptor_size);
    get_sampler_descriptor(device, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, static_cast<uint8_t*>(p_heap) + 1 * sampler_descriptor_size);
    get_sampler_descriptor(device, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, static_cast<uint8_t*>(p_heap) + 2 * sampler_descriptor_size);
    get_sampler_descriptor(device, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, static_cast<uint8_t*>(p_heap) + 3 * sampler_descriptor_size, &reduction_info);
    get_sampler_descriptor(device, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, static_cast<uint8_t*>(p_heap) + 4 * sampler_descriptor_size);
    get_sampler_descriptor(device, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, static_cast<uint8_t*>(p_heap) + 5 * sampler_descriptor_size);
    get_sampler_descriptor(device, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, static_cast<uint8_t*>(p_heap) + 6 * sampler_descriptor_size);
    get_sampler_descriptor(device, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT, static_cast<uint8_t*>(p_heap) + 7 * sampler_descriptor_size);
}

void SamplerHeapManager::get_sampler_descriptor(
    VkDevice device,
    VkFilter filter,
    VkSamplerMipmapMode mipmap,
    VkSamplerAddressMode address,
    void* descriptor,
    VkSamplerReductionModeCreateInfo* reduce /* = 0 */
)
{
    VkSamplerCreateInfo sampler_info{};

    VkSamplerReductionModeCreateInfo reduction_info{};
    if (reduce != nullptr)
    {
        sampler_info.pNext = reduce;
    }

    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = filter;
    sampler_info.minFilter = filter;
    sampler_info.mipmapMode = mipmap;
    sampler_info.addressModeU = address;
    sampler_info.addressModeV = address;
    sampler_info.addressModeW = address;
    sampler_info.maxLod = VK_LOD_CLAMP_NONE;
    sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK; // hack, only relevant when CLAMP_TO_BORDER is used

    VkHostAddressRangeEXT host_address_range{ descriptor, sampler_descriptor_size };
    vkWriteSamplerDescriptorsEXT(device, 1, &sampler_info, &host_address_range);
}

void SamplerHeapManager::build_desc_set_bindings(std::vector<VkDescriptorSetAndBindingMappingEXT>& mappings)
{
    VkDescriptorSetAndBindingMappingEXT desc_set_and_binding_mapping{};
    desc_set_and_binding_mapping.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT;
    desc_set_and_binding_mapping.descriptorSet = 3;
    desc_set_and_binding_mapping.firstBinding = 0;
    desc_set_and_binding_mapping.bindingCount = 1;
    desc_set_and_binding_mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_ALL_EXT;
    desc_set_and_binding_mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;

    VkDescriptorMappingSourceConstantOffsetEXT constant_offset{};
    constant_offset.heapOffset = 0;
    constant_offset.heapArrayStride = sampler_descriptor_size;

    VkDescriptorMappingSourceDataEXT source_data{};
    source_data.constantOffset = constant_offset;

    desc_set_and_binding_mapping.sourceData = source_data;

    mappings.push_back(desc_set_and_binding_mapping);
}

uint32_t ResourceHeapManager::add_uav(AllocatedImage& image, VkImageViewType type, VkImageAspectFlags aspect, uint32_t mip /* = 0 */)
{
    auto new_id = uav_infos.size();
    uav_infos.emplace_back(ImageInfo{ image.image, image.format, type, aspect, mip });

    return new_id;
}

uint32_t ResourceHeapManager::add_srv(AllocatedImage& image, VkImageViewType type, VkImageAspectFlags aspect)
{
    auto new_id = srv_infos.size();
    srv_infos.emplace_back(ImageInfo{ image.image, image.format, type, aspect, 0 });

    return new_id;
}

uint32_t ResourceHeapManager::add_buffer(AllocatedBuffer& buffer, VkDescriptorType type)
{
    auto new_id = buffer_infos.size();
    buffer_infos.emplace_back(BufferInfo{ buffer.buffer, buffer.size, type });

    return new_id;
}

uint32_t ResourceHeapManager::add_acceleration_structure(VkAccelerationStructureKHR as, VkDeviceSize size)
{
    auto new_id = buffer_infos.size();
    as_infos.emplace_back(ASInfo{ as, size });

    return new_id;
}

void ResourceHeapManager::write_resource_heap(VkDevice device, void* p_heap, bool rebuild /* = false */)
{
    uint32_t offset = 0;

    for (uint32_t i = 0; i < buffer_infos.size(); i++)
    {
        void* descriptor = static_cast<uint8_t*>(p_heap) + i * buffer_descriptor_size + offset;
        auto& buf = buffer_infos[i];
        get_buffer_descriptor(device, descriptor, buf);
    }
    offset += buffer_infos.size() * buffer_descriptor_size;

    for (uint32_t i = 0; i < as_infos.size(); i++)
    {
        void* descriptor = static_cast<uint8_t*>(p_heap) + i * buffer_descriptor_size + offset;
        auto& as = as_infos[i];
        get_as_descriptor(device, descriptor, as);
    }
    offset += as_infos.size() * buffer_descriptor_size;

    for (uint32_t i = 0; i < uav_infos.size(); ++i)
    {
        auto& info = uav_infos[i];
        void* descriptor = static_cast<uint8_t*>(p_heap) + i * image_descriptor_size + offset;
        get_image_descriptor(device, descriptor, info, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    }
    offset += uav_infos.size() * image_descriptor_size;

    auto texture_count = rebuild ? texture_offset : srv_infos.size();
    for (uint32_t i = 0; i < texture_count; ++i)
    {
        auto& info = srv_infos[i];
        void* descriptor = static_cast<uint8_t*>(p_heap) + i * image_descriptor_size + offset;
        get_image_descriptor(device, descriptor, info, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
    }
}

void ResourceHeapManager::get_image_descriptor(
    VkDevice device,
    void* descriptor,
    ImageInfo& img_info,
    VkDescriptorType descriptor_type
)
{
    VkImage image = img_info.image;
    VkFormat format = img_info.format;
    VkImageViewType view_type = img_info.type;
    VkImageAspectFlags aspect_flags = img_info.aspect;
    uint32_t mip = img_info.mip;

    VkImageViewCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.image = image;
    info.viewType = view_type;
    info.format = format;

    VkImageSubresourceRange subresource_range{};
    subresource_range.aspectMask = aspect_flags;
    subresource_range.baseMipLevel = mip;
    subresource_range.levelCount = VK_REMAINING_MIP_LEVELS;
    subresource_range.baseArrayLayer = 0;
    subresource_range.layerCount = VK_REMAINING_ARRAY_LAYERS;
    info.subresourceRange = subresource_range;

    VkImageDescriptorInfoEXT img_descriptor_info{};
    img_descriptor_info.sType = VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT;
    img_descriptor_info.pView = &info;
    img_descriptor_info.layout = VK_IMAGE_LAYOUT_GENERAL;

    VkResourceDescriptorDataEXT descriptor_data{};
    descriptor_data.pImage = &img_descriptor_info;

    VkResourceDescriptorInfoEXT descriptor_info{};
    descriptor_info.sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT;
    descriptor_info.type = descriptor_type;
    descriptor_info.data = descriptor_data;

    VkHostAddressRangeEXT host_address_range{ descriptor, image_descriptor_size };
    VK_CHECK(vkWriteResourceDescriptorsEXT(device, 1, &descriptor_info, &host_address_range));
};

void ResourceHeapManager::get_as_descriptor(
    VkDevice device,
    void* descriptor,
    ASInfo& as_info
)
{
    VkAccelerationStructureKHR as = as_info.as;
    VkDeviceSize size = as_info.size;

    VkAccelerationStructureDeviceAddressInfoKHR address_info{};
    address_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    address_info.accelerationStructure = as;
    VkDeviceAddress addr = vkGetAccelerationStructureDeviceAddressKHR(device, &address_info);

    VkDeviceAddressRangeEXT addr_range{ .address = addr, .size = size };

    VkResourceDescriptorDataEXT descriptor_data{};
    descriptor_data.pAddressRange = &addr_range;

    VkResourceDescriptorInfoEXT descriptor_info{};
    descriptor_info.sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT;
    descriptor_info.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    descriptor_info.data = descriptor_data;

    VkHostAddressRangeEXT host_address_range{ descriptor, buffer_descriptor_size };
    VK_CHECK(vkWriteResourceDescriptorsEXT(device, 1, &descriptor_info, &host_address_range));
};

void ResourceHeapManager::get_buffer_descriptor(VkDevice device, void* descriptor, BufferInfo& buf_info)
{
    VkBuffer buffer = buf_info.buffer;
    VkDeviceSize size = buf_info.size;
    VkDescriptorType type = buf_info.type;

    VkDeviceAddress addr = get_buffer_address(device, buffer);

    VkDeviceAddressRangeEXT addr_range{ .address = addr, .size = size };

    VkResourceDescriptorDataEXT descriptor_data{};
    descriptor_data.pAddressRange = &addr_range;

    VkResourceDescriptorInfoEXT descriptor_info{};
    descriptor_info.sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT;
    descriptor_info.type = type;
    descriptor_info.data = descriptor_data;

    VkHostAddressRangeEXT host_address_range{ descriptor, buffer_descriptor_size };
    VK_CHECK(vkWriteResourceDescriptorsEXT(device, 1, &descriptor_info, &host_address_range));
};

void ResourceHeapManager::update_uav(uint32_t handle, AllocatedImage& image, uint32_t mip /* = 0 */)
{
    auto& info = uav_infos[handle];
    info.image = image.image;
    info.mip = mip;
}

void ResourceHeapManager::update_srv(uint32_t handle, AllocatedImage& image)
{
    auto& info = srv_infos[handle];
    info.image = image.image;
}

void ResourceHeapManager::update_buffer(uint32_t handle, AllocatedBuffer& buffer, VkDeviceSize size)
{
    auto& info = buffer_infos[handle];
    info.buffer = buffer.buffer;
    info.size = size;
}

uint32_t ResourceHeapManager::set_texture_offset()
{
    texture_offset = srv_infos.size();
    return texture_offset;
}

void ResourceHeapManager::build_desc_set_bindings(std::vector<VkDescriptorSetAndBindingMappingEXT>& mappings)
{
    // TODO: use finer grain flags instead of VK_SPIRV_RESOURCE_TYPE_ALL_EXT
    for (uint32_t i = 0; i < buffer_infos.size(); ++i)
    {
        VkDescriptorSetAndBindingMappingEXT desc_set_and_binding_mapping{};
        desc_set_and_binding_mapping.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT;
        desc_set_and_binding_mapping.descriptorSet = 0;
        desc_set_and_binding_mapping.firstBinding = i;
        desc_set_and_binding_mapping.bindingCount = 1;
        desc_set_and_binding_mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_ALL_EXT;
        desc_set_and_binding_mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;

        VkDescriptorMappingSourceConstantOffsetEXT constant_offset{};
        constant_offset.heapOffset = i * buffer_descriptor_size;
        constant_offset.heapArrayStride = buffer_descriptor_size;

        VkDescriptorMappingSourceDataEXT source_data{};
        source_data.constantOffset = constant_offset;
        desc_set_and_binding_mapping.sourceData = source_data;
        mappings.push_back(desc_set_and_binding_mapping);
    }

    auto offset = buffer_infos.size();
    auto heap_offset = buffer_descriptor_size * buffer_infos.size();
    for (uint32_t i = 0; i < as_infos.size(); ++i)
    {
        VkDescriptorSetAndBindingMappingEXT desc_set_and_binding_mapping{};
        desc_set_and_binding_mapping.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT;
        desc_set_and_binding_mapping.descriptorSet = 0;
        desc_set_and_binding_mapping.firstBinding = i + offset;
        desc_set_and_binding_mapping.bindingCount = 1;
        desc_set_and_binding_mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_ALL_EXT;
        desc_set_and_binding_mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;

        VkDescriptorMappingSourceConstantOffsetEXT constant_offset{};
        constant_offset.heapOffset = i * buffer_descriptor_size + heap_offset;
        constant_offset.heapArrayStride = buffer_descriptor_size;

        VkDescriptorMappingSourceDataEXT source_data{};
        source_data.constantOffset = constant_offset;
        desc_set_and_binding_mapping.sourceData = source_data;
        mappings.push_back(desc_set_and_binding_mapping);
    }

    // UAV
    heap_offset += buffer_descriptor_size * as_infos.size();
    {
        VkDescriptorSetAndBindingMappingEXT desc_set_and_binding_mapping{};
        desc_set_and_binding_mapping.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT;
        desc_set_and_binding_mapping.descriptorSet = 1;
        desc_set_and_binding_mapping.firstBinding = 0;
        desc_set_and_binding_mapping.bindingCount = 1;
        desc_set_and_binding_mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_ALL_EXT;
        desc_set_and_binding_mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;

        VkDescriptorMappingSourceConstantOffsetEXT constant_offset{};
        constant_offset.heapOffset = heap_offset;
        constant_offset.heapArrayStride = image_descriptor_size;

        VkDescriptorMappingSourceDataEXT source_data{};
        source_data.constantOffset = constant_offset;

        desc_set_and_binding_mapping.sourceData = source_data;

        mappings.push_back(desc_set_and_binding_mapping);
    }

    // SRV
    heap_offset += image_descriptor_size * uav_infos.size();
    {
        VkDescriptorSetAndBindingMappingEXT desc_set_and_binding_mapping{};
        desc_set_and_binding_mapping.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT;
        desc_set_and_binding_mapping.descriptorSet = 2;
        desc_set_and_binding_mapping.firstBinding = 0;
        desc_set_and_binding_mapping.bindingCount = 1;
        desc_set_and_binding_mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_ALL_EXT;
        desc_set_and_binding_mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;

        VkDescriptorMappingSourceConstantOffsetEXT constant_offset{};
        constant_offset.heapOffset = heap_offset;
        constant_offset.heapArrayStride = image_descriptor_size;

        VkDescriptorMappingSourceDataEXT source_data{};
        source_data.constantOffset = constant_offset;

        desc_set_and_binding_mapping.sourceData = source_data;

        mappings.push_back(desc_set_and_binding_mapping);
    }
}

void write_buffer_descriptor(VkDevice device, void* descriptor, VkDeviceAddress buf_addr, VkDeviceSize buf_size, VkDescriptorType type, uint32_t buffer_descriptor_size)
{
    VkDeviceAddressRangeEXT addr_range{ .address = buf_addr, .size = buf_size };

    VkResourceDescriptorDataEXT descriptor_data{};
    descriptor_data.pAddressRange = &addr_range;

    VkResourceDescriptorInfoEXT descriptor_info{};
    descriptor_info.sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT;
    descriptor_info.type = type;
    descriptor_info.data = descriptor_data;

    VkHostAddressRangeEXT host_address_range{ descriptor, buffer_descriptor_size };
    VK_CHECK(vkWriteResourceDescriptorsEXT(device, 1, &descriptor_info, &host_address_range));
};
