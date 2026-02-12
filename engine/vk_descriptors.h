#pragma once

#include <volk.h>
#include <deque>
#include <span>
#include <vector>

struct DescriptorLayoutBuilder
{
	std::vector<VkDescriptorSetLayoutBinding> bindings{};

	void add_binding(uint32_t binding, VkDescriptorType type, VkShaderStageFlags shader_stage);
	void clear();

	VkDescriptorSetLayout build(VkDevice device, const void* pNext = nullptr, VkDescriptorSetLayoutCreateFlags flags = 0) const;
};

// TODO: refactor for bindless?
class DescriptorAllocatorGrowable
{
public:
	struct PoolSizeRatio
	{
		VkDescriptorType type{};
		float ratio{};
	};

	void init(VkDevice device, uint32_t initial_sets, std::span<PoolSizeRatio> pool_ratios);
	void clear_pools(VkDevice device);
	void destroy_pools(VkDevice device);

	VkDescriptorSet allocate(VkDevice device, VkDescriptorSetLayout layout, const void* pNext = nullptr);

private:
	VkDescriptorPool get_pool(VkDevice device);
	VkDescriptorPool create_pool(VkDevice device, uint32_t max_sets, std::span<PoolSizeRatio> pool_ratios);

	std::vector<PoolSizeRatio> ratios{};
	std::vector<VkDescriptorPool> full_pools{};
	std::vector<VkDescriptorPool> ready_pools{};
	uint32_t sets_per_pool{};
};

// TODO: refactor for bindless?
struct DescriptorWriter
{
	std::deque<VkDescriptorImageInfo> image_infos{};
	std::deque<VkDescriptorBufferInfo> buffer_infos{};
	std::vector<VkWriteDescriptorSet> writes{};

	void write_image(int binding, VkImageView image, VkSampler sampler, VkImageLayout layout, VkDescriptorType type);
	void write_buffer(int binding, VkBuffer buffer, size_t size, size_t offset, VkDescriptorType type);

	void clear();
	void update_set(VkDevice device, VkDescriptorSet set);
};