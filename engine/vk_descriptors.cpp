#include <vk_descriptors.h>
#include <vk_types.h>

void DescriptorLayoutBuilder::add_binding(uint32_t binding, VkDescriptorType type, VkShaderStageFlags shader_stage)
{
	VkDescriptorSetLayoutBinding newbind{};
	newbind.binding = binding;
	newbind.descriptorCount = 1; // (!) what about bindless indexing
	newbind.descriptorType = type;
	newbind.stageFlags = shader_stage;
	
	bindings.push_back(newbind);
}

void DescriptorLayoutBuilder::clear()
{
	bindings.clear();
}

VkDescriptorSetLayout DescriptorLayoutBuilder::build(VkDevice device, void* pNext /*= nullptr*/, VkDescriptorSetLayoutCreateFlags flags/*= 0*/) 
{
	VkDescriptorSetLayoutCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	info.pNext = pNext;
	info.flags = flags;
	info.bindingCount = static_cast<uint32_t>(bindings.size());
	info.pBindings = bindings.data();

	VkDescriptorSetLayout layout{};
	vkCreateDescriptorSetLayout(device, &info, nullptr, &layout);

	return layout;
}

void DescriptorAllocatorGrowable::init(VkDevice device, uint32_t initial_sets, std::span<PoolSizeRatio> pool_ratios)
{
	ratios.clear();

	for (auto r : pool_ratios)
	{
		ratios.push_back(r);
	}

	VkDescriptorPool new_pool = create_pool(device, initial_sets, pool_ratios);

	sets_per_pool = initial_sets * 1.5; // grow it next allocation

	ready_pools.push_back(new_pool);
}

void DescriptorAllocatorGrowable::clear_pools(VkDevice device)
{
	for (auto p : ready_pools)
	{
		vkResetDescriptorPool(device, p, 0);
	}

	for (auto p : full_pools)
	{
		vkResetDescriptorPool(device, p, 0);
		ready_pools.push_back(p);
	}

	full_pools.clear();
}

void DescriptorAllocatorGrowable::destroy_pools(VkDevice device)
{
	for (auto p : ready_pools)
	{
		vkDestroyDescriptorPool(device, p, nullptr);
	}
	ready_pools.clear();

	for (auto p : full_pools)
	{
		vkDestroyDescriptorPool(device, p, nullptr);
	}
	full_pools.clear();
}

VkDescriptorSet DescriptorAllocatorGrowable::allocate(VkDevice device, VkDescriptorSetLayout layout, void* pNext /*= nullptr*/)
{
	VkDescriptorPool pool = get_pool(device);

	VkDescriptorSetAllocateInfo allocate_info{};
	allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocate_info.pNext = pNext;
	allocate_info.descriptorPool = pool;
	allocate_info.descriptorSetCount = 1;
	allocate_info.pSetLayouts = &layout;

	VkDescriptorSet ds{};

	VkResult result = vkAllocateDescriptorSets(device, &allocate_info, &ds);

	if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL)
	{
		full_pools.push_back(pool);

		pool = get_pool(device);
		allocate_info.descriptorPool = pool;
		
		VK_CHECK(vkAllocateDescriptorSets(device, &allocate_info, &ds));
	}

	ready_pools.push_back(pool);

	return ds;
}

VkDescriptorPool DescriptorAllocatorGrowable::get_pool(VkDevice device)
{
	VkDescriptorPool new_pool{};
	if (ready_pools.size() != 0)
	{
		new_pool = ready_pools.back();
		ready_pools.pop_back();
	}
	else
	{
		new_pool = create_pool(device, sets_per_pool, ratios);

		sets_per_pool *= 1.5;
		if (sets_per_pool > 4092)
			sets_per_pool = 4092;
	}

	return new_pool;
}

VkDescriptorPool DescriptorAllocatorGrowable::create_pool(VkDevice device, uint32_t max_sets, std::span<PoolSizeRatio> pool_ratios) 
{
	std::vector<VkDescriptorPoolSize> pool_sizes{};
	for (PoolSizeRatio& ratio : pool_ratios)
	{
		pool_sizes.push_back(VkDescriptorPoolSize{
			.type = ratio.type,
			.descriptorCount = static_cast<uint32_t>(ratio.ratio * max_sets) // pool total = count * sets
		});
	}

	VkDescriptorPoolCreateInfo pool_info{};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.flags = 0; // (!) bindless indexing? update after bind?
	pool_info.maxSets = max_sets;
	pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
	pool_info.pPoolSizes = pool_sizes.data();

	VkDescriptorPool new_pool{};
	vkCreateDescriptorPool(device, &pool_info, nullptr, &new_pool);

	return new_pool;
}

void DescriptorWriter::write_image(int binding, VkImageView image, VkSampler sampler, VkImageLayout layout, VkDescriptorType type) 
{
	VkDescriptorImageInfo& info = image_infos.emplace_back(VkDescriptorImageInfo{
		.sampler = sampler,
		.imageView = image,
		.imageLayout = layout
	});

	VkWriteDescriptorSet write{}; 
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstBinding = binding;
	write.dstSet = VK_NULL_HANDLE; // handle later
	write.descriptorCount = 1;
	write.descriptorType = type;
	write.pImageInfo = &info;

	writes.push_back(write);
}

void DescriptorWriter::write_buffer(int binding, VkBuffer buffer, size_t size, size_t offset, VkDescriptorType type)
{
	VkDescriptorBufferInfo& info = buffer_infos.emplace_back(VkDescriptorBufferInfo{
		.buffer = buffer,
		.offset = offset,
		.range = size
	});

	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstBinding = binding;
	write.dstSet = VK_NULL_HANDLE; // handle later
	write.descriptorCount = 1;
	write.descriptorType = type;
	write.pBufferInfo = &info;

	writes.push_back(write);
}

void DescriptorWriter::clear()
{
	image_infos.clear();
	buffer_infos.clear();
	writes.clear();
}

void DescriptorWriter::update_set(VkDevice device, VkDescriptorSet set)
{
	for (VkWriteDescriptorSet& write : writes)
	{
		write.dstSet = set;
	}

	vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}
