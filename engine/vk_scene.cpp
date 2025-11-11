#include <vk_scene.h>
#include <vk_engine.h>

#include <vulkan/vulkan.h>

#include <vector>
#include <algorithm>

// PREREQ: pass objects
void RenderScene::build_indirect_batch()
{
	batches.clear();

	VkBuffer last_mesh{};
	ShaderPass* last_material{};

	for (size_t i = 0; i < pass_objects.size(); i++)
	{
		auto po = pass_objects[i];

		bool same_mesh = renderables[po.handle].mesh->index_buffer == last_mesh;
		bool same_material = po.material == last_material;

		if (same_mesh && same_material)
			batches.back().count++;
		else
		{
			last_mesh = renderables[po.handle].mesh->index_buffer; // (!) not great for cache?
			last_material = po.material;

			IndirectBatch new_batch{};
			new_batch.mesh = last_mesh;
			new_batch.forward_pass = last_material;
			new_batch.first = i;
			new_batch.count = 1;
			batches.push_back(new_batch);
		}
	}
}

// PREREQ: pass objects
void RenderScene::build_indirect_buffer()
{
	clear_indirect_buffer.clear();

	for (size_t i = 0; i < pass_objects.size(); i++)
	{
		VkDrawIndexedIndirectCommand draw_command{};

		const auto& obj = renderables[pass_objects[i].handle];
		draw_command.indexCount = obj.index_count;
		draw_command.instanceCount = 0;
		draw_command.firstIndex = obj.first_index;
		draw_command.vertexOffset = 0;
		draw_command.firstInstance = i;

		clear_indirect_buffer.push_back(draw_command);
	}
}

void RenderScene::reset_indirect_buffer(VkDrawIndexedIndirectCommand* draw_indirect_buffer)
{
	for (size_t i = 0; i < clear_indirect_buffer.size(); i++)
	{
		*(draw_indirect_buffer + i) = clear_indirect_buffer[i];
	}
}

void RenderScene::build_object_buffer()
{
	ObjectData* object_data = static_cast<ObjectData*>(object_buffer.info.pMappedData);

	for (size_t i = 0; i < renderables.size(); i++)
	{
		const auto& obj = renderables[i];

		object_data[i].transform = obj.transform;
		object_data[i].origin = obj.bounds.origin;
		object_data[i].material_id = obj.material_id;
		object_data[i].extent = obj.bounds.extents;
		object_data[i].vertex_buffer_address = obj.vertex_buffer_address;
	}
}

// PREREQ: unbatched objects
void RenderScene::build_pass_objects()
{
	for (auto o : unbatched_objects)
	{
		const auto& obj = renderables[o];

		PassObject po{};
		po.handle = o;
		po.material = obj.material->forward_pass; // (!) hardcoded

		pass_objects.push_back(po);
	}
}

void RenderScene::sort_objects()
{
	std::sort(pass_objects.begin(), pass_objects.end(), [&](const PassObject& a, const PassObject& b) {
		if (a.material != b.material)
			return a.material < b.material;
		else
			return renderables[a.handle].index_buffer < renderables[b.handle].index_buffer;
		});
}

// PREREQ: object buffer, indirect batch & pass objects
void RenderScene::build_ginstance_buffer()
{
	GPUInstance* ginstance_data = static_cast<GPUInstance*>(ginstance_buffer.info.pMappedData);

	size_t index{};
	for (size_t i = 0; i < batches.size(); i++)
	{
		const auto& batch = batches[i];
		
		for (uint32_t b = 0; b < batch.count; b++)
		{
			ginstance_data[index].object_id = pass_objects[index].handle;
			ginstance_data[index].batch_id = static_cast<uint32_t>(i);
			index++;
		}
	}
}

// PREREQ: object buffer & pass objects
void RenderScene::build_instance_buffer()
{
	uint32_t* instance_data = static_cast<uint32_t*>(instance_buffer.info.pMappedData);

	// pass objects must be sorted here
	for (size_t i = 0; i < pass_objects.size(); i++)
	{
		 instance_data[i]  = pass_objects[i].handle;
	}
}