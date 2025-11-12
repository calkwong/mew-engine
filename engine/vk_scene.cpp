#include <vk_scene.h>
#include <vk_engine.h>

#include <vulkan/vulkan.h>

#include <vector>
#include <algorithm>

// PREREQ: pass objects
void RenderScene::build_indirect_batch()
{
	batches.clear();

	uint32_t last_primitive{};
	ShaderPass* last_material{};

	for (size_t i = 0; i < pass_objects.size(); i++)
	{
		auto po = pass_objects[i];

		bool same_primitive = po.primitive_id == last_primitive; // (!) should be mesh to fix actual instancing
		bool same_material = po.material == last_material;

		if (same_primitive && same_material)
			batches.back().count++;
		else
		{
			last_primitive = po.primitive_id; // (!) not great for cache?
			last_material = po.material;

			IndirectBatch new_batch{};
			new_batch.primitive = last_primitive;
			new_batch.forward_pass = last_material;
			new_batch.first = i;
			new_batch.count = 1;
			batches.push_back(new_batch);
		}
	}
}

// PREREQ: indirect batch
void RenderScene::build_multi_batch()
{
	multibatches.clear();

	ShaderPass* last_material{};
	for (size_t i = 0; i < batches.size(); i++)
	{
		ShaderPass* new_material = batches[i].forward_pass;

		if (last_material == new_material)
		{
			multibatches.back().count++;
		}
		else
		{
			last_material = new_material;

			MultiBatch multibatch{};
			multibatch.first = static_cast<uint32_t>(i);
			multibatch.count = 1;
			multibatches.push_back(multibatch);
		}

	}
}

// PREREQ: pass objects
void RenderScene::build_indirect_buffer()
{
	clear_indirect_buffer.clear();

	for (size_t i = 0; i < batches.size(); i++)
	{
		VkDrawIndexedIndirectCommand draw_command{};

		//const auto& obj = renderables[pass_objects[i].handle];
		auto pass_object_id = batches[i].first;

		const auto& primitive = primitives[pass_objects[pass_object_id].primitive_id];
		draw_command.indexCount = primitive.count;
		draw_command.instanceCount = 0;
		draw_command.firstIndex = primitive.start_index;
		draw_command.vertexOffset = 0;
		draw_command.firstInstance = pass_object_id; // (!!) double check 

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
		//object_data[i].vertex_buffer_address = obj.vertex_buffer_address;
	}
}

// PREREQ: unbatched objects
void RenderScene::build_pass_objects()
{
	for (auto o : unbatched_objects)
	{
		const auto& obj = renderables[o];

		PassObject po{};
		po.primitive_id = obj.primitive_id;
		po.renderables_id = o;
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
			//return renderables[a.handle].index_buffer < renderables[b.handle].index_buffer;
			return a.primitive_id < b.primitive_id;
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
			ginstance_data[index].object_id = pass_objects[index].renderables_id;
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
		 instance_data[i]  = pass_objects[i].renderables_id; // for indexing into ObjectBuffer
	}
}

uint32_t RenderScene::add_primitive(uint32_t start_index, uint32_t count)
{
	uint32_t size = static_cast<uint32_t>(primitives.size());

	for (size_t i = 0; i < size; i++)
	{
		if (primitives[i].start_index == start_index && primitives[i].count == count)
			return static_cast<uint32_t>(i);
	}

	primitives.push_back(DrawPrimitive{ start_index, count });
	return size;
}