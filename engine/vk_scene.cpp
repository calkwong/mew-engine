#include <vk_scene.h>
#include <vk_pipelines.h>

#include <vulkan/vulkan.h>

#include <vector>
#include <algorithm>

void RenderScene::init()
{
	for (size_t i = 0; i < shadow_pass.size(); i++)
	{
		shadow_pass[i].type = MeshPassType::Shadow;
	}
	forward_pass.type = MeshPassType::Forward;
	transparent_pass.type = MeshPassType::Transparent;
}

// PREREQ: pass objects
void RenderScene::build_indirect_batch(MeshPass& pass)
{
	pass.batches.clear();

	Handle<DrawPrimitive> last_primitive{};
	ShaderPass* last_material{};

	for (size_t i = 0; i < pass.pass_objects.size(); i++)
	{
		PassObject& obj = pass.pass_objects[i];

		bool same_primitive = obj.primitive_id.handle == last_primitive.handle; 
		bool same_material = obj.material == last_material;

		if (same_primitive && same_material)
			pass.batches.back().count++;
		else
		{
			last_primitive = obj.primitive_id; 
			last_material = obj.material;

			IndirectBatch new_batch{};
			new_batch.primitive_id = last_primitive;
			new_batch.material = last_material;
			new_batch.first = static_cast<uint32_t>(i);
			new_batch.count = 1;
			pass.batches.push_back(new_batch);
		}
	}
}

// PREREQ: indirect batch
void RenderScene::build_multi_batch(MeshPass& pass)
{
	pass.multibatches.clear();

	ShaderPass* last_material{};
	for (size_t i = 0; i < pass.batches.size(); i++)
	{
		ShaderPass* new_material = pass.batches[i].material;

		if (last_material == new_material)
		{
			pass.multibatches.back().count++;
		}
		else
		{
			last_material = new_material;

			MultiBatch multibatch{};
			multibatch.first = static_cast<uint32_t>(i);
			multibatch.count = 1;
			pass.multibatches.push_back(multibatch);
		}

	}
}

// PREREQ: pass objects
void RenderScene::build_indirect_buffer(MeshPass& pass)
{
	VkDrawIndexedIndirectCommand* draw_commands = static_cast<VkDrawIndexedIndirectCommand*>(pass.clear_indirect_buffer.info.pMappedData);

	for (size_t i = 0; i < pass.batches.size(); i++)
	{
		VkDrawIndexedIndirectCommand draw_command{};

		uint32_t pass_object_id = pass.batches[i].first;

		const DrawPrimitive& primitive = primitives[pass.pass_objects[pass_object_id].primitive_id.handle];
		draw_command.indexCount = primitive.count;
		draw_command.instanceCount = 0;
		draw_command.firstIndex = primitive.start_index;
		draw_command.vertexOffset = 0;
		draw_command.firstInstance = pass_object_id;

		draw_commands[i] = draw_command;
	}
}

void RenderScene::reset_indirect_buffer(MeshPass& pass, VkCommandBuffer cmd)
{
	VkBufferCopy copy{};
	copy.dstOffset = 0;
	copy.srcOffset = 0;
	copy.size = pass.clear_indirect_buffer.info.size;

	vkCmdCopyBuffer(cmd, pass.clear_indirect_buffer.buffer, pass.draw_indirect_buffer.buffer, 1, &copy);
}

void RenderScene::build_object_buffer()
{
	ObjectData* object_data = static_cast<ObjectData*>(object_buffer.info.pMappedData);

	for (size_t i = 0; i < renderables.size(); i++)
	{
		const RenderObject& obj = renderables[i];

		object_data[i].transform = obj.transform;
		object_data[i].origin = obj.bounds.origin;
		object_data[i].material_id = obj.material_id;
		object_data[i].extent = obj.bounds.extents;
	}
}

// PREREQ: unbatched objects
void RenderScene::build_pass_objects(MeshPass& pass)
{
	for (uint32_t o : pass.unbatched_objects)
	{
		const RenderObject& obj = renderables[o];

		PassObject pass_obj{};
		pass_obj.primitive_id = obj.primitive_id;
		pass_obj.renderable_id.handle = o;

		switch (pass.type)
		{
		case MeshPassType::Shadow:
			pass_obj.material = obj.material->shadow_pass;
			break;
		case MeshPassType::Forward:
		case MeshPassType::Transparent:
			pass_obj.material = obj.material->forward_pass;
			break;
		default: 
			break;
		}

		pass.pass_objects.push_back(pass_obj);
	}
}

void RenderScene::sort_objects(MeshPass& pass)
{
	std::sort(pass.pass_objects.begin(), pass.pass_objects.end(), [&](const PassObject& a, const PassObject& b) {
		if (a.material != b.material)
			return a.material < b.material;
		else
			return a.primitive_id.handle < b.primitive_id.handle;
		});
}

// PREREQ: object buffer, indirect batch & pass objects
void RenderScene::build_ginstance_buffer(MeshPass& pass)
{
	GPUInstance* ginstance_data = static_cast<GPUInstance*>(pass.ginstance_buffer.info.pMappedData);

	size_t index{};
	for (size_t i = 0; i < pass.batches.size(); i++)
	{
		const auto& batch = pass.batches[i];
		
		for (uint32_t b = 0; b < batch.count; b++)
		{
			ginstance_data[index].object_id = pass.pass_objects[index].renderable_id.handle;
			ginstance_data[index].batch_id = static_cast<uint32_t>(i);
			index++;
		}
	}
}