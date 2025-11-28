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

	//Handle<DrawPrimitive> last_primitive{};
	ShaderPass* last_material{};

	for (size_t i = 0; i < pass.pass_objects.size(); i++)
	{
		PassObject& obj = pass.pass_objects[i];

		//bool same_primitive = obj.primitive_id.handle == last_primitive.handle; 
		bool same_material = obj.material == last_material;

		//if (same_primitive && same_material)
		if (same_material)
			pass.batches.back().count++;
		else
		{
			//last_primitive = obj.primitive_id; 
			last_material = obj.material;

			IndirectBatch new_batch{};
			//new_batch.primitive_id = last_primitive;
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
	//for (size_t i = 0; i < pass.batches.size(); i++)
	for (size_t i = 0; i < pass.pass_objects.size(); i++)
	{
		ShaderPass* new_material = pass.pass_objects[i].material;

		if (last_material == new_material)
			pass.multibatches.back().max_draw_count++;
		else
		{
			MultiBatch multibatch{};
			multibatch.pipeline = new_material;
			multibatch.offset = static_cast<uint32_t>(i);
			multibatch.max_draw_count = 1;
			pass.multibatches.push_back(multibatch);
			last_material = new_material;
		}
	}
}

// PREREQ: pass objects
void RenderScene::build_indirect_buffer(MeshPass& pass)
{
	GPUIndirect* indirect_objects = static_cast<GPUIndirect*>(pass.clear_indirect_buffer.info.pMappedData);

	VkPipeline last_pipeline = VK_NULL_HANDLE;
	uint32_t pipeline_count = -1;
	//for (size_t i = 0; i < pass.batches.size(); i++)
	for (size_t i = 0; i < pass.pass_objects.size(); i++)
	{
		GPUIndirect indirect_data{};

		//uint32_t pass_object_id = pass.batches[i].first;
		//const DrawPrimitive& primitive = primitives[pass.pass_objects[pass_object_id].primitive_id.handle];
		const DrawPrimitive& primitive = primitives[pass.pass_objects[i].primitive_id.handle];
		indirect_data.command.indexCount = primitive.count;
		indirect_data.command.instanceCount = 0;
		indirect_data.command.firstIndex = primitive.start_index;
		indirect_data.command.vertexOffset = 0;

		// if using draw indirect count with instancing, just an ID for indexing, not associated with pass object
		//indirect_data.command.firstInstance = i; 
		// 
		// if not supporting instancing, firstInstance = actual passObjectID
		indirect_data.command.firstInstance = pass.pass_objects[i].renderable_id.handle;

		//VkPipeline current_pipeline = pass.pass_objects[pass_object_id].material->pipeline;
		VkPipeline current_pipeline = pass.pass_objects[i].material->pipeline;

		if (current_pipeline != last_pipeline)
		{
			last_pipeline = current_pipeline;
			pipeline_count++;
		}
		
		indirect_objects[i] = indirect_data;
	}
}

void RenderScene::build_object_buffer()
{
	ObjectData* object_data = static_cast<ObjectData*>(object_buffer.info.pMappedData);

	for (size_t i = 0; i < renderables.size(); i++)
	{
		const RenderObject& obj = renderables[i];

		object_data[i].transform = obj.transform;
		object_data[i].origin = obj.bounds.origin;
		object_data[i].radius = obj.bounds.radius;
		object_data[i].material_id = obj.material_id;
		//object_data[i].extent = obj.bounds.extents;
	}
}

// PREREQ: unbatched objects
void RenderScene::build_pass_objects(MeshPass& pass)
{
	pass.pass_objects.clear();

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
			return a.primitive_id.handle < b.primitive_id.handle; // only truly necessary if doing instanced draw indirect count
		});
}