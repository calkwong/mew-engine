#include <vk_scene.h>
#include <vk_types.h>
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

	ShaderPass* last_material{};

	for (size_t i = 0; i < pass.pass_objects.size(); i++)
	{
		PassObject& obj = pass.pass_objects[i];

		bool same_material = obj.material == last_material;

		if (same_material)
			pass.batches.back().count++;
		else
		{
			last_material = obj.material;

			IndirectBatch new_batch{};
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

void RenderScene::build_object_buffer()
{
	ObjectData* object_data = static_cast<ObjectData*>(object_buffer.info.pMappedData);

	uint32_t offset = 0;

	for (size_t i = 0; i < renderables.size(); i++)
	{
		const RenderObject& obj = renderables[i];

		object_data[i].transform = obj.transform;
		object_data[i].material_id = obj.material_id;
		object_data[i].meshlet_bit_offset = offset; // TODO: refactor in future, should be per-pass

		offset += obj.meshlet_bits;
		max_meshtask_commands += (obj.meshlet_bits + 31) / 32; // TODO: using clustercull workgroup size, remove magic number
	}

	total_meshlets_bits = offset;
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

// PREREQ: sorted Pass Objects
void RenderScene::build_mesh_buffer()
{
	DrawPrimitive* mesh_data = static_cast<DrawPrimitive*>(mesh_buffer.info.pMappedData);

	for (size_t i = 0; i < primitives.size(); i++)
	{
		auto& mesh = primitives[i];

		mesh_data[i].center = mesh.center;
		mesh_data[i].radius = mesh.radius;
		mesh_data[i].mesh_lods = mesh.mesh_lods;
		mesh_data[i].lod_count = mesh.lod_count;
		mesh_data[i].vertex_offset = mesh.vertex_offset;
	}
}

// PREREQ: sorted Pass Objects, Object Buffer, Mesh Buffer (or DrawPrimitive)
void RenderScene::build_instance_buffer(MeshPass& pass)
{
	GPUInstance* instance = static_cast<GPUInstance*>(pass.instance_buffer.info.pMappedData);

	for (size_t i = 0; i < pass.pass_objects.size(); i++)
	{
		auto& obj = pass.pass_objects[i];

		instance[i].mesh_id = obj.primitive_id.handle;
		instance[i].object_id = obj.renderable_id.handle;
	}
}