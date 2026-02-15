#include "common.h"
#include "vk_scene.h"
#include "vk_math.h"
#include "resources.h"

#include <array>
#include <vector>

// POST MESH SHADERS: do we still need this?
void RenderScene::init()
{
	opaque_pass.type = MeshPassType::Opaque;
	mask_pass.type = MeshPassType::Mask;
	transparent_pass.type = MeshPassType::Transparent;
}

void RenderScene::build_object_buffer()
{
	ObjectData* object_data = static_cast<ObjectData*>(object_buffer.info.pMappedData);

	uint32_t offset = 0;

	for (size_t i = 0; i < renderables.size(); i++)
	{
		const RenderObject& obj = renderables[i];

		object_data[i].transform = obj.transform;
		object_data[i].mesh_id = obj.primitive_id.handle;
		object_data[i].material_id = obj.material_id;
		object_data[i].meshlet_bit_offset = offset; // TODO: used in meshletVisibilityBit, should probably rename this to make it less confusing
		object_data[i].post_pass = obj.post_pass;

		offset += obj.meshlet_bits; // TODO: used in meshletVisibilityBit, should probably rename this to make it less confusing
		max_meshtask_commands += (obj.meshlet_bits + 31) / 32; // TODO: using clustercull workgroup size, remove magic number
	}

	total_meshlets_bits = offset;
}

void RenderScene::build_mesh_buffer()
{
	DrawPrimitive* mesh_data = static_cast<DrawPrimitive*>(mesh_buffer.info.pMappedData);

	for (size_t i = 0; i < primitives.size(); i++)
	{
		const auto& mesh = primitives[i];

		mesh_data[i].center = mesh.center;
		mesh_data[i].radius = mesh.radius;
		mesh_data[i].mesh_lods = mesh.mesh_lods;
		mesh_data[i].lod_count = mesh.lod_count;
		mesh_data[i].vertex_offset = mesh.vertex_offset;
	}
}