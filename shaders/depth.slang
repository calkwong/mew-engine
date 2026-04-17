import bindings;
import types;
import util;

struct Interp
{
	float4 position : SV_Position;
	float2 uv;
	nointerpolation uint material_id;
};

struct PushConstants
{
	float4x4 view_proj;
	MaterialData* material_buffer;
	ObjectData* object_buffer;
	Vertex* vertex_buffer;
};

[[vk::push_constant]] PushConstants pc;

[shader("vertex")]
Interp vs_main(
    uint instance_id : SV_VulkanInstanceID,
    uint vertex_id : SV_VulkanVertexID,
)
{
    Interp interp;

	ObjectData o = pc.object_buffer[instance_id]; // instance_id from drawIndirectCommand
	Vertex v = pc.vertex_buffer[vertex_id];

	float3 pos = float3(v.px, v.py, v.pz);
	float4 world_pos = float4(rotate_quat(pos, o.orientation) * o.scale + o.translation, 1.0);

	interp.position = mul(pc.view_proj, world_pos);
	interp.uv = float2(v.uv_x, v.uv_y);
	interp.material_id = o.material_id;

	return interp;
}

[vk::constant_id(0)]
const int OPAQUE = 1;

[shader("pixel")]
void ps_main(
    Interp in
)
{
   	if (OPAQUE == 0)
	{
		MaterialData m = pc.material_buffer[in.material_id];

		float4 albedo = m.base_color_factor;
		if (m.diffuse_id != 0)
		    albedo *= textures[m.diffuse_id].Sample(samplers[LINEAR_SAMPLER], in.uv);

		if (albedo.a < 0.5)
			discard;
	}
}
