// References
// https://dl.acm.org/doi/epdf/10.1145/2556700.2556705
// https://github.com/Devsh-Graphics-Programming/Nabla/blob/master/include/nbl/builtin/glsl/ext/OIT/insert_node.glsl
// https://interplayoflight.wordpress.com/2022/07/02/order-independent-transparency-part-2/

import bindings;
import types;
import util;
import oit;

[[vk::binding(0, 4)]] globallycoherent RasterizerOrderedStructuredBuffer<OITData> oit_buffer;

struct Interp
{
    float4 position : SV_Position;
    float3 normal;
    float2 uv;
    float4 tangent;
    nointerpolation uint material_id;
};

struct PushConstants
{
	ObjectData* object_buffer;
	Vertex* vertex_buffer;
	Meshlet* meshlet_buffer;
	uint* meshlet_indices_buffer;
	uint* cluster_indices_buffer;
	MaterialData* material_buffer;
	// PrefixSum* prefix_sum_buffer;
	uint* padding;
	uint2 screen_size;
	float4 jitter_offset;
};

[[vk::push_constant]] PushConstants pc;

[shader("vertex")]
Interp vs_main(
    uint instance_id : SV_VulkanInstanceID,
    uint vertex_id : SV_VulkanVertexID
)
{
    Interp interp;

	ObjectData o = pc.object_buffer[instance_id]; // instance_id from drawIndirectCommand
	Vertex v = pc.vertex_buffer[vertex_id];

	float3 pos = float3(v.px, v.py, v.pz);
	float4 world_pos = float4(rotate_quat(pos, o.orientation) * o.scale + o.translation, 1.0);

	float3 normal;
	float4 tangent;
	unpack_tbn(v.normal, uint(v.tangent), normal, tangent);

	interp.normal = rotate_quat(normal, o.orientation);
	interp.tangent = float4(rotate_quat(tangent.xyz, o.orientation), tangent.w);

	interp.uv = float2(v.uv_x, v.uv_y);
	interp.material_id = o.material_id;

    float4 clip_pos = mul(uniforms.view_proj, world_pos);

	interp.position = clip_pos;

	return interp;
}

[earlydepthstencil]
[shader("pixel")]
void ps_main(
    Interp in
)
{
	MaterialData m = pc.material_buffer[in.material_id];

	float4 albedo = m.base_color_factor;
	if (m.diffuse_id != 0)
		albedo *= textures[m.diffuse_id].Sample(samplers[LINEAR_SAMPLER], in.uv);

	albedo.a = 0.5; // TODO: remove, just for testing
	float4 premultiplied_alpha = float4(albedo.xyz * albedo.a, 1.0);
	uint color = packUnorm4x8(premultiplied_alpha);
	uint depth = asuint(in.position.z);
	float transmission = 1.0 - albedo.a;

	uint2 screen_coords = uint2(floor(in.position.xy));

	uint index = screen_coords.x + screen_coords.y * pc.screen_size.x;
	OITData fragment = oit_buffer[index];
	insert_node(fragment, color, depth, transmission);

	oit_buffer[index] = fragment;
}
