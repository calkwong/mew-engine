// References
// https://dl.acm.org/doi/epdf/10.1145/2556700.2556705
// https://github.com/Devsh-Graphics-Programming/Nabla/blob/master/include/nbl/builtin/glsl/ext/OIT/insert_node.glsl
// https://interplayoflight.wordpress.com/2022/07/02/order-independent-transparency-part-2/

import bindings;
import types;
import util;

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
	OITData* oit_buffer;
	// PrefixSum* prefix_sum_buffer;
	uint* padding;
	float4 jitter_offset;
};

[[vk::push_constant]] PushConstants pc;

void swap_node(inout uint color_a, inout uint depth_a, inout float transmission_a, inout uint color_b, inout uint depth_b, inout float transmission_b)
{
	uint colorT = color_a;
	uint depthT = depth_a;
	float transmissionT = transmission_a;

	color_a = color_b;
	depth_a = depth_b;
	transmission_a = transmission_b;

	color_b = colorT;
	depth_b = depthT;
	transmission_b = transmissionT;
}

void insert_node(uint2 coords, uint color, uint depth, float transmission)
{
	uint index = coords.x + coords.y * 1700; // TODO: hardcoded

	OITData frag = oit_buffer[index];
	uint4 colors = frag.colors;
	uint4 depths = frag.depths;
	float4 transmissions = frag.transmissions;

	bool4 not_valid_bool = transmissions == float4(1.0);

	float4 not_valid_mask = select(not_valid_bool, float4(1.0), float4(0.0));
	depths = asuint(lerp(asfloat(depths), float4(0.0), not_valid_mask)); // reverse z
	bool4 closer_mask = uint4(depth) >= depths; // potential precision issue?

	// swap nodes if closer
	if (closer_mask[0])
		swap_node(colors[0], depths[0], transmissions[0], color, depth, transmission);
	if (closer_mask[1])
		swap_node(colors[1], depths[1], transmissions[1], color, depth, transmission);
	if (closer_mask[2])
		swap_node(colors[2], depths[2], transmissions[2], color, depth, transmission);
	if (closer_mask[3])
		swap_node(colors[3], depths[3], transmissions[3], color, depth, transmission);

	// compress array if overflow
	if (!not_valid_bool[MLAB_NODES-1])
	{
		colors[MLAB_NODES-1] = packUnorm4x8(unpackUnorm4x8ToFloat(colors[MLAB_NODES-1]) + unpackUnorm4x8ToFloat(color) * transmissions[MLAB_NODES-1]);
		transmissions[MLAB_NODES-1] *= transmission;
	}

	OITData oit_data;
	oit_data.colors = colors;
	oit_data.depths = depths;
	oit_data.transmissions = transmissions;

	oit_buffer[index] = oit_data;
}

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

static const int MLAB_NODES = 4;

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

	insert_node(screen_coords, color, depth, transmission);
}
