// References
// https://dl.acm.org/doi/epdf/10.1145/2556700.2556705
// https://github.com/Devsh-Graphics-Programming/Nabla/blob/master/include/nbl/builtin/glsl/ext/OIT/insert_node.glsl
// https://interplayoflight.wordpress.com/2022/07/02/order-independent-transparency-part-2/

// note: no culling for transparency

import bindings;
import types;
import util;
import prefix_sum;
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
	PrefixSum* prefix_sum_buffer;
	uint2 screen_size;
	float4 jitter_offset;
};

[[vk::push_constant]] PushConstants pc;

#define WG_SIZE_X 128

[shader("mesh")]
[numthreads(WG_SIZE_X, 1, 1)]
[outputtopology("triangle")]
void mesh_main(
    uint3 wg_id : SV_GroupID,
    uint t_index : SV_GroupIndex,
    out vertices Interp verts[64],
    out indices uint3 tris[124]
)
{
	uint id = wg_id.x;

	uint64_t counter = get_prefix_sum_counter(pc.prefix_sum_buffer);
	uint total_meshlet_count = uint(counter);
	uint target = pc.cluster_indices_buffer[id];

	uint prefix_sum_idx = prefix_sum_binary_search(pc.prefix_sum_buffer, target);
    PrefixSumData prefix_sum_data = pc.prefix_sum_buffer.data[prefix_sum_idx];
    uint meshlet_lod_offset = prefix_sum_data.lod_offset;

    // we do some packing hence this is essential!
    meshlet_lod_offset = bitfieldExtract(meshlet_lod_offset, 0u, 31u);

    uint object_id = prefix_sum_data.instance_id;

   	bool overflow = prefix_sum_idx == total_meshlet_count;

    uint inclusive_sum = overflow ? total_meshlet_count : pc.prefix_sum_buffer.data[prefix_sum_idx + 1].sum;
    uint exclusive_sum = prefix_sum_data.sum;
    uint local_meshlet_count = inclusive_sum - exclusive_sum;

    // actual meshlet index
    uint local_meshlet_index = target - (inclusive_sum - local_meshlet_count);

    uint meshlet_offset = local_meshlet_index + meshlet_lod_offset;
	Meshlet meshlet = pc.meshlet_buffer[meshlet_offset];

	uint vertex_count = meshlet.vertex_count;
	uint triangle_count = meshlet.triangle_count;

	SetMeshOutputCounts(vertex_count, triangle_count);

	ObjectData o = pc.object_buffer[object_id];

	uint m_vertex_offset = meshlet.data_offset; // offset into meshlet_indices_buffer
	uint m_triangle_offset = m_vertex_offset + vertex_count;

	for (uint i = t_index; i < vertex_count; i += WG_SIZE_X)
	{
		uint vertex_index = pc.meshlet_indices_buffer[m_vertex_offset + i];
		Vertex v = pc.vertex_buffer[vertex_index];

		float3 pos = float3(v.px, v.py, v.pz);
		float4 world_pos = float4(rotate_quat(pos, o.orientation) * o.scale + o.translation, 1.0);

		float3 normal;
		float4 tangent;
		unpack_tbn(v.normal, uint(v.tangent), normal, tangent);

		verts[i].normal = rotate_quat(normal, o.orientation);
		verts[i].tangent = float4(rotate_quat(tangent.xyz, o.orientation), tangent.w);

		verts[i].uv = float2(v.uv_x, v.uv_y);
		verts[i].material_id = o.material_id;

	    float4 clip_pos = mul(uniforms.view_proj, world_pos);

		verts[i].position = clip_pos;
	}

#ifdef TRIANGLE_CULLING
    GroupMemoryBarrierWithGroupSync();
#endif

	for (uint i = t_index; i < triangle_count; i += WG_SIZE_X)
	{
		uint idx0 = pc.meshlet_indices_buffer[m_triangle_offset + i * 3 + 0];
		uint idx1 = pc.meshlet_indices_buffer[m_triangle_offset + i * 3 + 1];
		uint idx2 = pc.meshlet_indices_buffer[m_triangle_offset + i * 3 + 2];

		tris[i] = uint3(idx0, idx1, idx2);
	}
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
