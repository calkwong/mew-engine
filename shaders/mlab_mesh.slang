// References
// https://dl.acm.org/doi/epdf/10.1145/2556700.2556705
// https://github.com/Devsh-Graphics-Programming/Nabla/blob/master/include/nbl/builtin/glsl/ext/OIT/insert_node.glsl
// https://interplayoflight.wordpress.com/2022/07/02/order-independent-transparency-part-2/

import bindings;
import types;
import util;
import prefix_sum;

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
	PrefixSum* prefix_sum_buffer;
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

#ifdef TRIANGLE_CULLING
groupshared float3 vertex_clip[64];
#endif

#define WG_SIZE_X 128

[shader("mesh")]
[numthreads(WG_SIZE_X, 1, 1)]
[outputtopology("triangle")]
void mesh_main(
    uint3 wg_id : SV_GroupID,
    uint t_index : SV_GroupIndex,
    out vertices Interp verts[64],
    out indices uint3 tris[124],
#ifdef TRIANGLE_CULLING
    out primitives PerPrimitive prims[124]
#endif
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

		// TODO: only account for screen size if doing small primitive culling
#ifdef TRIANGLE_CULLING
		if (OPAQUE == 1)
		{
            vertex_clip[i] = float3(clip_pos.xy / clip_pos.w * 0.5 + 0.5, clip_pos.w);
        }
#endif
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

#ifdef TRIANGLE_CULLING
		// TODO: With our stress test scene, triangle culling reduces ~7M (50%) of clipping invocations
		//       but performance oddly takes a hit from ~3.6ms to ~4.3ms. Disabling triangle culling via bool
		//       improves performance slightly to ~4.1ms, but still a considerable performance hit.
		//       No idea why - possibly payload related?
        if (OPAQUE == 1)
        {
            float2 va = vertex_clip[idx0].xy;
            float2 vb = vertex_clip[idx1].xy;
            float2 vc = vertex_clip[idx2].xy;

            float2 eb = vb - va;
            float2 ec = vc - va;

            // backface + zero area culling
            bool culled = eb.x * ec.y <= eb.y * ec.x;

            // triangle culling only works if all triangles are in front of camera
            culled = culled && (vertex_clip[idx0].z > 0 && vertex_clip[idx1].z > 0 && vertex_clip[idx2].z > 0);

            // note: flickering artifacts when placed in runtime branch (last tested in GLSL)
            prims[i].cull_primitive = culled;
        }
#endif

	}
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
