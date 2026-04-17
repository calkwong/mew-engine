import bindings;
import types;
import util;
import prefix_sum;

struct Interp
{
	float4 position : SV_Position;
	float2 uv;
	nointerpolation uint material_id;
	// float4 clip_pos;
	// float4 prev_clip_pos;
};

struct PerPrimitive
{
    uint draw_id;
	uint triangle_id;

	bool cull_primitive : SV_CullPrimitive;
};

[vk::constant_id(0)]
const int OPAQUE = 1;

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

#define TRIANGLE_CULLING

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
    out primitives PerPrimitive prims[124]
)
{
    uint id = wg_id.x;

    uint64_t counter = get_prefix_sum_counter(pc.prefix_sum_buffer);
	uint total_meshlet_count = uint(counter);
    uint target = pc.cluster_indices_buffer[id];

    uint prefix_sum_idx = prefix_sum_binary_search(pc.prefix_sum_buffer, target);
    PrefixSumData prefix_sum_data = pc.prefix_sum_buffer.data[prefix_sum_idx];
    uint meshlet_lod_offset = prefix_sum_data.lod_offset;

    // we do some packing during culling hence this is essential!
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
        uint vertex_index = pc.meshlet_indices_buffer[m_vertex_offset + i]; // vertex indices for meshlets are global, really need to document this
        Vertex v = pc.vertex_buffer[vertex_index];

        float3 pos = float3(v.px, v.py, v.pz);
        float4 world_pos = float4(rotate_quat(pos, o.orientation) * o.scale + o.translation, 1.0);

        verts[i].uv = float2(v.uv_x, v.uv_y);
        verts[i].material_id = o.material_id; // only used for alpha clipping

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

    GroupMemoryBarrierWithGroupSync();

    for (uint i = t_index; i < triangle_count; i += WG_SIZE_X)
    {
        prims[i].draw_id = object_id;

        // WARNING: unsafe if meshlet_offset is large
        prims[i].triangle_id = (i << 25) | meshlet_offset;

        uint idx0 = pc.meshlet_indices_buffer[m_triangle_offset + i * 3 + 0];
        uint idx1 = pc.meshlet_indices_buffer[m_triangle_offset + i * 3 + 1];
        uint idx2 = pc.meshlet_indices_buffer[m_triangle_offset + i * 3 + 2];

        tris[i] = uint3(idx0, idx1, idx2);

#ifdef TRIANGLE_CULLING
        // backface + zero area culling
        if (OPAQUE == 1)
        {
            float2 va = vertex_clip[idx0].xy;
            float2 vb = vertex_clip[idx1].xy;
            float2 vc = vertex_clip[idx2].xy;

            float2 eb = vb - va;
            float2 ec = vc - va;

            // triangle backface + zero area culling
            bool culled = eb.x * ec.y <= eb.y * ec.x;

            // triangle culling only works if all triangles are in front of camera
            culled = culled && (vertex_clip[idx0].z > 0 && vertex_clip[idx1].z > 0 && vertex_clip[idx2].z > 0);

            // this cannot be in a runtime branch! compiler likely sets this to a default value after dead code removal if it cannot find gl_CullPrimitiveEXT
            // if it exists and we place it in a branch, we get flickering artifacts all over
            prims[i].cull_primitive = culled;
        }
#endif
    }
}

[shader("pixel")]
void ps_main(
    Interp in,
    PerPrimitive prim,
    out uint2 visibility_id : SV_Target0,
    // out float2 velocity_id : SV_Target1
)
{
	if (OPAQUE == 0)
	{
		MaterialData m = pc.material_buffer[in.material_id];
		float4 albedo = textures[m.diffuse_id].Sample(samplers[LINEAR_SAMPLER], in.uv);

		if (albedo.a < 0.5)
			discard;
	}

	// TODO: visibility_id can be packed into a single value
	visibility_id = uint2(prim.draw_id, prim.triangle_id);
	// velocity_id = float2(0.0);
}
