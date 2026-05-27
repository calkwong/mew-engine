import bindings;
import types;
import util;
import prefix_sum;

// for now never enable this in gbuffer path as performance tanks;
// yet to figure out why despite less work for rasterizer, payload related?
// #define TRIANGLE_CULLING
#define PBR

struct Interp
{
    float4 position : SV_Position;
    float3 normal;
    float2 uv;
    float4 tangent;
    nointerpolation uint material_id;
    // float4 clip_pos;
	// float4 prev_clip_pos;
};

#ifdef TRIANGLE_CULLING
struct PerPrimitive
{
	bool cull_primitive : SV_CullPrimitive;
};
#endif

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

[vk::constant_id(0)]
const int OPAQUE = 1;

#ifdef TRIANGLE_CULLING
groupshared float3 vertex_clip[64];
#endif

#define WG_SIZE_X 128

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

[shader("pixel")]
void ps_main(
    Interp in,
    bool is_front_face : SV_IsFrontFace,
    out float4 out_albedo : SV_Target0,
    out float4 out_normal : SV_Target1,
    out float4 out_metal_roughness_occlusion : SV_Target2,
    out float4 out_emissive : SV_Target3
)
{
	MaterialData m = pc.material_buffer[in.material_id];

	float4 albedo = m.base_color_factor;
	if (m.diffuse_id != 0)
	{
		float4 sampled_albedo = textures[m.diffuse_id].Sample(samplers[LINEAR_SAMPLER], in.uv);

		if (OPAQUE == 0)
		{
			if (sampled_albedo.a < 0.5)
				discard;
		}

		albedo *= sampled_albedo;
	}

	float3 N;
	float metallic;
	float roughness;

#ifdef PBR
	N = in.normal; // mikktspace

	if (OPAQUE == 0)
	{
		N = is_front_face ? N : -N;
	}

	if (m.normal_id != 0)
	{
		float4 T = in.tangent; // mikktspace

		// T.w is naturally flipped during tangent generation so mikktspace is consistent with glTF handedness
		if (OPAQUE == 0)
			T.w = is_front_face ? T.w : -T.w;

		float3 shading_normal = textures[m.normal_id].Sample(samplers[LINEAR_SAMPLER], in.uv).xyz;
		shading_normal = shading_normal * 2.0 - 1.0;
		N = tangent_to_world_space(shading_normal, T, N);
	}
	else
	{
        N = normalize(N);
	}

	metallic = m.metallic_factor;
	float perceptual_roughness = m.roughness_factor;
	float2 metal_roughness = float2(0.0);
	if (m.metalroughness_id != 0)
	{
		metal_roughness = textures[m.metalroughness_id].Sample(samplers[LINEAR_SAMPLER], in.uv).bg;
		metallic *= metal_roughness.x;
		perceptual_roughness *= metal_roughness.y;
	}
	perceptual_roughness = max(perceptual_roughness, 0.045); // frostbite engine clamp value for analytical lights (fp32)
#else
	N = normalize(in.normal);
#endif

    float ao = 1.0;
    if (m.occlusion_id != 0)
    {
        ao = textures[m.occlusion_id].Sample(samplers[LINEAR_SAMPLER], in.uv).r;
    }

    float3 emissive = m.emissive_factor;
	if (m.emissive_id != 0)
	{
	    emissive *= textures[m.emissive_id].Sample(samplers[LINEAR_SAMPLER], in.uv).rgb;
	}

    out_albedo = albedo;
    out_normal = float4(N, 1);
    out_metal_roughness_occlusion = float4(metallic, perceptual_roughness, ao, 1.0);
    out_emissive = float4(emissive, 1.0);
}
