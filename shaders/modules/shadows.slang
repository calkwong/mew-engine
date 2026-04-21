module shadows;

import bindings;
import types;

internal static const int CASCADE_COUNT = 4;
internal static const float EPSILON = 0.1; // avoids self intersection

// TODO: this should be in the main shader file, not here, but for convenience's sake
[[vk::binding(0, 4)]] internal RaytracingAccelerationStructure acceleration_structure;

internal float2 intersection_uv(uint mesh_id, uint prim_index, float2 barycentrics,
    ObjectData* object_buffer, MeshData* mesh_buffer, uint* index_buffer, Vertex* vertex_buffer
)
{
    uint vertex_offset = mesh_buffer[mesh_id].vertex_offset;
    uint index_offset = mesh_buffer[mesh_id].lods[0].first_index;

    uint idx0 = index_buffer[index_offset + (prim_index * 3 + 0)];
    uint idx1 = index_buffer[index_offset + (prim_index * 3 + 1)];
    uint idx2 = index_buffer[index_offset + (prim_index * 3 + 2)];

    float2 uv0 = float2(vertex_buffer[vertex_offset + idx0].uv_x, vertex_buffer[vertex_offset + idx0].uv_y);
    float2 uv1 = float2(vertex_buffer[vertex_offset + idx1].uv_x, vertex_buffer[vertex_offset + idx1].uv_y);
    float2 uv2 = float2(vertex_buffer[vertex_offset + idx2].uv_x, vertex_buffer[vertex_offset + idx2].uv_y);

    float w0 = 1.0 - barycentrics.x - barycentrics.y;
    float w1 = barycentrics.x;
    float w2 = barycentrics.y;

    return (uv0 * w0 + uv1 * w1 + uv2 * w2);
}

// note:
// - experiment with a RT buffer to reduce indirection - niagara tried this with no performance benefits
// - try 2 ray approach
// - opacity micromap
// - soft shadows
// - trace at lower resolution
public bool in_shadow(float3 pos, ObjectData* object_buffer, MeshData* mesh_buffer, MaterialData* material_buffer, uint* index_buffer, Vertex* vertex_buffer)
{
    RayDesc shadow_ray_desc;
    shadow_ray_desc.Direction = normalize(uniforms.sunlight_dir.xyz);
    shadow_ray_desc.Origin = pos;
    shadow_ray_desc.TMin = EPSILON;
    shadow_ray_desc.TMax = 1e4;

    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> rq;

    let ray_flags = RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;

    rq.TraceRayInline(acceleration_structure, ray_flags, 0xFF, shadow_ray_desc);

    while (rq.Proceed())
    {
        uint instance_id = rq.CandidateRayInstanceCustomIndex();
        uint prim_index = rq.CandidatePrimitiveIndex();

        uint mesh_id = object_buffer[instance_id].mesh_id;
        uint material_id = object_buffer[instance_id].material_id;
        uint albedo_id = material_buffer[material_id].diffuse_id;
        float2 uv = intersection_uv(mesh_id, prim_index, rq.CandidateTriangleBarycentrics(), object_buffer, mesh_buffer, index_buffer, vertex_buffer);

        // enable mip sampling or no?
        float4 color = textures[albedo_id].SampleLevel(samplers[LINEAR_SAMPLER], uv, 0.0);
        // float4 color = textures[albedo_id].Sample(samplers[LINEAR_SAMPLER], uv);
        if (color.a < 0.5)
        {
        }
        else
        {
            rq.CommitNonOpaqueTriangleHit();
        }
    }

    bool hit = (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT);

    return hit;
}

public float calculate_shadow(float3 world_pos, inout uint cascade_index, uint shadowmap_id)
{
    float3 ndc;

    for (int i = 0; i < CASCADE_COUNT; i++)
    {
        ndc = mul(uniforms.shadow_transforms[i], float4(world_pos, 1.0)).xyz;
        // z plane check is dependent on cascade selection strategy and potentially if we're doing pancaking?
        if (abs(ndc.x) < 1.0 && abs(ndc.y) < 1.0 && ndc.z > 0.0 && ndc.z < 1.0)
        {
            cascade_index = i;
            break;
        }
    }

    float current_depth = ndc.z;

	if (current_depth < 0.0) // necessary
        return 1.0;

	float2 uv = ndc.xy * 0.5 + 0.5;
	uv.y = 1.0 - uv.y;

	float layers;
	float2 offset;
	textures[shadowmap_id + cascade_index].GetDimensions(0, offset.x, offset.y, layers);
	offset = float2(1.0) / offset;

	float shadow = 0.0;
	float closest_depth = 0.0;

    // TODO: bias
	for (int y = -1; y <= 1; y++)
	{
		for (int x = -1; x <= 1; x++)
		{
			float2 sample_uv = float2(uv.x + x * offset.x, uv.y + y * offset.y);
			closest_depth = textures[shadowmap_id + cascade_index].SampleLevel(samplers[SHADOW_SAMPLER], sample_uv, 0.0).r;

			if (closest_depth > current_depth)
				shadow += 0.0;
			else
				shadow += 1.0;
		}
	}

	shadow /= 9.0;

	return shadow;
}
