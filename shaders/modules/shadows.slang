module shadows;

import bindings;

internal static const int CASCADE_COUNT = 4;
internal static const float EPSILON = 0.1; // avoids self intersection

// TODO: this should be in the main shader file, not here, but for convenience's sake
[[vk::binding(0, 4)]] internal RaytracingAccelerationStructure acceleration_structure;

public bool in_shadow(float3 pos)
{
    RayDesc shadow_ray_desc;
    shadow_ray_desc.Direction = normalize(uniforms.sunlight_dir.xyz);
    shadow_ray_desc.Origin = pos;
    shadow_ray_desc.TMin = EPSILON;
    shadow_ray_desc.TMax = 1e4;

    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;

    let ray_flags = RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;

    sq.TraceRayInline(acceleration_structure, ray_flags, 0xFF, shadow_ray_desc);
    sq.Proceed();

    bool hit = (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT);

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
