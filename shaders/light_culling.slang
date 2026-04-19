import bindings;
import types;

struct PushConstants
{
	float4x4 view;
	float4x4 light_rot;
	ClusterAABB* cluster_buffer;
	PointLight* light_buffer;
	uint* light_index_buffer;
	LightGrid* light_grid_buffer;
	uint* light_count_buffer;
};

[[vk::push_constant]] PushConstants pc;

float distance_squared(float3 a, float3 b)
{
	float3 d = a - b;
	return dot(d, d);
}

bool sphere_aabb_intersect(float4x4 view, float4x4 light_rot, ClusterAABB aabb, float3 center, float radius)
{
	float3 light_center = mul(view, mul(light_rot, float4(center, 1.0))).xyz;
	float3 closest = clamp(light_center, aabb.min.xyz, aabb.max.xyz);

	float d = distance_squared(closest, light_center);
	return d <= (radius * radius);
}

// TODO: move this to a config file
#define MAX_LIGHTS 1000 // assumes all lights are set and active
#define WG_SIZE_X 16 // CLUSTER_X
#define WG_SIZE_Y 8 // CLUSTER_Y
#define WG_SIZE_Z 4 // CLUSTER_Z
#define THREADS (WG_SIZE_X * WG_SIZE_Y * WG_SIZE_Z)

groupshared PointLight shared_lights[THREADS];

[shader("compute")]
[numthreads(WG_SIZE_X, WG_SIZE_Y, WG_SIZE_Z)]
void main(
    uint3 wg_id : SV_GroupID,
    uint t_index : SV_GroupIndex,
)
{
    // clusters will never be OOB due to fixed clusters
	uint cluster_index = wg_id.z * THREADS + t_index;

	ClusterAABB cluster_aabb = pc.cluster_buffer[cluster_index];

	uint visible_light_count = 0;
	// TODO: set light limit per cluster
	uint visible_lights[MAX_LIGHTS];
	uint light_offset = 0;

	while (light_offset < MAX_LIGHTS)
	{
	    uint batch_size = min(THREADS, MAX_LIGHTS - light_offset);

		if (t_index < batch_size)
		{
		    shared_lights[t_index] = pc.light_buffer[light_offset + t_index];
		}

		GroupMemoryBarrierWithGroupSync();

		for (uint i = 0; i < batch_size; i++)
		{
    		PointLight light = shared_lights[i];
    		float3 light_pos = light.pos.xyz;
    		float light_radius = light.pos.w;

    		bool intersect = sphere_aabb_intersect(pc.view, pc.light_rot, cluster_aabb, light_pos, light_radius);

    		if (intersect)
    		{
                visible_lights[visible_light_count++] = light_offset + i;
    		}
		}

		light_offset += THREADS;
	}

	GroupMemoryBarrierWithGroupSync();

	uint offset;
	InterlockedAdd(*pc.light_count_buffer, visible_light_count, offset);

	pc.light_grid_buffer[cluster_index].offset = offset;
	pc.light_grid_buffer[cluster_index].count = visible_light_count;

	for (uint i = 0; i < visible_light_count; i++)
	{
		pc.light_index_buffer[offset + i] = visible_lights[i];
	}
}
