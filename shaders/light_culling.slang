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
	uint2 workgroups;
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
#define MAX_LIGHTS 1000
#define SUBGROUP_SIZE 32

#define WG_SIZE_X 128

static const uint NUM_SUBGROUPS = WG_SIZE_X / SUBGROUP_SIZE; // nvidia
groupshared uint shared_counts[NUM_SUBGROUPS];
groupshared uint base_offset;
static const uint BATCHES = (MAX_LIGHTS + WG_SIZE_X - 1) / WG_SIZE_X;
static const uint GROUP_LIGHT_OFFSET = SUBGROUP_SIZE * BATCHES;
groupshared uint group_light_list[NUM_SUBGROUPS][SUBGROUP_SIZE * BATCHES];

[shader("compute")]
[numthreads(WG_SIZE_X, 1, 1)]
void main(
    uint3 wg_id : SV_GroupID,
    uint t_index : SV_GroupIndex,
)
{
	uint cluster_index =
		wg_id.x
		+ pc.workgroups.x * wg_id.y
		+ pc.workgroups.x * pc.workgroups.y * wg_id.z;

	ClusterAABB cluster_aabb = pc.cluster_buffer[cluster_index];

	uint local_light_list[BATCHES];
	uint local_light_count = 0;

	// each thread performs approx MAX_LIGHTS/threads light intersection test
	for (uint batch = 0; batch < BATCHES; batch++)
	{
		uint light_index = batch * WG_SIZE_X + t_index;
		if (light_index >= MAX_LIGHTS)
			break;

		PointLight light = pc.light_buffer[light_index];
		float3 light_pos = light.pos.xyz;
		float light_radius = light.pos.w;

		bool intersect = sphere_aabb_intersect(pc.view, pc.light_rot, cluster_aabb, light_pos, light_radius);

		if (intersect)
		{
			local_light_list[local_light_count++] = light_index; // record light index
		}
	}

	// subgroup # of lights
	uint group_light_count = WaveActiveSum(local_light_count);

	// subgroup exclusive add idx
	uint idx = WavePrefixSum(local_light_count);

	// TODO: potentially not portable, assumption of wave size = 32
	uint subgroup_id = t_index / WaveGetLaneCount();

	for (uint i = 0; i < local_light_count; i++)
	{
	    // note: every light is unique
		group_light_list[subgroup_id][idx + i] = local_light_list[i];
	}

	uint lane_id = WaveGetLaneIndex();

	if (lane_id == 0)
	{
		shared_counts[subgroup_id] = group_light_count;
	}

	GroupMemoryBarrierWithGroupSync();

	if (t_index == 0)
	{
	    // sum group_light_count per subgroup, and replace with exclusive sum
	    uint exclusive_sum = 0;
		uint total_count = shared_counts[0];
		shared_counts[0] = exclusive_sum;

		// TODO: optimization possible
	    for (uint i = 1; i < NUM_SUBGROUPS; i++)
		{
		    exclusive_sum = total_count;
			total_count += shared_counts[i];
			shared_counts[i] = exclusive_sum;
		}

		InterlockedAdd(*pc.light_count_buffer, total_count, base_offset);

        pc.light_grid_buffer[cluster_index].offset = base_offset;
    	pc.light_grid_buffer[cluster_index].count = total_count;
	}

	GroupMemoryBarrierWithGroupSync();

	for (uint i = lane_id; i < group_light_count; i = i + SUBGROUP_SIZE)
	{
        uint group_offset = shared_counts[subgroup_id];
		pc.light_index_buffer[base_offset + group_offset + i] = group_light_list[subgroup_id][i];
	}
}
