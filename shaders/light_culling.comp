#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#include "bindings.glsl"
#include "buffer_references.glsl"

// 1 cluster per workgroup
layout (local_size_x = 128) in;

layout( push_constant ) uniform constants
{
	mat4 view;
	mat4 light_rot;
	ClusterBuffer cluster_buffer;
	LightBuffer light_buffer;
	LightIndexBuffer light_index_buffer;
	LightGridBuffer light_grid_buffer;
	LightCountBuffer light_count_buffer;
};

float distance_squared(vec3 a, vec3 b)
{
	vec3 d = a - b;
	return dot(d, d);
}

bool sphere_aabb_intersect(ClusterAABB aabb, vec3 center, float radius)
{
	vec3 light_center = vec3(view * light_rot * vec4(center, 1.0));
	vec3 closest = clamp(light_center, aabb.min.xyz, aabb.max.xyz);

	float d = distance_squared(closest, light_center);
	return d <= (radius * radius);
}

// TODO: move this to a config file
#define MAX_LIGHTS 1000
#define SUBGROUP_SIZE 32

const uint NUM_SUBGROUPS = gl_WorkGroupSize.x / SUBGROUP_SIZE; // nvidia
shared uint shared_counts[NUM_SUBGROUPS];
shared uint base_offset;
const uint BATCHES = (MAX_LIGHTS + gl_WorkGroupSize.x - 1) / gl_WorkGroupSize.x;
const uint GROUP_LIGHT_OFFSET = SUBGROUP_SIZE * BATCHES;
shared uint group_light_list[NUM_SUBGROUPS][SUBGROUP_SIZE * BATCHES];

void main()
{
	uint cluster_index =
		gl_WorkGroupID.x
		+ gl_NumWorkGroups.x * gl_WorkGroupID.y
		+ gl_NumWorkGroups.x * gl_NumWorkGroups.y * gl_WorkGroupID.z;

	ClusterAABB cluster_aabb = cluster_buffer.aabb[cluster_index];

	uint local_light_list[BATCHES];
	uint local_light_count = 0;

	// each thread performs approx MAX_LIGHTS/threads light intersection test
	for (uint batch = 0; batch < BATCHES; batch++)
	{
		uint light_index = batch * gl_WorkGroupSize.x + gl_LocalInvocationIndex;
		if (light_index >= MAX_LIGHTS)
			break;

		PointLight light = light_buffer.lights[light_index];
		vec3 light_pos = light.pos.xyz;
		float light_radius = light.pos.w;

		bool intersect = sphere_aabb_intersect(cluster_aabb, light_pos, light_radius);

		if (intersect)
		{
			local_light_list[local_light_count++] = light_index; // record light index
		}
	}

	// subgroup # of lights
	uint group_light_count = subgroupAdd(local_light_count);

	// subgroup exclusive add idx
	uint idx = subgroupExclusiveAdd(local_light_count);

	for (uint i = 0; i < local_light_count; i++)
	{
	    // note: every light is unique
		group_light_list[gl_SubgroupID][idx + i] = local_light_list[i];
	}

	if (gl_SubgroupInvocationID == 0)
	{
		shared_counts[gl_SubgroupID] = group_light_count;
	}

	barrier();

	if (gl_LocalInvocationIndex == 0)
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

		base_offset = atomicAdd(light_count_buffer.count, total_count);

        light_grid_buffer.grid[cluster_index].offset = base_offset;
    	light_grid_buffer.grid[cluster_index].count = total_count;
	}

	barrier();

	for (uint i = gl_SubgroupInvocationID; i < group_light_count; i = i + SUBGROUP_SIZE)
	{
        uint group_offset = shared_counts[gl_SubgroupID];
		light_index_buffer.indices[base_offset + group_offset + i] = group_light_list[gl_SubgroupID][i];
	}
}
