import bindings;
import types;

struct PushConstants
{
	float4x4 inverse_proj;
	ClusterAABB* cluster_buffer;
	float2 screen_size;
	float2 cluster_dim;
	float nearz;
	float farz;
	uint depth_slices;
};

[[vk::push_constant]] PushConstants pc;

float3 screen_to_view(float2 screen_pos, float2 screen_size, float4x4 inverse_proj)
{
	float2 pos = min(screen_pos, screen_size); // clamp to screen_size
	float4 ndc = float4(float2(pos / screen_size), 1.0, 1.0); // reverse z
	ndc.xy = ndc.xy * 2.0 - 1.0;

	float4 view_pos = mul(inverse_proj, ndc);
	view_pos.xyz /= view_pos.w;

	return view_pos.xyz;
}

// calculates view space position of one of the cluster's corners
float3 line_intersection_to_zplane(float3 point, float z)
{
	float t = z / point.z;

	return t * point;
}

#define WG_SIZE_X 16
#define WG_SIZE_Y 8

[shader("compute")]
[numthreads(WG_SIZE_X, WG_SIZE_Y, 1)]
void main(
    uint3 t_id : SV_GroupThreadID,
    uint t_index : SV_GroupIndex,
    uint3 wg_id : SV_GroupID
)
{
	// left-bot & right-top cluster corners in screen space
	float2 min_screen_pos = float2(t_id.x, t_id.y) * pc.cluster_dim;
	float2 max_screen_pos = float2(t_id.x + 1, t_id.y + 1) * pc.cluster_dim;

	// left-bot & right-top cluster corners in view space
	float3 min_view_pos = screen_to_view(min_screen_pos, pc.screen_size, pc.inverse_proj);
	float3 max_view_pos = screen_to_view(max_screen_pos, pc.screen_size, pc.inverse_proj);

	uint slice = wg_id.z;
	uint num_slices = pc.depth_slices;

	// negation necessary as our z is negative when pointing into the screen
	float cluster_nearz = -pc.nearz * pow(pc.farz / pc.nearz, float(slice) / float(num_slices));
	float cluster_farz = -pc.nearz * pow(pc.farz / pc.nearz, float(slice + 1) / float(num_slices));

	// determine min max of tile
	float3 min_near = line_intersection_to_zplane(min_view_pos, cluster_nearz);
	float3 min_far  = line_intersection_to_zplane(min_view_pos, cluster_farz);
	float3 max_near = line_intersection_to_zplane(max_view_pos, cluster_nearz);
	float3 max_far  = line_intersection_to_zplane(max_view_pos, cluster_farz);

	float3 aabb_min = min(min(min_near, min_far), min(max_near, max_far));
	float3 aabb_max = max(max(min_near, min_far), max(max_near, max_far));

	uint clusters_per_slice = WG_SIZE_X * WG_SIZE_Y;
	uint cluster_index = wg_id.z * clusters_per_slice + t_index;

	ClusterAABB aabb;
	aabb.min = float4(aabb_min, 1.0);
	aabb.max = float4(aabb_max, 1.0);

	pc.cluster_buffer[cluster_index] = aabb;
}
