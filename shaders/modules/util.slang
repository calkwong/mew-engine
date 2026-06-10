module util;

public float3 srgb_to_linear(float3 c)
{
	return pow(c, float3(2.2));
}

public float4 srgb_to_linear(float4 c)
{
	return float4(pow(c.xyz, float3(2.2)), c.w);
}

public float3 linear_to_srgb(float3 c)
{
	return pow(c, float3(1.0 / 2.2));
}

public float4 linear_to_srgb(float4 c)
{
	return float4(pow(c.xyz, float3(1.0 / 2.2)), c.w);
}

// 10-10-10-2
public float3 decode_normal(uint n)
{
	return normalize(((int3(n) >> int3(20, 10, 0)) & int3(1023)) / 511.0 - 1.0);
}

// 8-8
public float2 decode_tangent(uint t)
{
	return ((int2(t) >> int2(8, 0)) & int2(255)) / 127.0 - 1.0;
}

public float2 encode_oct(float3 n)
{
	n /= (abs(n.x) + abs(n.y) + abs(n.z));
	float2 s = float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
	n.xy = n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * s;
	//n.xy = n.xy * 0.5 + 0.5;
	return n.xy;
}

// https://x.com/Stubbesaurus/status/937994790553227264/
public float3 decode_oct(float2 f)
{
	float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
	float t = max(-n.z, 0.0);
	n.xy += float2(n.x >= 0.0 ? -t : t, n.y >= 0.0 ? -t : t);
	return normalize(n);
}

public void unpack_tbn(uint n, uint t, out float3 normal, out float4 tangent)
{
	normal = decode_normal(n);
	float2 tp = decode_tangent(t);
	tangent.xyz = decode_oct(tp);
	tangent.w = (n & (1 << 30)) != 0 ? 1.0 : -1.0;
}

public float3 rotate_quat(float3 v, float4 q)
{
	return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

public uint hash(uint a)
{
   a = (a+0x7ed55d16) + (a<<12);
   a = (a^0xc761c23c) ^ (a>>19);
   a = (a+0x165667b1) + (a<<5);
   a = (a+0xd3a2646c) ^ (a<<9);
   a = (a+0xfd7046c5) + (a<<3);
   a = (a^0xb55a4f09) ^ (a>>16);
   return a;
}

// assumptions: infinite far plane, reverse-z
// returns positive z, negate depending on usage
public float linearize_depth(float near, float depth)
{
	return near / depth;
}

#define CLUSTER_DEPTH_SLICES 24
// equation (3): https://www.aortiz.me/2018/12/21/CG.html#part-2
// slide 5: https://advances.realtimerendering.com/s2016/Siggraph2016_idTech6.pdf
// view_z must be positive!
public uint get_cluster_z_slice(float view_z, float scale, float bias)
{
    uint slice = uint(floor(log(view_z) * scale - bias));

    return clamp(slice, 0, CLUSTER_DEPTH_SLICES - 1);
}

#define CLUSTER_X 16
#define CLUSTER_Y 8
public uint get_cluster_index(float2 frag_coord, float2 cluster_dim, uint slice)
{
    uint2 clusters = uint2(floor(frag_coord / cluster_dim));

    return clusters.x + CLUSTER_X * clusters.y + slice * CLUSTER_X * CLUSTER_Y;
}

// https://www.shadertoy.com/view/XlGcRh
public uint pcg(inout uint seed)
{
    uint state = seed * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

public float random_value(inout uint seed)
{
    return float(pcg(seed)) / 4294967295.0; // 2^32 - 1
}

public float3 tangent_to_world_space(float3 sample, float3 N)
{
    float3 up = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);

    return normalize(T * sample.x + B * sample.y + N * sample.z);
}

// mikktspace, TBN not normalized
public float3 tangent_to_world_space(float3 sample, float4 T, float3 N)
{
    float3 B = cross(N, T.xyz) * T.w;

    return normalize(T.xyz * sample.x + B * sample.y + N * sample.z);
}

public float3 tangent_to_world_space(float3 sample, float3 T, float3 B, float3 N)
{
    return normalize(T * sample.x + B * sample.y + N * sample.z);
}

public float3 reconstruct_world_pos(float depth, float4x4 inverse_view_proj, float2 uv_pos)
{
	float2 ndc = uv_pos * 2.0 - 1.0;
	ndc.y *= -1.0; // flip as window coords are top down
	float4 world_pos = mul(inverse_view_proj, float4(ndc, depth, 1.0));

	return world_pos.xyz / world_pos.w;
}

// TODO: convert a sampled vector to world space
// TODO: offset_ray, https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf Chapter 6
