// note: not functional, needs complete rewrite

import bindings;
import types;
import barycentrics;
import util;
import sh;
import pbr;
import sampling;
//import rt_helpers; // rewrite needed, lost this untracked file when switching os

[[vk::binding(0, 4)]] RaytracingAccelerationStructure acceleration_structure;

struct PushConstants
{
	float4 cluster_size; // xy is cluster dim, z is # of cluster slices
	float2 screen_size;
	PointLight* light_buffer;
	uint* light_index_buffer;
	LightGrid* light_grid_buffer;
	// OITData* oit_buffer;
	uint* padding1;
	uint* meshlet_indices_buffer;
	Meshlet* meshlet_buffer;
	Vertex* vertex_buffer;
	ObjectData* object_buffer;
	MaterialData* material_buffer;
	uint* index_buffer;
	MeshData* mesh_buffer;
	SH* sh_buffer;
	uint depth_id;
	uint gbuffer_id;
	uint shadowmap_id;
	uint light_culling;
	float near;
	float scale;
	float bias;
	uint resolve_transparent;
	uint shadows;
	uint shadows_rt;
	float max_prefiltered_lod;
	float metallic; // unused, for furnace test
	float roughness; // unused, for furnace test
	uint debug;
};

[[vk::push_constant]] PushConstants pc;

#define PBR

static const float PI = 3.14159265359;

float3 get_environment_lighting(/* float y */)
{
    // return lerp(float3(1.0), float3(0.5, 0.7, 1.0), y);
    return float3(0.5, 0.7, 1.0);
}

// notes:
// hybrid implementation where visibility is determined by rasterization, while RT drives everything else

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
    uint2 global_id : SV_DispatchThreadID
)
{
    float2 frag_coord = float2(global_id.x, global_id.y) + float2(0.5);

    if (frag_coord.x >= pc.screen_size.x || frag_coord.y >= pc.screen_size.y)
        return;

    float4 out_color;
    float2 in_uv = frag_coord / pc.screen_size;

	uint2 data = textures_u[pc.gbuffer_id].SampleLevel(samplers[NEAREST_SAMPLER], in_uv, 0.0).rg;

	uint draw_id = data.x;
	uint packed_id = data.y;
	uint triangle_id = bitfieldExtract(packed_id, 25u, 7u);
	uint meshlet_id = bitfieldExtract(packed_id, 0u, 25u);

	Meshlet meshlet = pc.meshlet_buffer[meshlet_id];
	uint triangle_offset = meshlet.data_offset + meshlet.vertex_count;

	uint idx0 = pc.meshlet_indices_buffer[triangle_offset + triangle_id * 3 + 0];
	uint idx1 = pc.meshlet_indices_buffer[triangle_offset + triangle_id * 3 + 1];
	uint idx2 = pc.meshlet_indices_buffer[triangle_offset + triangle_id * 3 + 2];

	uint vertex_index0 = pc.meshlet_indices_buffer[meshlet.data_offset + idx0];
	uint vertex_index1 = pc.meshlet_indices_buffer[meshlet.data_offset + idx1];
	uint vertex_index2 = pc.meshlet_indices_buffer[meshlet.data_offset + idx2];

	Vertex v0 = pc.vertex_buffer[vertex_index0];
	Vertex v1 = pc.vertex_buffer[vertex_index1];
	Vertex v2 = pc.vertex_buffer[vertex_index2];

	ObjectData obj = pc.object_buffer[draw_id];

	float3 wp0 = rotate_quat(float3(v0.px, v0.py, v0.pz), obj.orientation) * obj.scale + obj.translation;
	float3 wp1 = rotate_quat(float3(v1.px, v1.py, v1.pz), obj.orientation) * obj.scale + obj.translation;
	float3 wp2 = rotate_quat(float3(v2.px, v2.py, v2.pz), obj.orientation) * obj.scale + obj.translation;

	float4 p0 = mul(uniforms.view_proj, float4(wp0, 1.0));
	float4 p1 = mul(uniforms.view_proj, float4(wp1, 1.0));
	float4 p2 = mul(uniforms.view_proj, float4(wp2, 1.0));

    // we want y up here, not y with top left origin
	float2 pNdc = in_uv * 2.0 - 1.0;
	pNdc.y = -pNdc.y;

	BarycentricDeriv bary = calculate_barycentric(p0, p1, p2, pNdc, pc.screen_size);
	float2 uv;
	float2 uv_ddx;
	float2 uv_ddy;
	interpolate_with_deriv(bary, float2(v0.uv_x, v0.uv_y), float2(v1.uv_x, v1.uv_y), float2(v2.uv_x, v2.uv_y), uv, uv_ddx, uv_ddy);

	float3 debug_uv = float3(uv, 0.0);

	float3 world_pos = interpolate(bary, wp0, wp1, wp2);

	float3 np0, np1, np2;
	float4 tp0, tp1, tp2;
	unpack_tbn(v0.normal, uint(v0.tangent), np0, tp0);
	unpack_tbn(v1.normal, uint(v1.tangent), np1, tp1);
	unpack_tbn(v2.normal, uint(v2.tangent), np2, tp2);

	float3 V = normalize(uniforms.camera_pos.xyz - world_pos);

	// alternative: store triangle face bit in visibility buffer
	float3 e1 = wp1 - wp0;
	float3 e2 = wp2 - wp0;
	float3 gN = cross(e1, e2);
	bool is_back_face = dot(gN, V) < 0.0;

	float3 n0 = rotate_quat(np0, obj.orientation);
	float3 n1 = rotate_quat(np1, obj.orientation);
	float3 n2 = rotate_quat(np2, obj.orientation);

    float3 N = interpolate(bary, n0, n1, n2); // mikktspace

	N = is_back_face ? -N : N;

	uint material_id = obj.material_id;
	MaterialData m = pc.material_buffer[material_id];

	float4 albedo = m.base_color_factor;

	if (m.diffuse_id != 0)
	{
		albedo.xyz *= textures[m.diffuse_id].SampleGrad(samplers[LINEAR_SAMPLER], uv, uv_ddx, uv_ddy).xyz;
	}

	float3 color = albedo.xyz;

	float4 t0 = float4(rotate_quat(tp0.xyz, obj.orientation), tp0.w);
	float4 t1 = float4(rotate_quat(tp1.xyz, obj.orientation), tp1.w);
	float4 t2 = float4(rotate_quat(tp2.xyz, obj.orientation), tp2.w);

	float4 T = interpolate(bary, t0, t1, t2); // mikktspace
	float sign = T.w; // sign is flipped during tangent generation so mikktspace is consistent with glTF handedness

	sign = is_back_face ? -sign : sign;

#ifdef PBR
	if (m.normal_id != 0)
	{
		float3 B = sign * cross(N, T.xyz);
		float3 shading_normal = textures[m.normal_id].SampleGrad(samplers[LINEAR_SAMPLER], uv, uv_ddx, uv_ddy).xyz;
		shading_normal = shading_normal * 2.0 - 1.0;
		N = normalize(shading_normal.x * T.xyz + shading_normal.y * B + shading_normal.z * N);
	}
	else
	    N = normalize(N);

	float metallic = m.metallic_factor;
	float perceptual_roughness = m.roughness_factor;
	float2 metal_roughness = float2(0.0);
	if (m.metalroughness_id != 0)
	{
		metal_roughness = textures[m.metalroughness_id].SampleGrad(samplers[LINEAR_SAMPLER], uv, uv_ddx, uv_ddy).bg;
		metallic *= metal_roughness.x;
		perceptual_roughness *= metal_roughness.y;
	}

	perceptual_roughness = max(perceptual_roughness, 0.045); // frostbite engine clamp value for analytical lights (fp32)
	float roughness = perceptual_roughness * perceptual_roughness;

	// float3 Fr = float3(0.0);

	// float3 L = normalize(uniforms.sunlight_dir.xyz);
	// float3 H = normalize(L + V);

	// float NdotL = max(dot(N, L), 0.0);
	// float NdotH = max(dot(N, H), 0.0);
	// float NdotV = max(dot(N, V), 0.001);
	// float VdotH = max(dot(V, H), 0.0);

	// float3 f0 = float3(0.04);
	// f0 = lerp(f0, albedo.xyz, metallic);

	// float3 F = F_Schlick(VdotH, f0);

	// float3 kS = F;
	// float3 kD = float3(1.0) - kS;

	// kD *= 1.0 - metallic;
	// float3 Fd = kD * albedo.xyz / PI;

	// float D = D_GGX(NdotH, roughness);
	// float G = V_SmithGGXCorrelated(NdotV, NdotL, roughness);
	// Fr = D * G * F;

	// float3 light_color = uniforms.sunlight_color.xyz;
	// color = (Fd + Fr) * light_color * NdotL;

#else
	color += albedo.xyz;
#endif

	float depth = textures[pc.depth_id].SampleLevel(samplers[NEAREST_SAMPLER], in_uv, 0.0).r;

	if (depth == 0.0)
	{
        color = float3(1.0);
	}

	uint seed = global_id.x + global_id.y * 1700;
	float3 scatter_dir = random_cosine_hemisphere(seed, N);

	// TODO: fix

	out_color = float4(color, 1.0);

    // TODO: hardcoded draw id to 0
	images_rgba32f[0].Store(int2(frag_coord), out_color);
}
