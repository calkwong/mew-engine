import bindings;
import types;
import util;

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

struct PushConstants
{
	ObjectData* object_buffer;
	Vertex* vertex_buffer;
	Meshlet* meshlet_buffer;
	uint* meshlet_indices_buffer;
	uint* cluster_indices_buffer;
	MaterialData* material_buffer;
	// PrefixSum* prefix_sum_buffer;
	uint* padding;
	uint2 screen_size;
	float4 jitter_offset;
};

[[vk::push_constant]] PushConstants pc;

[vk::constant_id(0)]
const int OPAQUE = 1;

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

[shader("pixel")]
void ps_main(
    Interp in,
    bool is_front_face : SV_IsFrontFace,
    out float4 out_albedo : SV_Target0,
    out float4 out_normal : SV_Target1,
    out float2 out_metal_roughness : SV_Target2
    // out float2 out_velocity : SV_Target3
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
		float3 T = in.tangent.xyz; // mikktspace
		float sign = in.tangent.w; // sign is flipped during tangent generation so mikktspace is consistent with glTF handedness

		if (OPAQUE == 0)
			sign = is_front_face ? sign : -sign;

		float3 B = sign * cross(N, T);

		float3 shading_normal = textures[m.normal_id].Sample(samplers[LINEAR_SAMPLER], in.uv).xyz;
		shading_normal = shading_normal * 2.0 - 1.0;
		N = shading_normal.x * T.xyz + shading_normal.y * B + shading_normal.z * N;
	}

    N = normalize(N);

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

    out_albedo = albedo;
    out_normal = float4(N, 1);
    out_metal_roughness = float2(metallic, perceptual_roughness);
}
