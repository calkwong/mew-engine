import bindings;
import types;
import pbr;
import sh;
import util;
import oit;
import shadows;

struct PushConstants
{
    float4 cluster_size; // xy is cluster dim, z is # of cluster slices
	float2 screen_size;
	PointLight* light_buffer;
	uint* light_index_buffer;
	LightGrid* light_grid_buffer;
	OITData* oit_buffer;
	uint padding[10]; // padding for visibility buffer variant
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
	float max_prefiltered_lod;
	float pc_metallic; // unused, for furnace test
	float pc_roughness; // unused, for furnace test
	uint debug;
};

[[vk::push_constant]] PushConstants pc;

float3 reconstruct_world_pos(float depth, float4x4 inverse_view_proj, float2 uv_pos)
{
	//float2 ndc = gl_FragCoord.xy / screen_size;
	float2 ndc = uv_pos * 2.0 - 1.0;
	ndc.y *= -1.0; // flip as window coords are top down
	float4 world_pos = mul(inverse_view_proj, float4(ndc, depth, 1.0));

	return world_pos.xyz / world_pos.w;
}

#define DEBUG_SHADING_NORMALS 1
#define DEBUG_TANGENTS 2
#define DEBUG_TANGENT_SIGN 3
#define DEBUG_UV 4
#define DEBUG_SHADOWMAP 5
#define DEBUG_CASCADES 6
#define DEBUG_MESHLETS 7
#define DEBUG_VELOCITY 8

#define PBR
#define IBL

static const float PI = 3.14159265359;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
    uint3 global_id : SV_DispatchThreadID
)
{
    float2 frag_coord = float2(global_id.x, global_id.y) + float2(0.5);

    if (frag_coord.x >= pc.screen_size.x || frag_coord.y >= pc.screen_size.y)
        return;

    float4 out_color;
    float2 in_uv = frag_coord / pc.screen_size;

	uint albedo_id = pc.gbuffer_id;
	uint normal_id = pc.gbuffer_id + 1;
	uint metalroughness_id = pc.gbuffer_id + 2;

	float3 N = textures[normal_id].SampleLevel(samplers[NEAREST_SAMPLER], in_uv, 0.0).xyz;
	N = normalize(N); // necessary to remove banding, RGB32 does not need this

	float4 albedo = float4(textures[albedo_id].SampleLevel(samplers[NEAREST_SAMPLER], in_uv, 0.0).xyz, 1.0);
	float depth = textures[pc.depth_id].SampleLevel(samplers[NEAREST_SAMPLER], in_uv, 0.0).r;
	float3 world_pos = reconstruct_world_pos(depth, uniforms.inverse_view_proj, in_uv);

	float3 color = float3(0.0);
	float3 ambient = albedo.xyz * 0.1; // when IBL is off, likely going to look physically incorrect

#ifdef PBR
	float2 metal_roughness = textures[metalroughness_id].SampleLevel(samplers[NEAREST_SAMPLER], in_uv, 0.0).xy;
	float metallic = metal_roughness.x;
	float perceptual_roughness = metal_roughness.y;
	float roughness = perceptual_roughness * perceptual_roughness;

	float3 Fr = float3(0.0);

	float3 L = normalize(uniforms.sunlight_dir.xyz);
	float3 V = normalize(uniforms.camera_pos.xyz - world_pos);
	float3 H = normalize(L + V);

	float NdotL = max(dot(N, L), 0.0);
	float NdotH = max(dot(N, H), 0.0);
	float NdotV = max(dot(N, V), 0.001);
	float VdotH = max(dot(V, H), 0.0);

	float3 f0 = float3(0.04);
	f0 = lerp(f0, albedo.xyz, metallic);

	float3 F = F_Schlick(VdotH, f0);

	float3 kS = F;
	float3 kD = float3(1.0) - kS;

	kD *= 1.0 - metallic;
	float3 Fd = kD * albedo.xyz / PI;

	float D = D_GGX(NdotH, roughness);
	float G = V_SmithGGXCorrelated(NdotV, NdotL, roughness);
	Fr = D * G * F;

	float3 light_color = uniforms.sunlight_color.xyz;
	color = (Fd + Fr) * light_color * NdotL;

	#ifdef IBL
		//perceptual_roughness = pc_roughness; // sphere test
		//metallic = pc_metallic; // sphere test

		float3 R = reflect(-V, N);
		float prefiltered_mip = pc.max_prefiltered_lod * perceptual_roughness;

		{
			float3 irradiance = evaluate_sh(pc.sh_buffer->r_coefficients, pc.sh_buffer->g_coefficients, pc.sh_buffer->b_coefficients, N);
			float3 prefiltered = textures_cube[uint(uniforms.textures[2])].SampleLevel(samplers[CUBE_SAMPLER], R, prefiltered_mip).xyz;
			float2 brdf = textures[uint(uniforms.textures[3])].SampleLevel(samplers[LINEAR_CLAMP_SAMPLER], float2(NdotV, perceptual_roughness), 0.0).rg;

			//float3 white = float3(1.0); // sphere test
			//f0 = float3(0.04); // sphere test
			f0 = lerp(f0, albedo.xyz, metallic);
			//f0 = lerp(f0, white, metallic); // sphere test

			F = F_SchlickRoughness(NdotV, f0, perceptual_roughness);
			kS = F;
			kD = float3(1.0) - kS;
			kD *= 1.0 - metallic;

			// TODO: multiscattering?

		    // TODO: we could bake the division by PI into SH
			float3 diffuse = kD * irradiance * (1.0 / PI) * albedo.xyz;
			float3 specular = prefiltered * (F * brdf.x + brdf.y);
			ambient = diffuse + specular;
			//out_color = float4(ambient, 1.0); // sphere test
			// imageStore(images_rgba32f[0], int2(frag_coord), out_color); // sphere test
			//return; // sphere test
		}
	#endif
#else
	color += albedo.xyz;
#endif

	uint cascade_index = 0;
	if (pc.shadows == 1)
	{
		float occluded = calculate_shadow(world_pos, cascade_index, pc.shadowmap_id);
        color *= occluded;
	}

	color += ambient;

	if (pc.light_culling == 1)
	{
    	float view_z = linearize_depth(pc.near, depth); // alternative: calculate via matrix multiply, likely more expensive?
    	uint slice = get_cluster_z_slice(view_z, pc.scale, pc.bias);
    	uint cluster_index = get_cluster_index(float2(frag_coord.x, pc.screen_size.y - frag_coord.y), pc.cluster_size.xy, slice);

		LightGrid grid = pc.light_grid_buffer[cluster_index];
		uint offset = grid.offset;
		uint count = grid.count;

		for (int i = 0; i < count; i++)
		{
			uint index = pc.light_index_buffer[offset + i];

			PointLight point_light = pc.light_buffer[index];
			float3 light_pos = point_light.pos.xyz;
			float light_radius = point_light.pos.w;
			float3 light_color = point_light.color.xyz;

			light_pos = mul(uniforms.light_rot, float4(light_pos, 1.0)).xyz;
			float3 distance = light_pos - world_pos;
			float3 L = normalize(distance);
			float NdotL = max(dot(N, L), 0.0);
#ifdef PBR
			float3 Fr = float3(0.0);

			float3 H = normalize(L + V);

			float NdotH = max(dot(N, H), 0.0);

			// some intermediate values taken from dir light as they are unchanged
			float D = D_GGX(NdotH, roughness);
			float G = V_SmithGGXCorrelated(NdotV, NdotL, roughness);
			Fr = D * G * F;

			float attenuation = get_square_falloff_attenuation(distance, light_radius);
			color += (Fd + Fr) * light_color * attenuation * NdotL;
#else
			float attenuation = get_square_falloff_attenuation(distance, light_radius);
			color += (albedo.xyz * light_color * attenuation * NdotL);
#endif
		}
	}

	if (depth == 0.0)
	{
		float2 ndc = in_uv * 2.0 - 1.0;
		ndc.y *= -1.0;
		float3 sample_dir = mul(uniforms.inverse_view_proj, float4(ndc, 0.0, 1.0)).xyz;
		color = textures_cube[uint(uniforms.textures[0])].SampleLevel(samplers[CUBE_SAMPLER], sample_dir, 0.0).xyz;
	}

	if (pc.resolve_transparent == 1)
	{
       	uint index = uint(frag_coord.x) + uint(frag_coord.y) * uint(pc.screen_size.x);
        color = composite_transparent(pc.oit_buffer, index, color, uint2(frag_coord));
	}

	out_color = float4(color, 1.0);

    bool valid = depth != 0.0;

	if (pc.debug != 0)
    {
        switch (pc.debug)
        {
        case DEBUG_SHADING_NORMALS:
            N = N * 0.5 + 0.5;
            out_color = float4(N, 1.0);
            break;
        case DEBUG_TANGENTS:
        case DEBUG_TANGENT_SIGN:
        case DEBUG_UV:
            valid = false;
            break;
        case DEBUG_SHADOWMAP:
            float2 uv = in_uv;

            if (uv.x < 0.5 && uv.y < 0.5)
            {
                uint cascade = 0;
                cascade = uv.x < 0.25 ? cascade : cascade + 1;
                cascade = uv.y < 0.25 ? cascade : cascade + 2;

                float2 shadow_uv = uv * 4.0;
                shadow_uv = fmod(shadow_uv, float2(1.0));

                out_color.xyz = textures[pc.shadowmap_id + cascade].SampleLevel(samplers[NEAREST_SAMPLER], shadow_uv, 0.0).rrr;
                valid = true; // overwrite skybox
            }
            break;
        case DEBUG_CASCADES:
            switch (cascade_index)
            {
            case 0:
                out_color.xyz *= float3(1, 0, 0);
                break;
            case 1:
                out_color.xyz *= float3(0, 1, 0);
                break;
            case 2:
                out_color.xyz *= float3(0, 0, 1);
                break;
            case 3:
                out_color.xyz *= float3(1, 1, 0);
                break;
            }
            break;
        case DEBUG_MESHLETS:
            valid = false;
            break;
        default:
            break;
        }

        if (!valid)
            out_color = float4(0,0,0,1);
    }

    // TODO: hardcoded draw id to 0
    images_rgba32f[0].Store(int2(frag_coord), out_color);
}
