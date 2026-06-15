import bindings;
import types;
import barycentrics;
import pbr;
import sh;
import util;
import shadows;
import volumetrics;

struct PushConstants
{
	float4 cluster_size; // xy is cluster dim, z is # of cluster slices
	float2 screen_size;
	PointLight* light_buffer;
	uint* light_index_buffer;
	LightGrid* light_grid_buffer;
	uint* meshlet_indices_buffer;
	Meshlet* meshlet_buffer;
	Vertex* vertex_buffer;
	ObjectData* object_buffer;
	MaterialData* material_buffer;
	uint* index_buffer;
	MeshData* mesh_buffer;
	SH* sh_buffer;
	uint draw_id;
	uint depth_id;
	uint gbuffer_id;
	uint shadowmap_id;
	uint light_culling;
	float near;
	float scale;
	float bias;
	uint shadows;
	uint shadows_rt;
	float max_prefiltered_lod;
	uint debug;
	uint volumetrics;
	uint volumetrics_tex;
	float volumetrics_scale;
	float volumetrics_bias;
	uint3 volumetrics_froxel_dim;
};

[[vk::push_constant]] PushConstants pc;

enum Debug
{
    Placeholder,
    Normal, // shading not geometric
    Albedo,
    Metallic,
    Roughness,
    Tangent,
    TangentSign,
    Uv,
    Shadow,
    ShadowCascades,
    Meshlets
};

#define IBL

static const float PI = 3.14159265359;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
    uint3 global_id : SV_DispatchThreadID,
)
{
    float2 frag_coord = float2(global_id.x, global_id.y) + float2(0.5);

    if (frag_coord.x >= pc.screen_size.x || frag_coord.y >= pc.screen_size.y)
        return;

    SamplerState nearest_samp = get_sampler<SamplerState>(NEAREST_SAMPLER);
    SamplerState linear_samp = get_sampler<SamplerState>(LINEAR_SAMPLER);
    SamplerState cube_samp = get_sampler<SamplerState>(CUBE_SAMPLER);
    SamplerState linear_clamp_samp = get_sampler<SamplerState>(LINEAR_CLAMP_SAMPLER);

    float4 out_color;
    float2 in_uv = frag_coord / pc.screen_size;

	uint2 data = get_sampled_image<Texture2D<uint2>>(pc.gbuffer_id).SampleLevel(nearest_samp, in_uv, 0.0).rg;

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
		albedo.xyz *= get_sampled_image<Texture2D>(m.diffuse_id).SampleGrad(linear_samp, uv, uv_ddx, uv_ddy).xyz;
	}

	float3 color = float3(0.0);

	float4 t0 = float4(rotate_quat(tp0.xyz, obj.orientation), tp0.w);
	float4 t1 = float4(rotate_quat(tp1.xyz, obj.orientation), tp1.w);
	float4 t2 = float4(rotate_quat(tp2.xyz, obj.orientation), tp2.w);

	float4 T = interpolate(bary, t0, t1, t2); // mikktspace
	// T.w is naturally flipped during tangent generation so mikktspace is consistent with glTF handedness
	T.w = is_back_face ? -T.w : T.w;

	if (m.normal_id != 0)
	{
		float3 shading_normal = get_sampled_image<Texture2D>(m.normal_id).SampleGrad(linear_samp, uv, uv_ddx, uv_ddy).xyz;
		shading_normal = shading_normal * 2.0 - 1.0;
		N = tangent_to_world_space(shading_normal, T, N);
	}
	else
	    N = normalize(N);

	float metallic = m.metallic_factor;
	float perceptual_roughness = m.roughness_factor;
	float2 metal_roughness = float2(0.0);
	if (m.metalroughness_id != 0)
	{
		metal_roughness = get_sampled_image<Texture2D>(m.metalroughness_id).SampleGrad(linear_samp, uv, uv_ddx, uv_ddy).bg;
		metallic *= metal_roughness.x;
		perceptual_roughness *= metal_roughness.y;
	}

	perceptual_roughness = max(perceptual_roughness, 0.045); // frostbite engine clamp value for analytical lights (fp32)
	float roughness = perceptual_roughness * perceptual_roughness;

	float3 L = normalize(uniforms.sunlight_dir.xyz);
	float3 H = normalize(L + V);

	float3 light_color = uniforms.sunlight_color.xyz;
	BRDFData brdf_data = prepare_brdf_data(N, L, V, albedo.xyz, roughness, metallic);
	color = eval_brdf(brdf_data, albedo.xyz, roughness, metallic) * light_color * brdf_data.NdotL;

#ifdef IBL
	float3 R = reflect(-V, N);
	float prefiltered_mip = pc.max_prefiltered_lod * perceptual_roughness;

	float3 irradiance = evaluate_sh(pc.sh_buffer->r_coefficients, pc.sh_buffer->g_coefficients, pc.sh_buffer->b_coefficients, N);
	float3 radiance = get_sampled_image<TextureCube>(uniforms.prefiltered_id).SampleLevel(cube_samp, R, prefiltered_mip).xyz;
	float2 brdf_lut = get_sampled_image<Texture2D>(uniforms.brdf_id).SampleLevel(linear_clamp_samp, float2(brdf_data.NdotV, perceptual_roughness), 0.0).rg;

	float3 ambient = eval_indirect_multiscattering(brdf_data, irradiance, radiance, brdf_lut, albedo.xyz, perceptual_roughness, metallic);
	float ao = 1.0;

	if (m.occlusion_id != 0)
	{
		// note: occlusion strength ignored
	    ao = get_sampled_image<Texture2D>(m.occlusion_id).SampleGrad(linear_samp, uv, uv_ddx, uv_ddy).r;
		ambient *= ao;
	}
#else
    float3 ambient = albedo.xyz * 0.1;
#endif

	uint cascade_index = 0;
	if (pc.shadows == 1)
	{
	    if (pc.shadows_rt == 0)
		{
    		float occluded = calculate_shadow(world_pos, cascade_index, pc.shadowmap_id);
            color *= occluded;
		}
	    else
		{
            bool shadowed = in_shadow(world_pos, pc.object_buffer, pc.mesh_buffer, pc.material_buffer, pc.index_buffer, pc.vertex_buffer);
            color = shadowed ? float3(0.0) : color;
		}
	}

	color += ambient;

	float depth = get_sampled_image<Texture2D>(pc.depth_id).SampleLevel(nearest_samp, in_uv, 0.0).r;

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

            update_brdf_data(brdf_data, L);

			float attenuation = get_square_falloff_attenuation(distance, light_radius);
			color += eval_brdf(brdf_data, albedo.xyz, roughness, metallic) * light_color * attenuation * brdf_data.NdotL;
		}
	}

	float3 emissive = m.emissive_factor;
	if (m.emissive_id != 0)
	{
	    emissive *= get_sampled_image<Texture2D>(m.emissive_id).SampleGrad(linear_samp, uv, uv_ddx, uv_ddy).rgb;
	}
	color += emissive;

	if (depth == 0.0)
	{
		float2 ndc = in_uv * 2.0 - 1.0;
		ndc.y *= -1.0;
		float3 sample_dir = mul(uniforms.inverse_view_proj, float4(ndc, 0.0, 1.0)).xyz;
		color = get_sampled_image<TextureCube>(uniforms.skybox_id).SampleLevel(cube_samp, sample_dir, 0.0).xyz;
	}

	if (pc.volumetrics != 0)
	{
        uint3 sample_uvw = depth_to_froxel(depth, pc.near, pc.volumetrics_scale, pc.volumetrics_bias, frag_coord, pc.screen_size, pc.volumetrics_froxel_dim);
        // TODO: dithering
        float3 uvw = float3(sample_uvw) / float3(pc.volumetrics_froxel_dim);
        Texture3D<float4> volumetrics_tex = get_sampled_image<Texture3D<float4>>(pc.volumetrics_tex);
        float4 scattering_transmittance = volumetrics_tex.SampleLevel(linear_clamp_samp, uvw, 0.0);
        color.rgb = color.rgb * scattering_transmittance.a + scattering_transmittance.rgb;
	}

	out_color = float4(color, 1.0);

    bool valid = depth != 0.0;

	if (pc.debug != 0)
	{
		switch (int(pc.debug))
		{
		case Debug::Normal:
			N = N * 0.5 + 0.5;
			out_color = float4(N, 1.0);
			break;
		case Debug::Albedo:
		    out_color = float4(linear_to_srgb(albedo.xyz), 1.0);
		    break;
        case Debug::Metallic:
            out_color = float4(float3(metallic), 1.0);
            break;
        case Debug::Roughness:
            out_color = float4(float3(perceptual_roughness), 1.0);
            break;
		case Debug::Tangent:
			T.xyz = T.xyz * 0.5 + 0.5;
			out_color = float4(T.xyz, 1.0);
			break;
		// TODO: fix - this currently will incorrectly produce flipped T.w for double sided cutouts
		case Debug::TangentSign:
		    float3 w = T.w > 0.0 ? float3(1.0) : float3(0.0);
		    out_color = float4(w, 1.0);
            break;
		case Debug::Uv:
			out_color = float4(debug_uv, 1.0);
			break;
		case Debug::Shadow:
		    float2 uv = in_uv;

		    if (uv.x < 0.5 && uv.y < 0.5)
		    {
		        uint cascade = 0;
                cascade = uv.x < 0.25 ? cascade : cascade + 1;
                cascade = uv.y < 0.25 ? cascade : cascade + 2;

		        float2 shadow_uv = uv * 4.0;
		        shadow_uv = fmod(shadow_uv, float2(1.0));

                out_color.xyz = get_sampled_image<Texture2D>(pc.shadowmap_id + cascade).SampleLevel(nearest_samp, shadow_uv, 0.0).rrr;
                valid = true; // overwrite skybox
		    }
		    break;
		case Debug::ShadowCascades:
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
		case Debug::Meshlets:
		    uint mhash = hash(meshlet_id);
            out_color.xyz = float3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0;
		    break;
		default:
		    break;
		}

		if (!valid)
		    out_color = float4(0,0,0,1);
	}

	RWTexture2D tex = get_storage_image<RWTexture2D>(pc.draw_id);
	tex.Store(int2(frag_coord), out_color);
}
