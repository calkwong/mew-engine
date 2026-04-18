import bindings;
import types;
import barycentrics;
import pbr;
import sh;
import util;

struct PushConstants
{
	float4 cluster_size; // xyz is cluster data struct dim, w is single cluster dim where width==height
	float2 screen_size;
	PointLight* light_buffer;
	uint* light_index_buffer;
	LightGrid* light_grid_buffer;
	OITData* oit_buffer;
	uint* meshlet_indices_buffer;
	Meshlet* meshlet_buffer;
	Vertex* vertex_buffer;
	ObjectData* object_buffer;
	MaterialData* material_buffer;
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
	uint pcf;
	float max_prefiltered_lod;
	float pc_metallic; // unused, for furnace test
	float pc_roughness; // unused, for furnace test
	uint debug;
	uint map;
};

[[vk::push_constant]] PushConstants pc;

// formula is for infinite far plane, reverse-z
// returns positive z, negate depending on usage
float linearize_depth(float near, float depth)
{
	return near / depth;
}

static const int CASCADE_COUNT = 4;
static const int MLAB_NODES = 4;

float3 composite_transparent(float3 input_color, uint2 screen_pos)
{
	float3 color = input_color;

		uint index = screen_pos.x + screen_pos.y * uint(pc.screen_size.x);
		OITData frags = pc.oit_buffer[index];

		// early return if nothing stored
		if (frags.transmissions[0] == 1.0)
		{
			return color;
		}

		pc.oit_buffer[index].transmissions = float4(1.0); // reset so we can skip vkcmdfillbuffer

		float3 composite = float3(0.);
		float t_accum = 1.0;
		for (int i = 0; i < MLAB_NODES; i++)
		{
			float t = frags.transmissions[i];
			composite = t != 1.0 ? float3(unpackUnorm4x8ToFloat(frags.colors[i]).xyz) * t_accum + composite : composite;
			t_accum *= t;
		}

		color *= t_accum;
		color += composite;

	return color;
}

float calculate_shadow(float3 world_pos, inout uint cascade_index)
{
	float d = mul(uniforms.view, float4(world_pos, 1.0)).z; // view space z

    for (uint i = 0; i < CASCADE_COUNT; i++)
    {
        if (d > uniforms.cascade_splits[i])
        {
            cascade_index = i;
            break;
        }
    }

    float3 ndc;
    float current_depth;
    if (pc.map == 1)
    {
        cascade_index = 0;

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
    }
    else // interval-based selection
    {
        ndc = mul(uniforms.shadow_transforms[cascade_index], float4(world_pos, 1.0)).xyz;
    }

	current_depth = ndc.z;

	if (current_depth < 0.0) // necessary
        return 1.0;

	float2 uv = ndc.xy * 0.5 + 0.5;
	uv.y = 1.0 - uv.y;

	float layers;
	float2 offset;
	textures[pc.shadowmap_id + cascade_index].GetDimensions(0, offset.x, offset.y, layers);
	offset = float2(1.0) / offset;

	float shadow = 0.0;
	float closest_depth = 0.0;

    // TODO: bias
	if (pc.pcf == 1)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				float2 sample_uv = float2(uv.x + x * offset.x, uv.y + y * offset.y);
				closest_depth = textures[pc.shadowmap_id + cascade_index].SampleLevel(samplers[SHADOW_SAMPLER], sample_uv, 0.0).r;

				if (closest_depth > current_depth)
					shadow += 0.0;
				else
					shadow += 1.0;
			}
		}

		shadow /= 9.0;
	}

	return shadow;
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
    float2 screen_pos = float2(global_id.x, global_id.y);

    if (screen_pos.x >= pc.screen_size.x || screen_pos.y >= pc.screen_size.y)
        return;

    float4 out_color;
    float2 in_uv = (screen_pos + float2(0.5)) / pc.screen_size;

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

	float3 color = float3(0.0);
	float3 ambient = albedo.xyz * 0.1; // when IBL is off, likely going to look physically incorrect

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

	float3 Fr = float3(0.0);

	float3 L = normalize(uniforms.sunlight_dir.xyz);
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
		// perceptual_roughness = pc_roughness; // sphere test
		// metallic = pc_metallic; // sphere test

		float3 R = reflect(-V, N);
		float prefiltered_mip = pc.max_prefiltered_lod * perceptual_roughness;

		{
			float3 irradiance = evaluate_sh(pc.sh_buffer->r_coefficients, pc.sh_buffer->g_coefficients, pc.sh_buffer->b_coefficients, N);
			float3 prefiltered = textures_cube[uint(uniforms.textures[2])].SampleLevel(samplers[CUBE_SAMPLER], R, prefiltered_mip).xyz;
			float2 brdf = textures[uint(uniforms.textures[3])].SampleLevel(samplers[LINEAR_CLAMP_SAMPLER], float2(NdotV, perceptual_roughness), 0.0).rg;

			// float3 white = float3(1.0); // sphere test
			// f0 = float3(0.04); // sphere test
			f0 = lerp(f0, albedo.xyz, metallic);
			// f0 = lerp(f0, white, metallic); // sphere test

			F = F_SchlickRoughness(NdotV, f0, perceptual_roughness);
			kS = F;
			kD = float3(1.0) - kS;
			kD *= 1.0 - metallic;

			// TODO: multiscattering?

		    // TODO: we could bake the division by PI into SH
			float3 diffuse = kD * irradiance * (1.0 / PI) * albedo.xyz;
			float3 specular = prefiltered * (F * brdf.x + brdf.y);
			ambient = diffuse + specular;
			// out_color = float4(ambient, 1.0); // sphere test
			// imageStore(images_rgba32f[0], int2(screen_pos), out_color); // sphere test
			// return; // sphere test
		}
	#endif
#else
	color += albedo.xyz;
#endif

	uint cascade_index = 0;
	if (pc.shadows == 1)
	{
		float occluded = calculate_shadow(world_pos, cascade_index);
        color *= occluded;
	}

	color += ambient;

	float depth = textures[pc.depth_id].SampleLevel(samplers[NEAREST_SAMPLER], in_uv, 0.0).r;

	// TODO: potential issue where faces away from point light being lit, exposed by sponza test scene
	if (pc.light_culling == 1)
	{
		int4 cluster_dim = int4(pc.cluster_size);
		int2 cluster_xy = int2(floor(float2(screen_pos.x, pc.screen_size.y - screen_pos.y) / cluster_dim.w));
		cluster_xy.x = clamp(cluster_xy.x, 0, cluster_dim.x - 1);
		cluster_xy.y = clamp(cluster_xy.y, 0, cluster_dim.y - 1);

		float viewZ = linearize_depth(pc.near, depth); // alternative: calculate via matrix multiply, likely more expensive?

		// equation (3): https://www.aortiz.me/2018/12/21/CG.html#part-2
		// slide 5: https://advances.realtimerendering.com/s2016/Siggraph2016_idTech6.pdf
		uint slice = uint(floor(log(viewZ) * pc.scale - pc.bias));
		slice = clamp(slice, 0, cluster_dim.z - 1);

		uint cluster_index = cluster_xy.x + cluster_dim.x * cluster_xy.y + slice * cluster_dim.x * cluster_dim.y;

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
			color += albedo.xyz * light_color * attenuation * NdotL;
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
		color = composite_transparent(color, uint2(screen_pos));
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
			T.xyz = T.xyz * 0.5 + 0.5;
			out_color = float4(T.xyz, 1.0);
			break;
		case DEBUG_TANGENT_SIGN:
		    float3 w = T.w > 0.0 ? float3(1.0) : float3(0.0);
		    out_color = float4(w, 1.0);
            break;
		case DEBUG_UV:
			out_color = float4(debug_uv, 1.0);
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
		    uint mhash = hash(meshlet_id);
			// uint mhash = hash(material_id);
            out_color.xyz = float3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0;
		    break;
		default:
		    break;
		}

		if (!valid)
		    out_color = float4(0,0,0,1);
	}

    // TODO: hardcoded draw id to 0
	images_rgba32f[0].Store(int2(screen_pos), out_color);
}
