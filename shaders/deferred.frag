#version 450

#extension GL_GOOGLE_include_directive : require

#define OIT_RESOLVE

#include "bindings.glsl"
#include "buffer_references.glsl"
#include "pbr.glsl"
#include "sh.glsl"

layout (location = 0) in vec2 in_uv;

layout (location = 0) out vec4 out_color;

layout( push_constant ) uniform constants
{
	vec4 cluster_size; // xyz is cluster data struct dim, w is single cluster dim where width==height
	vec2 screen_size;
	LightBuffer light_buffer;
	LightIndexBuffer light_index_buffer;
	LightGridBuffer light_grid_buffer;
	OITBuffer oit_buffer;
	uint padding[10]; // padding for visibility buffer variant
	SHBuffer sh_buffer;
	uint depth_id;
	uint gbuffer_id;  
	uint shadowmap_id;
	uint light_culling;
	float near;
	float scale;
	float bias;
	uint debug_meshlets;
	uint resolve_transparent;
	uint shadows;
	uint pcf;
	uint debug_shadowmap;
	uint debug_cascades;
	// gi
	float max_prefiltered_lod;
	float metallic; // unused, for debugging
	float roughness; // unused, for debugging
	uint debug; // unused, only in vis_deferred
	uint map;
};

// IMPORTANT! formula is for infinite far plane, reverse-z
// returns positive value, may need to negate depending on what we're using it for
float linearize_depth(float near, float depth)
{
	return near / depth;
}

const int CASCADE_COUNT = 4;
const int MLAB_NODES = 4;

vec3 composite_transparent(vec3 input_color)
{
	vec3 color = input_color;
	
		uvec2 screen_coords = uvec2(floor(gl_FragCoord.xy));
		uint index = screen_coords.x + screen_coords.y * uint(screen_size.x);
		OITData frags = oit_buffer.frags[index];
		
		// early return if nothing stored
		if (frags.transmissions[0] == 1.0)
		{
			return color;
		}
		
		oit_buffer.frags[index].transmissions = vec4(1.0); // reset so we can skip vkcmdfillbuffer
		
		vec3 composite = vec3(0.);
		float t_accum = 1.0;
		for (int i = 0; i < MLAB_NODES; i++)
		{
			float t = frags.transmissions[i];
			composite = t != 1.0 ? unpackUnorm4x8(frags.colors[i]).xyz * t_accum + composite : composite;
			t_accum *= t;
		}
		
		color *= t_accum;
		color += composite;
	
	return color;
}

float calculate_shadow(vec3 world_pos, inout uint cascade_index)
{
	float d = (uniforms.view * vec4(world_pos, 1.0)).z; // view space z

    for (uint i = 0; i < CASCADE_COUNT; i++)
    {
        if (d > uniforms.cascade_splits[i])
        {
            cascade_index = i;
            break;
        }
    }

    vec3 ndc;
    float current_depth;
    if (map == 1)
    {
        cascade_index = 0;

        for (int i = 0; i < CASCADE_COUNT; i++)
        {
            ndc = vec3(uniforms.shadow_transforms[i] * vec4(world_pos, 1.0));
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
        ndc = vec3(uniforms.shadow_transforms[cascade_index] * vec4(world_pos, 1.0));
    }

	current_depth = ndc.z;

	if (current_depth < 0.0) // necessary
        return 1.0;

	vec2 uv = ndc.xy * 0.5 + 0.5;
	uv.y = 1.0 - uv.y;

	vec2 offset = 1.0 / textureSize(sampler2D(textures[shadowmap_id + cascade_index], samplers[SHADOW_SAMPLER]), 0);

	float shadow = 0.0;
	float closest_depth = 0.0;

    // TODO: bias
	if (pcf == 1)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				vec2 sample_uv = vec2(uv.x + x * offset.x, uv.y + y * offset.y);
				closest_depth = texture(sampler2D(textures[shadowmap_id + cascade_index], samplers[SHADOW_SAMPLER]), sample_uv).r;

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

vec3 reconstruct_world_pos(float depth, mat4 inverse_view_proj)
{
	vec2 ndc = gl_FragCoord.xy / screen_size;
	ndc = ndc * 2.0 - 1.0;
	ndc.y *= -1.0; // flip as window coords are top down
	vec4 world_pos = inverse_view_proj * vec4(ndc, depth, 1.0);
	
	return world_pos.xyz / world_pos.w;
}

#define PBR
#define GI

void main()
{
	uint albedo_id = gbuffer_id;
	uint normal_id = gbuffer_id + 1;
	uint metalroughness_id = gbuffer_id + 2;

	vec3 N = texture(sampler2D(textures[normal_id], samplers[NEAREST_SAMPLER]), in_uv).xyz;
	N = normalize(N); // necessary to remove banding, RGB32 does not need this

	if (debug_meshlets == 1)
	{
		out_color = vec4(N, 1.0);
		return;
	}
	
	vec4 albedo = vec4(texture(sampler2D(textures[albedo_id], samplers[NEAREST_SAMPLER]), in_uv).xyz, 1.0);
	float depth = texture(sampler2D(textures[depth_id], samplers[NEAREST_SAMPLER]), in_uv).r;
	vec3 world_pos = reconstruct_world_pos(depth, uniforms.inverse_view_proj);
	
	vec3 color = vec3(0.0);
	vec3 ambient = albedo.xyz * 0.1; // when GI is off, likely going to look physically incorrect
	
#ifdef PBR
	vec2 metal_roughness = texture(sampler2D(textures[metalroughness_id], samplers[NEAREST_SAMPLER]), in_uv).xy;
	float metallic = metal_roughness.x;
	float perceptual_roughness = metal_roughness.y;
	float roughness = perceptual_roughness * perceptual_roughness;
	
	vec3 Fr = vec3(0.0);
	
	vec3 L = normalize(uniforms.sunlight_dir.xyz);
	vec3 V = normalize(uniforms.camera_pos.xyz - world_pos);
	vec3 H = normalize(L + V);
	
	float NdotL = max(dot(N, L), 0.0);
	float NdotH = max(dot(N, H), 0.0);
	float NdotV = max(dot(N, V), 0.001);
	float VdotH = max(dot(V, H), 0.0);
	
	vec3 f0 = vec3(0.04);
	f0 = mix(f0, albedo.xyz, metallic);
	
	vec3 F = F_Schlick(VdotH, f0);
		
	vec3 kS = F;
	vec3 kD = vec3(1.0) - kS;
	
	kD *= 1.0 - metallic;
	vec3 Fd = kD * albedo.xyz / PI;
	
	float D = D_GGX(NdotH, roughness);
	float G = V_SmithGGXCorrelated(NdotV, NdotL, roughness);
	Fr = D * G * F;
	
	vec3 light_color = uniforms.sunlight_color.xyz;
	color = (Fd + Fr) * light_color * NdotL; 
	
	#ifdef GI
		//perceptual_roughness = roughness; // sphere test
		//metallic = metallic; // sphere test
		
		vec3 R = reflect(-V, N);
		float prefiltered_mip = max_prefiltered_lod * perceptual_roughness;
		
		{
			vec3 irradiance = evaluate_sh(sh_buffer.r_coefficients, sh_buffer.g_coefficients, sh_buffer.b_coefficients, N);
			vec3 prefiltered = textureLod(samplerCube(textures_cube[uint(uniforms.textures[2])], samplers[CUBE_SAMPLER]), R, prefiltered_mip).xyz;
			vec2 brdf = texture(sampler2D(textures[uint(uniforms.textures[3])], samplers[LINEAR_CLAMP_SAMPLER]), vec2(NdotV, perceptual_roughness)).rg;
			
			//vec3 white = vec3(1.0); // sphere test
			//f0 = vec3(0.04); // sphere test
			f0 = mix(f0, albedo.xyz, metallic);
			//f0 = mix(f0, white, metallic); // sphere test
			
			F = F_SchlickRoughness(NdotV, f0, perceptual_roughness);
			kS = F;
			kD = vec3(1.0) - kS;
			kD *= 1.0 - metallic;
			
		    // TODO: we could bake the division by PI into SH
			vec3 diffuse = kD * irradiance * (1.0 / PI) * albedo.xyz;
			vec3 specular = prefiltered * (F * brdf.x + brdf.y);
			ambient = diffuse + specular;
			//out_color = vec4(ambient, 1.0); // sphere test
			//return; // sphere test
		}
	#endif
#else	
	color += albedo.xyz;
#endif
	
	uint cascade_index = 0;
	if (shadows == 1)
	{
		float occluded = calculate_shadow(world_pos, cascade_index);
		
		if (debug_cascades == 1)
		{
			switch (cascade_index)
			{
				case 0:
					color *= vec3(1, 0, 0);
					break;
				case 1:
					color *= vec3(0, 1, 0);
					break;
				case 2:
					color *= vec3(0, 0, 1);
					break;
				case 3:
					color *= vec3(1, 1, 0);
					break;
			}
		}
		else
		    color *= occluded;
	}
	
	color += ambient;
	
	if (light_culling == 1)
	{
		vec2 screen_pos = vec2(gl_FragCoord.xy);
		screen_pos.y = screen_size.y - screen_pos.y;
		
		ivec4 cluster_dim = ivec4(cluster_size);
		ivec2 cluster_xy = ivec2(floor(screen_pos.xy / cluster_dim.w));
		cluster_xy.x = clamp(cluster_xy.x, 0, cluster_dim.x - 1);
		cluster_xy.y = clamp(cluster_xy.y, 0, cluster_dim.y - 1);
		
		float viewz = linearize_depth(near, depth); // implicitly flipped
		
		// equation (3): https://www.aortiz.me/2018/12/21/CG.html#part-2 
		// slide 5: https://advances.realtimerendering.com/s2016/Siggraph2016_idTech6.pdf
		uint slice = uint(floor(log(viewz) * scale - bias));
		slice = clamp(slice, 0, cluster_dim.z - 1);
		
		uint cluster_index = cluster_xy.x + cluster_dim.x * cluster_xy.y + slice * cluster_dim.x * cluster_dim.y; 
		
		uint offset = light_grid_buffer.grid[cluster_index].offset;
		uint count = light_grid_buffer.grid[cluster_index].count;
		
		for (int i = 0; i < count; i++)
		{
			uint index = light_index_buffer.indices[offset + i];
			vec3 light_pos = light_buffer.lights[index].pos.xyz;
			light_pos = vec3(uniforms.light_rot * vec4(light_pos, 1.0));
			float light_radius = light_buffer.lights[index].pos.w;
			vec3 light_color = light_buffer.lights[index].color.xyz;
			vec3 distance = light_pos - world_pos;
			vec3 L = normalize(distance); 
			float NdotL = max(dot(N, L), 0.0);
#ifdef PBR
			vec3 Fr = vec3(0.0);

			vec3 H = normalize(L + V);
			
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
		vec2 ndc = in_uv * 2.0 - 1.0;
		ndc.y *= -1.0;
		vec3 sample_dir = vec3(uniforms.inverse_view_proj * vec4(ndc, 0.0, 1.0));
		color = texture(samplerCube(textures_cube[uint(uniforms.textures[0])], samplers[CUBE_SAMPLER]), sample_dir).xyz;
	}
	
	if (resolve_transparent == 1)
	{
		color = composite_transparent(color);
	}
	
	out_color = vec4(color, 1.0);
	
	if (debug_shadowmap != 0)
	{
		vec2 uv = gl_FragCoord.xy / screen_size;
		uint idx = debug_shadowmap - 1;
		vec3 depth = vec3(texture(sampler2D(textures[shadowmap_id + idx], samplers[NEAREST_SAMPLER]), uv).r);
		out_color.xyz = depth;
	}
}