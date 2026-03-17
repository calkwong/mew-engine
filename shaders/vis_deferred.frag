#version 450

#extension GL_GOOGLE_include_directive : require

#define OIT_RESOLVE

#include "bindings.glsl"
#include "buffer_references.glsl"
#include "vbuffer.glsl"
#include "pbr.glsl"
#include "sh.glsl"
#include "util.glsl"

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
	MeshletIndicesBuffer meshlet_indices_buffer;
	MeshletBuffer meshlet_buffer;
	VertexBuffer vertex_buffer;
	ObjectBuffer object_buffer;
	MaterialBuffer material_buffer;
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
	float max_prefiltered_log;
	float metallic; // unused, for debugging
	float roughness; // unused, for debugging
	uint debug;
};

// formula is for infinite far plane, reverse-z
// returns positive z, negate depending on usage
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
	vec3 light_frag_pos = vec3(uniforms.shadow_transforms[cascade_index] * vec4(world_pos, 1.0)); // ortho, no division by w needed
	
	float current_depth = light_frag_pos.z;
	
	if (current_depth < 0.0) // this is possible
		return 1.0;
	
	vec2 uv = vec2(light_frag_pos.x, light_frag_pos.y);
	uv = uv * 0.5 + 0.5;
	uv.y = 1.0 - uv.y;
	
	vec2 offset = 1.0 / textureSize(sampler2D(textures[shadowmap_id + cascade_index], samplers[NEAREST_SAMPLER]), 0);
	
	float shadow = 0.0;
	float closest_depth = 0.0;

	if (pcf == 1)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				vec2 sample_uv = vec2(uv.x + x * offset.x, uv.y + y * offset.y);
				closest_depth = texture(sampler2D(textures[shadowmap_id + cascade_index], samplers[NEAREST_SAMPLER]), sample_uv).r;
				
				if (closest_depth > current_depth)
					shadow += 0.0;
				else
					shadow += 1.0;
			}
		}
		
		shadow /= 9.0;
	}
	else
	{
		closest_depth = texture(sampler2D(textures[shadowmap_id + cascade_index], samplers[LINEAR_SAMPLER]), uv).r;
		
		float bias = 0.0;
		if (closest_depth > current_depth + bias)
			shadow = 0.0;
		else
			shadow = 1.0;
	}
	return shadow;
}

#define PBR
#define GI

void main()
{
	//// HANDLE LATER, NOT SUPPORTED ON VISIBILITY PATH
	//if (debug_meshlets == 1)
	//{
	//	vec3 normal = texture(sampler2D(textures[normal_id], samplers[NEAREST_SAMPLER]), in_uv).xyz;
	//	out_color = vec4(normal, 1.0);
	//	return; 
	//}
	
	uvec2 data = texture(usampler2D(textures_u[gbuffer_id], samplers[NEAREST_SAMPLER]), in_uv).rg; 
	
	uint draw_id = data.x; 
	uint packed_id = data.y;
	uint triangle_id = bitfieldExtract(packed_id, 25, 7);
	uint meshlet_id = bitfieldExtract(packed_id, 0, 25);
	
	Meshlet meshlet = meshlet_buffer.meshlets[meshlet_id];
	uint triangle_offset = meshlet.data_offset + meshlet.vertex_count;
	
	uint idx0 = meshlet_indices_buffer.indices[triangle_offset + triangle_id * 3 + 0];
	uint idx1 = meshlet_indices_buffer.indices[triangle_offset + triangle_id * 3 + 1];
	uint idx2 = meshlet_indices_buffer.indices[triangle_offset + triangle_id * 3 + 2];
	
	uint vertex_index0 = meshlet_indices_buffer.indices[meshlet.data_offset + idx0];
	uint vertex_index1 = meshlet_indices_buffer.indices[meshlet.data_offset + idx1];
	uint vertex_index2 = meshlet_indices_buffer.indices[meshlet.data_offset + idx2];
	
	Vertex v0 = vertex_buffer.vertices[vertex_index0];
	Vertex v1 = vertex_buffer.vertices[vertex_index1];
	Vertex v2 = vertex_buffer.vertices[vertex_index2];

	ObjectData obj = object_buffer.objects[draw_id];

	vec3 wp0 = rotate_quat(vec3(v0.px, v0.py, v0.pz), obj.orientation) * obj.scale + obj.translation;
	vec3 wp1 = rotate_quat(vec3(v1.px, v1.py, v1.pz), obj.orientation) * obj.scale + obj.translation;
	vec3 wp2 = rotate_quat(vec3(v2.px, v2.py, v2.pz), obj.orientation) * obj.scale + obj.translation;

	vec4 p0 = uniforms.view_proj * vec4(wp0, 1.0);
	vec4 p1 = uniforms.view_proj * vec4(wp1, 1.0);
	vec4 p2 = uniforms.view_proj * vec4(wp2, 1.0);
	
	// vulkan top left origin, hence flipping y is necessary
	vec2 pNdc = gl_FragCoord.xy / screen_size; 
	pNdc = pNdc * 2.0 - 1.0;
	pNdc.y = -pNdc.y; 
	
	BarycentricDeriv bary = calculate_barycentric(p0, p1, p2, pNdc, screen_size);
	vec2 uv;
	vec2 uv_ddx;
	vec2 uv_ddy;
	interpolate_with_deriv(bary, vec2(v0.uv_x, v0.uv_y), vec2(v1.uv_x, v1.uv_y), vec2(v2.uv_x, v2.uv_y), uv, uv_ddx, uv_ddy);
	
	vec3 debug_uv = vec3(uv, 0.0);
	
	vec3 world_pos = interpolate(bary, wp0, wp1, wp2);
	
	vec3 np0, np1, np2;
	vec4 tp0, tp1, tp2;
	unpack_tbn(v0.normal, uint(v0.tangent), np0, tp0);
	unpack_tbn(v1.normal, uint(v1.tangent), np1, tp1);
	unpack_tbn(v2.normal, uint(v2.tangent), np2, tp2);
	
	vec3 V = normalize(uniforms.camera_pos.xyz - world_pos);
	
	// alternative: store triangle face bit in visibility buffer
	vec3 e1 = wp1 - wp0;
	vec3 e2 = wp2 - wp0;
	vec3 gN = cross(e1, e2);
	bool is_back_face = dot(gN, V) < 0.0;
	
	vec3 n0 = rotate_quat(np0, obj.orientation);
	vec3 n1 = rotate_quat(np1, obj.orientation);
	vec3 n2 = rotate_quat(np2, obj.orientation);
	
	vec3 N = normalize(interpolate(bary, n0, n1, n2)); // TODO: mikktspace convention is NOT to normalize. we normalize here as khronos sponza breaks iirc?
	
	N = is_back_face ? -N : N;
	
	uint material_id = object_buffer.objects[draw_id].material_id;
	MaterialData m = material_buffer.materials[material_id];
	
	vec4 albedo = m.base_color_factor;
	
	if (m.diffuse_id != 0)
	{
		albedo.xyz *= textureGrad(sampler2D(textures[m.diffuse_id], samplers[LINEAR_SAMPLER]), uv, uv_ddx, uv_ddy).xyz;
	}
	
	vec3 color = vec3(0.0);
	vec3 ambient = albedo.xyz * 0.1; // when GI is off, likely going to look physically incorrect
	
	vec4 t0 = vec4(rotate_quat(tp0.xyz, obj.orientation), tp0.w);
	vec4 t1 = vec4(rotate_quat(tp1.xyz, obj.orientation), tp1.w);
	vec4 t2 = vec4(rotate_quat(tp2.xyz, obj.orientation), tp2.w);
	
	vec4 T = interpolate(bary, t0, t1, t2);
	T.xyz = normalize(T.xyz); // TODO: mikktspace convention is NOT to normalize. we normalize here as khronos sponza breaks iirc?
	float sign = T.w; // sign is flipped during tangent generation so mikktspace is consistent with glTF handedness
		
	sign = is_back_face ? -sign : sign;	
		
#ifdef PBR
	if (m.normal_id != 0)
	{
		vec3 B = sign * cross(N, T.xyz);
		vec3 shading_normal = textureGrad(sampler2D(textures[m.normal_id], samplers[LINEAR_SAMPLER]), uv, uv_ddx, uv_ddy).xyz;
		shading_normal = shading_normal * 2.0 - 1.0;
		N = normalize(shading_normal.x * T.xyz + shading_normal.y * B + shading_normal.z * N);
	}
	
	float metallic = m.metallic_factor;
	float perceptual_roughness = m.roughness_factor;
	vec2 metal_roughness = vec2(0.0);
	if (m.metalroughness_id != 0)
	{
		metal_roughness = textureGrad(sampler2D(textures[m.metalroughness_id], samplers[LINEAR_SAMPLER]), uv, uv_ddx, uv_ddy).bg;
		metallic *= metal_roughness.x;
		perceptual_roughness *= metal_roughness.y;
	}

	perceptual_roughness = max(perceptual_roughness, 0.045); // frostbite engine clamp value for analytical lights (fp32)
	float roughness = perceptual_roughness * perceptual_roughness;
	
	vec3 Fr = vec3(0.0);
	
	vec3 L = normalize(uniforms.sunlight_dir.xyz);
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
		float prefiltered_mip = max_prefiltered_log * perceptual_roughness;
		
		{
			vec3 irradiance = evaluate_sh(sh_buffer.r_coefficients, sh_buffer.g_coefficients, sh_buffer.b_coefficients, N);
			vec3 prefiltered = textureLod(samplerCube(textures_cube[uint(uniforms.textures[2])], samplers[CUBE_SAMPLER]), R, prefiltered_mip).xyz;
			vec2 brdf = texture(sampler2D(textures[uint(uniforms.textures[3])], samplers[LINEAR_CLAMP_SAMPLER]), vec2(NdotV, perceptual_roughness)).rg;
			
			//vec3 white = vec3(1.0); // sphere test
			//f0 = vec3(0.04); // sphere test
			f0 = mix(f0, albedo.xyz, metallic);
			//f0 = mix(f0, white, metallic); // sphere test

			// try normal schlick and visualize difference, remove comment after
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
	color += vec4(albedo);
#endif
	
	uint cascade_index = 0;
	if (shadows == 1)
	{
		float occluded = calculate_shadow(world_pos, cascade_index);
		
		color *= occluded;
		
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
	}
	
	color += ambient;
	
	float depth = texture(sampler2D(textures[depth_id], samplers[NEAREST_SAMPLER]), in_uv).r;
	
	if (light_culling == 1)
	{
		vec2 screen_pos = vec2(gl_FragCoord.xy);
		screen_pos.y = screen_size.y - screen_pos.y;
		
		ivec4 cluster_dim = ivec4(cluster_size);
		ivec2 cluster_xy = ivec2(floor(screen_pos.xy / cluster_dim.w));
		cluster_xy.x = clamp(cluster_xy.x, 0, cluster_dim.x - 1);
		cluster_xy.y = clamp(cluster_xy.y, 0, cluster_dim.y - 1);
		
		float viewZ = linearize_depth(near, depth); // alternative: calculate via matrix multiply, likely more expensive?
		
		// equation (3): https://www.aortiz.me/2018/12/21/CG.html#part-2 
		// slide 5: https://advances.realtimerendering.com/s2016/Siggraph2016_idTech6.pdf
		uint slice = uint(floor(log(viewZ) * scale - bias));
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
			color += albedo.xyz * light_color * attenuation * NdotL;
#endif
		}
	}
	
	// TODO: refactor and use depth/stencil buffer to reject pixels in future
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
	
	if (debug != 0)
	{
		switch (debug)
		{
		case 1: // geometry normals
			N = N * 0.5 + 0.5;
			N = srgb_to_linear(N);
			out_color = vec4(N, 1.0);
			break;
		case 2: // tangents
			T.xyz = T.xyz * 0.5 + 0.5;
			T.xyz = srgb_to_linear(T.xyz);
			out_color = vec4(T.xyz, 1.0);
			break;
		case 3: // tangent w
		    vec3 w = T.w > 0.0 ? vec3(1.0) : vec3(0.0);
		    w = srgb_to_linear(w);
		    out_color = vec4(w, 1.0);
            break;
		case 4: // uv
			debug_uv = srgb_to_linear(debug_uv);
			out_color = vec4(debug_uv, 1.0);
			break;
		}
		
		out_color = depth != 0.0 ? out_color : vec4(0,0,0,1);
	}
}