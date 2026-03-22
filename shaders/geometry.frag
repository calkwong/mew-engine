#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

#include "bindings.glsl"
#include "buffer_references.glsl"

layout (location = 0) in vec3 in_normal;
layout (location = 1) in vec2 in_uv;
layout (location = 2) in vec4 in_tangent;
layout (location = 3) in flat uint in_material_id;
layout (location = 4) in vec4 in_clip_pos;
layout (location = 5) in vec4 in_prev_clip_pos;

layout (location = 0) out vec4 out_albedo;
layout (location = 1) out vec4 out_normal;
layout (location = 2) out vec2 out_metal_roughness;
layout (location = 3) out vec2 out_velocity;

// 1 - OPAQUE
// 0 - MASK
layout (constant_id = 0) const int OPAQUE = 1;

layout( push_constant ) uniform constants
{
	ObjectBuffer object_buffer;
	VertexBuffer vertex_buffer;
	MeshTaskBuffer mesh_task_buffer;
	MeshletBuffer meshlet_buffer;
	MeshletIndicesBuffer meshlet_indices_buffer;
	ClusterIndicesBuffer cluster_indices_buffer; 
	MaterialBuffer material_buffer;
	OITBuffer oit_buffer;
	vec2 jitter_offset;
};

#define PBR

void main() 
{	
	MaterialData m = material_buffer.materials[in_material_id];
	
	vec4 albedo = m.base_color_factor;
	if (m.diffuse_id != 0)
	{
		vec4 sampled_albedo = texture(sampler2D(textures[m.diffuse_id], samplers[LINEAR_SAMPLER]), in_uv);
		
		if (OPAQUE == 0)
		{
			if (sampled_albedo.a < 0.5)
				discard;
		}
		
		albedo *= sampled_albedo;
	}
	
	vec3 N;
	float metallic;
	float roughness;
#ifdef PBR
	N = in_normal; // mikktspace

	if (OPAQUE == 0)
	{
		N = gl_FrontFacing ? N : -N;
	}
	
	if (m.normal_id != 0)
	{
		vec3 T = in_tangent.xyz; // mikktspace
		float sign = in_tangent.w; // sign is flipped during tangent generation so mikktspace is consistent with glTF handedness
		
		if (OPAQUE == 0)
			sign = gl_FrontFacing ? sign : -sign;
		
		vec3 B = sign * cross(N, T);
		
		vec3 shading_normal = texture(sampler2D(textures[m.normal_id], samplers[LINEAR_SAMPLER]), in_uv).xyz;
		shading_normal = shading_normal * 2.0 - 1.0;
		N = normalize(shading_normal.x * T.xyz + shading_normal.y * B + shading_normal.z * N);
	}
	else
	    N = normalize(N);
	
	metallic = m.metallic_factor;
	float perceptual_roughness = m.roughness_factor;
	vec2 metal_roughness = vec2(0.0);
	if (m.metalroughness_id != 0)
	{
		metal_roughness = texture(sampler2D(textures[m.metalroughness_id], samplers[LINEAR_SAMPLER]), in_uv).bg;
		metallic *= metal_roughness.x;
		perceptual_roughness *= metal_roughness.y;
	}
	perceptual_roughness = max(perceptual_roughness, 0.045); // frostbite engine clamp value for analytical lights (fp32)
#else
	N = normalize(in_normal); 
#endif

    vec2 current_ndc = in_clip_pos.xy / in_clip_pos.w;
    vec2 previous_ndc = in_prev_clip_pos.xy / in_prev_clip_pos.w;

    vec2 velocity = current_ndc - previous_ndc;
    velocity = velocity * 0.5 + 0.5;
    velocity.y *= -1.0; // flip for velocity in uv space

    velocity -= jitter_offset;

    out_albedo = albedo;
    out_normal = vec4(N, 1);
    out_metal_roughness = vec2(metallic, perceptual_roughness);
    out_velocity = vec2(velocity);
}