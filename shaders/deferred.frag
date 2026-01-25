#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

#include "scene.glsl"
#include "samplers.glsl"

layout(set = 1, binding = 0) uniform texture2D allTextures[];
layout(set = 2, binding = 0) uniform sampler samplers[];

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

struct PointLight
{
	vec4 pos;
	vec4 color;
};

layout(buffer_reference, std430) readonly buffer LightBuffer
{
	PointLight lights[];
};

layout( push_constant ) uniform constants
{
	LightBuffer lightBuffer;
	uint albedo_id;    // gbuffer ids
	uint normal_id;    // gbuffer ids
	uint world_pos_id; // gbuffer ids
} pc;

int MAX_LIGHTS = 1000;

void main()
{
	// TODO: nearest or linear?
	//vec3 normal = texture(sampler2D(allTextures[pc.normal_id], samplers[LINEAR_SAMPLER]), inUV).xyz;
	
	vec3 albedo = texture(sampler2D(allTextures[pc.albedo_id], samplers[LINEAR_SAMPLER]), inUV).xyz;
	vec3 worldPos = texture(sampler2D(allTextures[pc.world_pos_id], samplers[LINEAR_SAMPLER]), inUV).xyz;
	
	vec3 color = vec3(0.);
	
	// TODO: amend hard coded max light count
	for (int i = 0; i < MAX_LIGHTS; i++)
	{
		PointLight light = pc.lightBuffer.lights[i];
		vec3 lightCenter = light.pos.xyz;
		float lightRadius = light.pos.w;
		vec3 lightColor = light.color.xyz;
	
		float distance = distance(worldPos, lightCenter);
		
		if (distance < lightRadius)
		{
			color += albedo * lightColor;
		}
	}
	
	vec3 ambient = 0.05 * albedo;
	
	outFragColor = vec4(color + ambient, 1.0);
}