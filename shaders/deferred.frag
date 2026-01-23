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

layout( push_constant ) uniform constants
{
	uint albedo_id; // gbuffer ids
	uint normal_id; // gbuffer ids
} pc;

void main()
{
	// TODO: nearest or linear?
	vec3 normal = texture(sampler2D(allTextures[pc.normal_id], samplers[LINEAR_SAMPLER]), inUV).xyz;
	
	outFragColor = vec4(normal, 1.0);
}