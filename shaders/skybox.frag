// TODO: REFACTOR, PLENTY OF COMMITS BEHIND

#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "scene.glsl"
#include "samplers.glsl"

layout (location = 0) in vec3 inUVW;

layout(set = 1, binding = 0) uniform textureCube allTextures[];
layout(set = 2, binding = 0) uniform sampler samplers[];

layout (location = 0) out vec4 outFragColor;

layout( push_constant ) uniform constants
{
	mat4 inverseViewProj; // no translation in view matrix
	uint texture_id;
} pc;

void main()
{
	vec4 color = texture(samplerCube(allTextures[pc.texture_id], samplers[CUBE_SAMPLER]), inUVW); // tonemap req
	outFragColor = color;
}