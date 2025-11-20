#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "scene.glsl"

layout (location = 0) in vec2 inUV;

layout(set = 1, binding = 0) uniform texture2D allTextures[];
layout(set = 2, binding = 0) uniform sampler samplers[];

layout (location = 0) out vec4 outFragColor;

layout( push_constant ) uniform constants
{
	mat4 inverseViewProj; // unused
	uint texture_id;
} pc;

float near = 0.01; 
float far  = 100.0; 
  
float LinearizeDepth(float depth) 
{
    float z = depth * 2.0 - 1.0; // back to NDC 
    return (2.0 * near * far) / (far + near - z * (far - near));	
}

void main()
{
	float depth = texture(sampler2D(allTextures[pc.texture_id], samplers[3]), inUV).r; // tonemap req

	depth = LinearizeDepth(1.0 - depth);

	outFragColor = vec4(vec3(depth / far), 1);
}