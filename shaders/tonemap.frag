#version 450

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform texture2D allTextures[];
layout(set = 1, binding = 0) uniform sampler samplers[];

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

layout( push_constant ) uniform constants
{
	uint texture_id;
} pc;

const float exposure = 4.0;

vec3 Uncharted2Tonemap(vec3 x)
{
	float A = 0.15;
	float B = 0.50;
	float C = 0.10;
	float D = 0.20;
	float E = 0.02;
	float F = 0.30;
	return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

void main()
{
	vec4 color = texture(sampler2D(allTextures[pc.texture_id], samplers[0]), inUV);
	
	color.xyz = Uncharted2Tonemap(color.xyz * exposure);
	color.xyz = color.xyz * (1.0 / Uncharted2Tonemap(vec3(11.2)));
	
	outFragColor = color;
}