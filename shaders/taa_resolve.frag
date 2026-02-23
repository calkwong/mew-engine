#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "scene.glsl"
#include "samplers.glsl"

layout(set = 1, binding = 0) uniform texture2D allTextures[];
layout(set = 2, binding = 0) uniform sampler samplers[];

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

layout( push_constant ) uniform constants
{
	vec2 screenSize;
	vec2 currentJitter;
	uint color_id;
	uint accum_id;
	uint depth_id;
	uint velocity_id;
	uint debug;
} pc;

// currently unused, should we do this during lighting pass instead?
vec2 unjitterUV(vec2 uv, vec2 currentJitter)
{
	return uv - currentJitter.x * dFdx(uv) + currentJitter.y * dFdy(uv);
}

// TO EXPLORE:
// sample textures with lower mip/unjitteredUV
// variance clipping
// YCOCG space
// accumulate with a motion blurred for currentColor if high motion
// flickering fix

void main()
{
	vec2 uv = inUV;
	
	float depth = texture(sampler2D(allTextures[pc.depth_id], samplers[NEAREST_SAMPLER]), uv).r;
	
	// DEPTH DILATION TO MINIMIZE VELOCITY ALIASING
	float closestDepth = .0;
	vec2 closestUVOffset;
	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			vec2 uvOffset = vec2(x, y) / pc.screenSize;
			float neighbourDepth = texture(sampler2D(allTextures[pc.depth_id], samplers[NEAREST_SAMPLER]), uv + uvOffset).r;
			if (neighbourDepth > closestDepth)
			{
				closestDepth = neighbourDepth;
				closestUVOffset = uvOffset;
			}
		}
	}
	
	vec2 velocityUV = texture(sampler2D(allTextures[pc.velocity_id], samplers[NEAREST_SAMPLER]), uv + closestUVOffset).rg;
	//vec2 velocityUV = texture(sampler2D(allTextures[pc.velocity_id], samplers[NEAREST_SAMPLER]), uv).rg; // no depth dilation
	
	vec2 reprojectedUV = uv - velocityUV;
		
	// where to place unjitteredUV? lighting pass?
	//vec2 unjitteredUV = unjitterUV(uv, pc.currentJitter);
	//uv = unjitteredUV;
	
	// sampled as is with no filtering; UE4 filters and weighs according to distance from pixel center 
	// options: blackman-harris or mitchell
	// in the case of point sampling, if there is disocclusion, we can take a blurred sample (filter around current color?)
	vec3 currentColor = texture(sampler2D(allTextures[pc.color_id], samplers[NEAREST_SAMPLER]), uv).xyz; 
	
	// COLOR CLAMPING - currently 3x3 neighbourhood, we could blend this with a 5 taps '+' pattern per Karis UE4
	// YCoCg for better clamping?
	vec3 minColor = vec3(9999.0);
	vec3 maxColor = vec3(-9999.0);
	for (int x = -1; x <= 1; x++) // TODO: manually unroll this so we can reuse taps for blending etc.
	{
		for (int y = -1; y <= 1; y++)
		{
			vec2 offset = vec2(x, y) / pc.screenSize;
			vec2 textureUV = uv + offset;
			textureUV = clamp(textureUV, vec2(0.0), vec2(1.0));
			vec3 color = texture(sampler2D(allTextures[pc.color_id], samplers[NEAREST_SAMPLER]), textureUV).xyz; 
			color = max(color, vec3(0.0)); // to ensure we don't have garbage values; can we skip this with proper clearing on app side?
			minColor = min(minColor, color);
			maxColor = max(maxColor, color);
		}
	}

	// TODO: use Catmull-Rom; COD version has 5 reads optimization without corners?
	vec3 previousColor = texture(sampler2D(allTextures[pc.accum_id], samplers[LINEAR_SAMPLER]), reprojectedUV).xyz; 
	vec3 previousColorClamped = clamp(previousColor, minColor, maxColor);

	vec3 finalColor = currentColor * 0.1 + previousColorClamped * 0.9;
	
	outFragColor = vec4(finalColor, 1.0);
}