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

struct LightGrid
{
	uint offset;
	uint count;
};

layout(buffer_reference, std430) readonly buffer LightBuffer
{
	PointLight lights[];
};

layout(buffer_reference, std430) readonly buffer LightIndexBuffer
{
	uint indices[];
};

layout(buffer_reference, std430) readonly buffer LightGridBuffer
{
	LightGrid grid[];
};

layout( push_constant ) uniform constants
{
	vec4 clusterSize; // xyz is cluster data struct dim, w is single cluster dim where width==height
	vec2 screenSize;
	LightBuffer lightBuffer;
	LightIndexBuffer lightIndexBuffer;
	LightGridBuffer lightGridBuffer;
	uint depth_id;
	uint albedo_id;    // gbuffer ids
	uint normal_id;    // gbuffer ids
	uint world_pos_id; // gbuffer ids
	uint lightCulling;
	float near;
	float scale;
	float bias;
	uint debugMeshlets;
} pc;

float distanceSquared(vec3 a, vec3 b)
{
	vec3 d = a - b;
	return dot(d, d);
}

// formula is for infinite far plane, reverse-z
// returns positive value, may need to negate depending on what we're using it for
float linearizeDepthInfiniteReverse(float depth)
{
	return pc.near / depth;
}

int MAX_LIGHTS = 1000; // TODO: hardcoded
#define CLUSTERED_SHADING

void main()
{
	if (pc.debugMeshlets == 1)
	{
		vec3 normal = texture(sampler2D(allTextures[pc.normal_id], samplers[NEAREST_SAMPLER]), inUV).xyz;
		outFragColor = vec4(normal, 1.0);
		return;
	}
	vec3 albedo = texture(sampler2D(allTextures[pc.albedo_id], samplers[NEAREST_SAMPLER]), inUV).xyz;
	vec3 worldPos = texture(sampler2D(allTextures[pc.world_pos_id], samplers[NEAREST_SAMPLER]), inUV).xyz;
	
	vec3 color = vec3(0.);
	
	if (pc.lightCulling == 1)
	{
#ifndef CLUSTERED_SHADING
		for (int i = 0; i < MAX_LIGHTS; i++)
		{
			PointLight light = pc.lightBuffer.lights[i];
			vec3 lightCenter = light.pos.xyz;
			lightCenter = vec3(sceneData.lightRot * vec4(lightCenter, 1.0));
			float lightRadius = light.pos.w;
			vec3 lightColor = light.color.xyz;
		
			float distance = distance(worldPos, lightCenter);
			
			if (distance < lightRadius)
			{
				color += albedo * lightColor;
			}
		}
#else
		vec4 clipPos = sceneData.viewproj * vec4(worldPos, 1.0);
		vec3 ndc = clipPos.xyz / clipPos.w;
		vec2 screenPos = ndc.xy * 0.5 + 0.5;
		screenPos = screenPos * pc.screenSize;
		
		ivec4 clusterDim = ivec4(pc.clusterSize);
		ivec2 clusterXY = ivec2(floor(screenPos.xy / clusterDim.w));
		clusterXY.x = clamp(clusterXY.x, 0, clusterDim.x - 1);
		clusterXY.y = clamp(clusterXY.y, 0, clusterDim.y - 1);
		
		float viewZ = -(sceneData.view * vec4(worldPos, 1.0)).z; // possible precision tradeoff
		//float depth = texture(sampler2D(allTextures[pc.depth_id], samplers[NEAREST_SAMPLER]), inUV).r;
		//float viewZ = linearizeDepthInfiniteReverse(depth);
		
		// equation (3): https://www.aortiz.me/2018/12/21/CG.html#part-2 
		// slide 5: https://advances.realtimerendering.com/s2016/Siggraph2016_idTech6.pdf
		uint slice = uint(floor(log(viewZ) * pc.scale - pc.bias));
		slice = clamp(slice, 0, clusterDim.z - 1);
		
		uint clusterIndex = clusterXY.x + clusterDim.x * clusterXY.y + slice * clusterDim.x * clusterDim.y; 
		
		uint offset = pc.lightGridBuffer.grid[clusterIndex].offset;
		uint count = pc.lightGridBuffer.grid[clusterIndex].count;
		
		for (int i = 0; i < count; i++)
		{
			uint index = pc.lightIndexBuffer.indices[offset + i];
			vec3 lightPos = pc.lightBuffer.lights[index].pos.xyz;
			lightPos = vec3(sceneData.lightRot * vec4(lightPos, 1.0));
			float lightRadius = pc.lightBuffer.lights[index].pos.w;
			vec3 lightColor = pc.lightBuffer.lights[index].color.xyz;
			
			float d = distanceSquared(worldPos, lightPos);
	
			if (d <= lightRadius * lightRadius)
			{
				color += (albedo * lightColor);
			}
		}
#endif
	}
	
	vec3 ambient = 0.05 * albedo;
	
	outFragColor = vec4(color + ambient, 1.0);
	
	if (pc.lightCulling == 0)
		outFragColor = vec4(albedo, 1.0);
}