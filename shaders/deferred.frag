#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

#include "scene.glsl"
#include "samplers.glsl"
#include "pbr.glsl"

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

struct OITData
{
	uvec4 colors;
	uvec4 depths;
	vec4 transmissions; // could we pack this in color.a?
};

layout(buffer_reference, std430) buffer OITBuffer
{ 
	OITData frags[];
};

layout( push_constant ) uniform constants
{
	vec4 clusterSize; // xyz is cluster data struct dim, w is single cluster dim where width==height
	vec2 screenSize;
	LightBuffer lightBuffer;
	LightIndexBuffer lightIndexBuffer;
	LightGridBuffer lightGridBuffer;
	OITBuffer oitBuffer;
	uint padding[10]; // padding for visibility buffer variant
	uint depth_id;
	uint albedo_id;    // gbuffer ids
	uint normal_id;    // gbuffer ids
	uint metalroughness_id; // gbuffer ids
	uint shadowmap_id;
	uint lightCulling;
	float near;
	float scale;
	float bias;
	uint debugMeshlets;
	uint resolveTransparent;
	uint shadows;
	uint pcf;
	uint debugShadowmap;
	uint debugCascades;
} pc;

// formula is for infinite far plane, reverse-z
// returns positive value, may need to negate depending on what we're using it for
float linearizeDepthInfiniteReverse(float depth)
{
	return pc.near / depth;
}

const float AMBIENT = 0.1;
const int CASCADE_COUNT = 4;
int MAX_LIGHTS = 1000; // TODO: hardcoded
const int MLAB_NODES = 4;
#define CLUSTERED_SHADING

vec3 compositeTransparent(vec3 inputColor)
{
	vec3 color = inputColor;
	
		uvec2 screenCoords = uvec2(floor(gl_FragCoord.xy));
		uint index = screenCoords.x + screenCoords.y * uint(pc.screenSize.x);
		OITData frags = pc.oitBuffer.frags[index];
		
		// early return if nothing stored
		if (frags.transmissions[0] == 1.0)
		{
			return color;
		}
		
		pc.oitBuffer.frags[index].transmissions = vec4(1.0); // reset so we can skip vkcmdfillbuffer
		
		vec3 composite = vec3(0.);
		float accumT = 1.0;
		for (int i = 0; i < MLAB_NODES; i++)
		{
			float t = frags.transmissions[i];
			composite = t != 1.0 ? unpackUnorm4x8(frags.colors[i]).xyz * accumT + composite : composite;
			accumT *= t;
		}
		
		color *= accumT;
		color += composite;
	
	return color;
}

float calculateShadow(vec3 worldPos, inout uint cascadeIdx)
{
	float d = (sceneData.view * vec4(worldPos, 1.0)).z; // view space z
	for (uint i = 0; i < CASCADE_COUNT; i++)
	{
		if (d > sceneData.cascadeSplits[i])
		{	
			cascadeIdx = i;
			break;
		}
	}
	vec3 lightFragPos = vec3(sceneData.shadowTransforms[cascadeIdx] * vec4(worldPos, 1.0)); // ortho, no division by w needed 
	
	float currentDepth = lightFragPos.z;
	
	if (currentDepth < 0.0) // this is possible
		return 1.0;
	
	vec2 uv = vec2(lightFragPos.x, lightFragPos.y);
	uv = uv * 0.5 + 0.5;
	uv.y = 1.0 - uv.y;
	
	vec2 offset = 1.0 / textureSize(sampler2D(allTextures[pc.shadowmap_id + cascadeIdx], samplers[NEAREST_SAMPLER]), 0);
	
	float shadow = 0.0;
	float closestDepth = 0.0;

	if (pc.pcf == 1)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				vec2 sample_uv = vec2(uv.x + x * offset.x, uv.y + y * offset.y);
				closestDepth = texture(sampler2D(allTextures[pc.shadowmap_id + cascadeIdx], samplers[NEAREST_SAMPLER]), sample_uv).r;
				
				if (closestDepth > currentDepth)
					shadow += 0.3;
				else
					shadow += 1.0;
			}
		}
		
		shadow /= 9.0;
	}
	else
	{
		closestDepth = texture(sampler2D(allTextures[pc.shadowmap_id + cascadeIdx], samplers[LINEAR_SAMPLER]), uv).r;
		
		float bias = 0.0;
		if (closestDepth > currentDepth + bias)
			shadow = 0.0;
		else
			shadow = 1.0;
	}
	return shadow;
}

vec3 reconstructWorldPos(float depth, mat4 viewproj)
{
	vec2 ndc = gl_FragCoord.xy / pc.screenSize;
	ndc = ndc * 2.0 - 1.0;
	ndc.y *= -1.0; // flip as window coords are top down
	vec4 worldPos = inverse(viewproj) * vec4(ndc, depth, 1.0);
	
	return worldPos.xyz / worldPos.w;
}

#define PBR

void main()
{
	vec3 N = texture(sampler2D(allTextures[pc.normal_id], samplers[NEAREST_SAMPLER]), inUV).xyz;
	N = normalize(N); // necessary to remove banding, RGB32 does not need this

	if (pc.debugMeshlets == 1)
	{
		outFragColor = vec4(N, 1.0);
		return;
	}
	
	vec3 albedo = texture(sampler2D(allTextures[pc.albedo_id], samplers[NEAREST_SAMPLER]), inUV).xyz;
	float depth = texture(sampler2D(allTextures[pc.depth_id], samplers[NEAREST_SAMPLER]), inUV).r;
	vec3 worldPos = reconstructWorldPos(depth, sceneData.viewproj);
	
#ifdef PBR
	vec2 metalRoughness = texture(sampler2D(allTextures[pc.metalroughness_id], samplers[NEAREST_SAMPLER]), inUV).xy;
	float metallic = metalRoughness.x;
	float roughness = metalRoughness.y;
	roughness *= roughness;
	
	vec3 Fr = vec3(0.0);
	
	vec3 L = normalize(sceneData.sunlightDir.xyz); 
	vec3 V = normalize(sceneData.cameraPos.xyz - worldPos);
	vec3 H = normalize(L + V);
	
	float NdotL = max(dot(N, L), 0.0);
	float NdotH = max(dot(N, H), 0.0);
	float NdotV = max(dot(N, V), 0.001);
	
	vec3 f0 = vec3(0.04);
	f0 = mix(f0, albedo.xyz, metallic);
	
	vec3 F = F_Schlick(NdotV, f0);
		
	vec3 kS = F;
	vec3 kD = vec3(1.0) - kS;
	
	kD *= 1.0 - metallic;
	vec3 Fd = kD * albedo.xyz / PI;
	
	float D = D_GGX(NdotH, roughness);
	float G = V_SmithGGXCorrelated(NdotV, NdotL, roughness);
	Fr = D * G * F;
	
	vec3 lightColor = vec3(1.0); // HARDCODED SUNLIGHT VALUE
	vec3 Lo = (Fd + Fr) * lightColor * NdotL; 
	outFragColor = vec4(Lo, 1.0);
	outFragColor.xyz += albedo.xyz * AMBIENT; // for debugging without IBL 
#else	
	outFragColor = vec4(albedo, 1.0);
#endif
	
	uint cascadeIdx = 0;
	if (pc.shadows == 1)
	{
		float occluded = calculateShadow(worldPos, cascadeIdx);
		
		outFragColor.xyz *= occluded;
		
		if (pc.debugCascades == 1)
		{
			switch (cascadeIdx)
			{
				case 0:
					outFragColor.xyz *= vec3(1, 0, 0);
					break;
				case 1:
					outFragColor.xyz *= vec3(0, 1, 0);
					break;
				case 2:
					outFragColor.xyz *= vec3(0, 0, 1);
					break;
				case 3:
					outFragColor.xyz *= vec3(1, 1, 0);
					break;
			}
		}
	}
	
	if (pc.lightCulling == 1)
	{
		vec3 color = vec3(0.);
		vec4 clipPos = sceneData.viewproj * vec4(worldPos, 1.0);
		vec3 ndc = clipPos.xyz / clipPos.w;
		vec2 screenPos = ndc.xy * 0.5 + 0.5;
		screenPos = screenPos * pc.screenSize;
		
		ivec4 clusterDim = ivec4(pc.clusterSize);
		ivec2 clusterXY = ivec2(floor(screenPos.xy / clusterDim.w));
		clusterXY.x = clamp(clusterXY.x, 0, clusterDim.x - 1);
		clusterXY.y = clamp(clusterXY.y, 0, clusterDim.y - 1);
		
		float viewZ = linearizeDepthInfiniteReverse(depth); // implicitly flipped
		
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
			vec3 distance = lightPos - worldPos;
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
			
			float attenuation = getSquareFalloffAttenuation(distance, lightRadius);
			color += (Fd + Fr) * lightColor * attenuation * NdotL; 
#else
			float attenuation = getSquareFalloffAttenuation(distance, lightRadius);
			color += (albedo * lightColor * attenuation * NdotL);
#endif
		}
		outFragColor.xyz += color;
	}
	
	if (pc.resolveTransparent == 1)
	{
		outFragColor.xyz = compositeTransparent(outFragColor.xyz);
	}
	
	if (pc.debugShadowmap != 0)
	{
		vec2 uv = gl_FragCoord.xy / pc.screenSize;
		uint idx = pc.debugShadowmap - 1;
		vec3 depth = vec3(texture(sampler2D(allTextures[pc.shadowmap_id + idx], samplers[NEAREST_SAMPLER]), uv).r);
		outFragColor.xyz = depth;
	}
}