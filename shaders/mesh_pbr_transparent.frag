#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

#include "scene.glsl"

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inWorldPos;
layout (location = 2) in vec3 inViewPos;
layout (location = 3) in vec2 inUV;
layout (location = 4) in vec4 inTangent;

layout (location = 0) out vec4 outFragColor;

const float exposure = 4.0;
const float gamma = 2.2;
const float PI = 3.14159265359;

struct Vertex 
{
	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 tangent;
}; 

struct MaterialData
{
	vec4 baseColorFactor;
	float metallicFactor;
	float roughnessFactor;
	uint diffuseID;
	uint metalRoughnessID;
	uint normalID;
	uint occlusionID;
	uint emissiveID;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer
{ 
	Vertex vertices[];
};

layout(buffer_reference, std430) readonly buffer MaterialBuffer
{ 
	MaterialData materials[];
};

layout( push_constant ) uniform constants
{
	mat4 worldMatrix;
	VertexBuffer vertexBuffer;
	MaterialBuffer materialBuffer;
	uint materialID;
	uint debug_idx;
} pc;

layout(set = 1, binding = 0) uniform texture2D allTextures[];
layout(set = 1, binding = 0) uniform textureCube allCubemaps[];
layout(set = 2, binding = 0) uniform sampler samplers[];

float D_GGX(float NdotH, float roughness)
{
	float a = NdotH * roughness;
	float k = roughness / (1.0 - NdotH * NdotH + a * a);
	return k * k * (1.0 / PI);
}

float V_SmithGGXCorrelated(float NdotV, float NdotL, float roughness)
{
	float a2 = roughness * roughness;
	float GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
	float GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
	return 0.5 / (GGXV + GGXL);
}

vec3 F_SchlickRoughness(float u, vec3 f0, float roughness)
{
	return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(1.0 - u, 5.0);
}

// fresnel, from Filament (same eq, just rearranged)
vec3 F_Schlick(float u, vec3 f0) 
{
    float f = pow(1.0 - u, 5.0);
    return f + f0 * (1.0 - f);
}

#define CASCADE_COUNT 4

float calculate_shadow()
{
	float d = inViewPos.z; 
	uint cascade_index = 0;;
	for (uint i = 0; i < CASCADE_COUNT; i++)
	{
		if (d > sceneData.cascadeSplits[i])
		{	
			cascade_index = i;
			break;
		}
	}
	
	vec3 lightFragPos = vec3(sceneData.shadowTransforms[cascade_index] * vec4(inWorldPos, 1.0)); // ortho, no division by w needed
	
	float currentDepth = lightFragPos.z;
	
	if (currentDepth < 0.0)
		return 1.0;
	
	vec2 uv = vec2(lightFragPos.x, lightFragPos.y);
	uv = uv * 0.5 + 0.5;
	uv.y = 1.0 - uv.y;
	
	uint shadow_id = uint(sceneData.textures[3]);
	
	vec2 offset = 1.0 / textureSize(sampler2D(allTextures[shadow_id + cascade_index], samplers[2]), 0);
	
	float shadow = 0.0;
	float closestDepth = 0.0;
	for (int y = -1; y <= 1; y++)
	{
		for (int x = -1; x <= 1; x++)
		{
			vec2 sample_uv = vec2(uv.x + x * offset.x, uv.y + y * offset.y);
			closestDepth = texture(sampler2D(allTextures[shadow_id + cascade_index], samplers[2]), sample_uv).r;
			
			if (closestDepth > currentDepth)
				shadow += 0.0;
			else
				shadow += 1.0;
		}
	}
	
	shadow /= 9.0;
	return shadow;
}

#define PBR
#define IBL

void main() 
{	
	MaterialData m = pc.materialBuffer.materials[pc.materialID];
	
	uint irradiance_id = uint(sceneData.textures[0]);
	uint prefiltered_id = uint(sceneData.textures[1]);
	uint brdf_id = uint(sceneData.textures[2]);
	
	vec4 albedo = m.baseColorFactor;
	if (m.diffuseID != 0)
		albedo *= texture(sampler2D(allTextures[m.diffuseID], samplers[0]), inUV);
	vec3 lightColor = vec3(1.0);
	
	// normal mapping
	vec3 N = normalize(inNormal); // normalize or not? shouldnt mikktspace leave as is? added for khronos sponza
	vec3 T = normalize(inTangent.xyz); // normalize or not? shouldnt mikktspace leave as is? added for khronos sponza
	float sign = inTangent.w; // sign is flipped during tangent generation so mikktspace is consistent with glTF handedness
	vec3 B = sign * cross(N, T);
	
	vec3 sampleNormal = vec3(0.0);
	if (m.normalID != 0)
	{
		sampleNormal = texture(sampler2D(allTextures[m.normalID], samplers[0]), inUV).xyz;
		sampleNormal = sampleNormal * 2.0 - 1.0;
		N = normalize(sampleNormal.x * T + sampleNormal.y * B + sampleNormal.z * N);
	}
	
	N = gl_FrontFacing ? N : -N; 
	
	vec3 V = normalize(sceneData.cameraPos.xyz - inWorldPos);
	
	float metallic = m.metallicFactor;
	float perceptualRoughness = m.roughnessFactor;
	vec2 metalRoughness = vec2(0.0);
	if (m.metalRoughnessID != 0)
	{
		metalRoughness = texture(sampler2D(allTextures[m.metalRoughnessID], samplers[0]), inUV).bg;
		metallic *= metalRoughness.x;
		perceptualRoughness *= metalRoughness.y;
	}
	perceptualRoughness = max(perceptualRoughness, 0.045); // frostbite engine clamp value for analytical lights (fp32)
	float roughness = perceptualRoughness * perceptualRoughness;
	
	float NdotV = max(dot(N, V), 0.001);
	
	vec3 f0 = vec3(0.04);
	f0 = mix(f0, albedo.xyz, metallic);
	
	vec3 ambient = vec3(0.0);
	
	#ifdef IBL
		vec3 R = reflect(-V, N);
		float MAX_CURRENT_LOD = 7.0; // make into PC, 7 or 8?
		float mip_level = perceptualRoughness * MAX_CURRENT_LOD;
		vec3 irradiance = texture(samplerCube(allCubemaps[irradiance_id], samplers[1]), N).rgb;
		vec3 prefiltered = textureLod(samplerCube(allCubemaps[prefiltered_id], samplers[1]), R, mip_level).rgb;
		vec2 brdf = texture(sampler2D(allTextures[brdf_id], samplers[1]), vec2(NdotV, perceptualRoughness)).rg;
	
		vec3 F = F_SchlickRoughness(NdotV, f0, perceptualRoughness);
		vec3 kS = F;
		vec3 kD = vec3(1.0) - kS;
		kD *= 1.0 - metallic;
	
		vec3 diffuse = kD * irradiance * albedo.xyz; // division by PI already baked into irradiance map
		vec3 specular = prefiltered * (F * brdf.x + brdf.y);
		
		ambient.xyz = diffuse + specular;
		float occlusion = 1.0;
		if (m.occlusionID != 0)
		{
			occlusion = texture(sampler2D(allTextures[m.occlusionID], samplers[0]), inUV).r;
			ambient.xyz *= occlusion;
		}
	#endif
	
	vec3 Lo = vec3(0.0);
	
	#ifdef PBR
		// loop over lights
		for	(int i = 0; i < 1; i++)
		{
			vec3 Fr = vec3(0.0);
			
			vec3 L = normalize(sceneData.sunlightDir.xyz - inWorldPos);
			vec3 H = normalize(L + V);
			
			float NdotL = max(dot(N, L), 0.0);
			float NdotH = max(dot(N, H), 0.0);
			
			vec3 F = F_Schlick(NdotV, f0);
				
			vec3 kS = F;
			vec3 kD = vec3(1.0) - kS;
			kD *= 1.0 - metallic;
			vec3 Fd = kD * albedo.xyz / PI;
			
			float D = D_GGX(NdotH, roughness);
			float G = V_SmithGGXCorrelated(NdotV, NdotL, roughness);
			Fr = D * G * F;
			
			Lo += (Fd + Fr) * lightColor * NdotL; // (!) verify eq with lightColor
		}
	#endif
	
	// Combine with ambient
	//vec4 color = vec4(vec3(0), 1.0); // no direct lighting, sphere debug?
	
	vec4 color = vec4(Lo, albedo.a);
	//vec4 color = vec4(Lo, 1.0);
	//color += vec4(albedo.xyz * 0.1, 1); // 10% albedo as ambient, for debugging without IBL
	
	//float occluded = calculate_shadow();
	//color.xyz *= occluded;
	
	
	vec3 emission = vec3(0.0);
	if (m.emissiveID != 0)
	{
		emission = texture(sampler2D(allTextures[m.emissiveID], samplers[0]), inUV).xyz;
		color.xyz += emission;
	}
	
	float ibl_strength = 0.3;
	
	color.xyz += ambient * ibl_strength;

	// premultiplied alpha
	color.xyz *= color.a;

	outFragColor = color;
	
	switch (pc.debug_idx)
	{
		case 0: break;
		case 1: outFragColor = vec4(albedo.xyz, 1.0); break;
		case 2: outFragColor = vec4(normalize(inNormal), 1.0); outFragColor.xyz = outFragColor.xyz * 0.5 + 0.5; outFragColor.xyz = pow(outFragColor.xyz, vec3(2.2)); break; // geometry 
		case 3: outFragColor = vec4(sampleNormal, 1.0); outFragColor.xyz = outFragColor.xyz * 0.5 + 0.5; outFragColor.xyz = pow(outFragColor.xyz, vec3(2.2)); break; // texture normal
		case 4: outFragColor = vec4(N, 1.0); outFragColor.xyz = outFragColor.xyz * 0.5 + 0.5; outFragColor.xyz = pow(outFragColor.xyz, vec3(2.2)); break; // shading normal
		case 5: outFragColor = vec4(T, 1.0); outFragColor.xyz = outFragColor.xyz * 0.5 + 0.5; outFragColor.xyz = pow(outFragColor.xyz, vec3(2.2)); break;
		case 6: outFragColor = vec4(vec3(metalRoughness.x), 1.0); outFragColor.xyz = pow(outFragColor.xyz, vec3(2.2)); break;
		case 7: outFragColor = vec4(vec3(metalRoughness.y), 1.0); outFragColor.xyz = pow(outFragColor.xyz, vec3(2.2)); break;
		case 8: outFragColor = vec4(vec2(inUV), 0.0, 1.0); outFragColor.xyz = pow(outFragColor.xyz, vec3(2.2)); break;
		case 9: outFragColor = vec4(vec3(inTangent.w), 1.0); break;
		default: break;
	}
}
