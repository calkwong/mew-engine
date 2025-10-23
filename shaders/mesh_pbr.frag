#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

#include "scene.glsl"

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inWorldPos;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec4 inTangent;

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

float calculate_shadow()
{
	vec3 lightFragPos = vec3(sceneData.shadowTransform * vec4(inWorldPos, 1.0)); // ortho, no division by w needed
	
	float currentDepth = lightFragPos.z;
	
	if (currentDepth < 0.0 || currentDepth > 1.0)
		return 1.0;
	
	vec2 uv = vec2(lightFragPos.x, lightFragPos.y);
	uv = uv * 0.5 + 0.5;
	uv.y = 1.0 - uv.y;
	float closestDepth = texture(sampler2D(allTextures[sceneData.shadow_id], samplers[1]), uv).r;
	
	if (closestDepth > currentDepth)
		return 0.0;
	return 1.0;
}

#define PBR
//define IBL

void main() 
{	
	MaterialData m = pc.materialBuffer.materials[pc.materialID];
	
	vec4 albedo = texture(sampler2D(allTextures[m.diffuseID], samplers[0]), inUV) * m.baseColorFactor;
	vec3 lightColor = vec3(1.0);
	
	// normal mapping
	vec3 vN = inNormal;
	vec3 vT = inTangent.xyz;
	float sign = inTangent.w; // sign is flipped during tangent generation so mikktspace is consistent with glTF handedness
	vec3 vB = sign * cross(vN, vT);
	vec3 sampleNormal = texture(sampler2D(allTextures[m.normalID], samplers[0]), inUV).xyz;
	sampleNormal = sampleNormal * 2.0 - 1.0;
	vec3 N = normalize(sampleNormal.x * vT + sampleNormal.y * vB + sampleNormal.z * vN);
	//vec3 N = normalize(inNormal); // use geometry normal
	vec3 V = normalize(sceneData.cameraPos.xyz - inWorldPos);
	
	vec2 metalRoughness = texture(sampler2D(allTextures[m.metalRoughnessID], samplers[0]), inUV).bg;
	float metallic = metalRoughness.x * m.metallicFactor;
	float perceptualRoughness = metalRoughness.y * m.roughnessFactor;
	perceptualRoughness = max(perceptualRoughness, 0.045); // frostbite engine clamp value for analytical lights (fp32)
	float roughness = perceptualRoughness * perceptualRoughness;
	
	float NdotV = max(dot(N, V), 0.001);
	
	vec3 f0 = vec3(0.04);
	f0 = mix(f0, albedo.xyz, metallic);
	
	vec4 ambient = vec4(0.0);
	
	#ifdef IBL
		vec3 R = reflect(-V, N);
		float MAX_CURRENT_LOD = 7.0; // make into PC, 7 or 8?
		float mip_level = perceptualRoughness * MAX_CURRENT_LOD;
		vec3 irradiance = texture(samplerCube(allCubemaps[sceneData.irradiance_id], samplers[1]), N).rgb;
		vec3 prefiltered = textureLod(samplerCube(allCubemaps[sceneData.prefiltered_id], samplers[1]), R, mip_level).rgb;
		vec2 brdf = texture(sampler2D(allTextures[sceneData.brdf_id], samplers[1]), vec2(NdotV, perceptualRoughness)).rg;
	
		vec3 F = F_SchlickRoughness(NdotV, f0, perceptualRoughness);
		vec3 kS = F;
		vec3 kD = vec3(1.0) - kS;
		kD *= 1.0 - metallic;
	
		vec3 diffuse = kD * irradiance * albedo.xyz; // division by PI already baked into irradiance map
		vec3 specular = prefiltered * (F * brdf.x + brdf.y);
		
		ambient.xyz = diffuse + specular;
		ambient.xyz *= texture(sampler2D(allTextures[m.occlusionID], samplers[0]), inUV).r;
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
	//vec4 color = vec4(albedo.xyz * 0.1, 1); // 10% albedo as ambient
	//vec4 color = vec4(vec3(0), 1.0); // no direct lighting
	vec4 color = vec4(Lo, 1.0);
	float occluded = calculate_shadow();
	color.xyz *= occluded;
	color.xyz += texture(sampler2D(allTextures[m.emissiveID], samplers[0]), inUV).xyz;
	
	color += ambient;

	outFragColor = vec4(color);
	//outFragColor = albedo;
	//outFragColor = ambient;
	//outFragColor = vec4(vec3(perceptualRoughness), 1);
	//outFragColor = vec4(N, 1);
	//outFragColor = vec4(inNormal, 1);
	//outFragColor.xyz = outFragColor.xyz * 0.5 + 0.5;
}
