#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

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

layout(set = 0, binding = 0) uniform SceneData
{   
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec3 cameraPos;
	
} sceneData;

layout(set = 1, binding = 0) uniform texture2D allTextures[];
layout(set = 2, binding = 0) uniform sampler samplers[];


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

// ndf
float D_GGX(float NdotH, float roughness)
{
	float a = NdotH * roughness;
	float k = roughness / (1.0 - NdotH * NdotH + a * a);
	return k * k * (1.0 / PI);
}

// geometric
float V_SmithGGXCorrelated(float NdotV, float NdotL, float roughness)
{
	float a2 = roughness * roughness;
	float GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
	float GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
	return 0.5 / (GGXV + GGXL);
}

// fresnel with roughness
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

void main() 
{	
	MaterialData m = pc.materialBuffer.materials[pc.materialID];
	vec3 Lo = vec3(0.0);
	
	//vec4 albedo = texture(allTextures[m.diffuseID], inUV);
	vec4 albedo = texture(sampler2D(allTextures[m.diffuseID], samplers[0]), inUV);
	vec3 lightColor = vec3(1.0);
	
	// normal mapping
	vec3 vN = inNormal;
	vec3 vT = inTangent.xyz;
	float sign = inTangent.w; // sign is flipped during tangent generation so mikktspace is consistent with glTF handedness
	vec3 vB = sign * cross(vN, vT);
	//vec3 sampleNormal = texture(allTextures[m.normalID], inUV).xyz;
	vec3 sampleNormal = texture(sampler2D(allTextures[m.normalID], samplers[0]), inUV).xyz;
	sampleNormal = sampleNormal * 2.0 - 1.0;
	vec3 N = normalize(sampleNormal.x * vT + sampleNormal.y * vB + sampleNormal.z * vN);
	
	//vec3 N = normalize(inNormal);
	vec3 V = normalize(sceneData.cameraPos.xyz - inWorldPos);
	
	//vec2 metalRoughness = texture(allTextures[m.metalRoughnessID], inUV).bg;
	vec2 metalRoughness = texture(sampler2D(allTextures[m.metalRoughnessID], samplers[0]), inUV).bg;
	float metallic = metalRoughness.x;
	float perceptualRoughness = metalRoughness.y;
	perceptualRoughness = max(perceptualRoughness, 0.045); // frostbite engine clamp value for analytical lights (fp32)
	float roughness = perceptualRoughness * perceptualRoughness;
	
	vec3 f0 = mix(vec3(0.04), albedo.xyz, metallic);

	float NdotV = max(dot(N, V), 0.0);

	for	(int i = 0; i < 1; i++)
	{
		vec3 Fr = vec3(0.0);
		
		//vec3 L = normalize(sceneData.lights[i].xyz - inWorldPos);
		vec3 L = normalize(vec3(0, 0, 15) - inWorldPos); // hardcoded camera starting position
		vec3 H = normalize(L + V);
		
		float NdotL = max(dot(N, L), 0.0);
		float NdotH = max(dot(N, H), 0.0);
		
		vec3 F = F_Schlick(NdotV, f0);
			
		vec3 kS = F;
		vec3 kD = vec3(1.0) - kS;
		kD *= 1.0 - metallic;
		vec3 Fd = kD * albedo.xyz / PI;
		Fd = vec3(0.0);
		
		float D = D_GGX(NdotH, roughness);
		float G = V_SmithGGXCorrelated(NdotV, NdotL, roughness);
		Fr = D * G * F;
		
		Lo += (Fd + Fr) * lightColor * NdotL;
	}
	
	// Combine with ambient
	//vec4 color = vec4(albedo.xyz * 0.1, 1);
	//vec4 color = vec4(albedo.xyz * texture(allTextures[m.occlusionID], inUV).rrr, 1); 
	vec4 color = vec4(albedo.xyz * texture(sampler2D(allTextures[m.occlusionID], samplers[0]), inUV).rrr, 1); 
	//vec4 color = vec4(vec3(0), 1.0);
	color.xyz += Lo;
	//color.xyz += texture(allTextures[m.emissiveID], inUV).xyz;
	color.xyz += texture(sampler2D(allTextures[m.emissiveID], samplers[0]), inUV).xyz;

	// tonemapping
	color.xyz = Uncharted2Tonemap(color.xyz * exposure);
	color.xyz = color.xyz * (1.0 / Uncharted2Tonemap(vec3(11.2)));
	//color.xyz = pow(color.xyz, vec3(1.0 / gamma));

	outFragColor = vec4(color);
	//outFragColor = albedo;
	//outFragColor = vec4(N, 1);
	//outFragColor = vec4(inNormal, 1);
	//outFragColor = vec4(vec3(metallic), 1.0);
	//outFragColor.xyz = outFragColor.xyz * 0.5 + 0.5;
}
