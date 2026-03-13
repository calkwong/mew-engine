#version 450

#extension GL_GOOGLE_include_directive : require

#define OIT_RESOLVE

#include "bindings.glsl"
#include "buffer_references.glsl"
#include "pbr.glsl"
#include "sh.glsl"

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

layout( push_constant ) uniform constants
{
	vec4 clusterSize; // xyz is cluster data struct dim, w is single cluster dim where width==height
	vec2 screenSize;
	LightBuffer lightBuffer;
	LightIndexBuffer lightIndexBuffer;
	LightGridBuffer lightGridBuffer;
	OITBuffer oitBuffer;
	uint padding[10]; // padding for visibility buffer variant
	SHBuffer shBuffer;
	uint depth_id;
	uint gbuffer_id;  
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
	// gi
	float maxPrefilteredLod;
	float metallic; // unused, for debugging
	float roughness; // unused, for debugging
} pc;

// formula is for infinite far plane, reverse-z
// returns positive value, may need to negate depending on what we're using it for
float linearizeDepthInfiniteReverse(float depth)
{
	return pc.near / depth;
}

const int CASCADE_COUNT = 4;
const int MLAB_NODES = 4;

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
	
	vec2 offset = 1.0 / textureSize(sampler2D(textures[pc.shadowmap_id + cascadeIdx], samplers[NEAREST_SAMPLER]), 0);
	
	float shadow = 0.0;
	float closestDepth = 0.0;

	if (pc.pcf == 1)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				vec2 sample_uv = vec2(uv.x + x * offset.x, uv.y + y * offset.y);
				closestDepth = texture(sampler2D(textures[pc.shadowmap_id + cascadeIdx], samplers[NEAREST_SAMPLER]), sample_uv).r;
				
				if (closestDepth > currentDepth)
					shadow += 0.0;
				else
					shadow += 1.0;
			}
		}
		
		shadow /= 9.0;
	}
	else
	{
		closestDepth = texture(sampler2D(textures[pc.shadowmap_id + cascadeIdx], samplers[LINEAR_SAMPLER]), uv).r;
		
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
#define GI

void main()
{
	uint albedo_id = pc.gbuffer_id;
	uint normal_id = pc.gbuffer_id + 1;
	uint metalroughness_id = pc.gbuffer_id + 2;

	vec3 N = texture(sampler2D(textures[normal_id], samplers[NEAREST_SAMPLER]), inUV).xyz;
	N = normalize(N); // necessary to remove banding, RGB32 does not need this

	if (pc.debugMeshlets == 1)
	{
		outFragColor = vec4(N, 1.0);
		return;
	}
	
	vec4 albedo = vec4(texture(sampler2D(textures[albedo_id], samplers[NEAREST_SAMPLER]), inUV).xyz, 1.0);
	float depth = texture(sampler2D(textures[pc.depth_id], samplers[NEAREST_SAMPLER]), inUV).r;
	vec3 worldPos = reconstructWorldPos(depth, sceneData.viewproj);
	
	vec3 color = vec3(0.0);
	vec3 ambient = albedo.xyz * 0.1; // when GI is off, likely going to look physically incorrect
	
#ifdef PBR
	vec2 metalRoughness = texture(sampler2D(textures[metalroughness_id], samplers[NEAREST_SAMPLER]), inUV).xy;
	float metallic = metalRoughness.x;
	float perceptualRoughness = metalRoughness.y;
	float roughness = perceptualRoughness * perceptualRoughness;
	
	vec3 Fr = vec3(0.0);
	
	vec3 L = normalize(sceneData.sunlightDir.xyz); 
	vec3 V = normalize(sceneData.cameraPos.xyz - worldPos);
	vec3 H = normalize(L + V);
	
	float NdotL = max(dot(N, L), 0.0);
	float NdotH = max(dot(N, H), 0.0);
	float NdotV = max(dot(N, V), 0.001);
	float VdotH = max(dot(V, H), 0.0);
	
	vec3 f0 = vec3(0.04);
	f0 = mix(f0, albedo.xyz, metallic);
	
	vec3 F = F_Schlick(VdotH, f0);
		
	vec3 kS = F;
	vec3 kD = vec3(1.0) - kS;
	
	kD *= 1.0 - metallic;
	vec3 Fd = kD * albedo.xyz / PI;
	
	float D = D_GGX(NdotH, roughness);
	float G = V_SmithGGXCorrelated(NdotV, NdotL, roughness);
	Fr = D * G * F;
	
	vec3 lightColor = vec3(15.0); // HARDCODED SUNLIGHT VALUE
	color = (Fd + Fr) * lightColor * NdotL; 
	
	#ifdef GI
		//perceptualRoughness = pc.roughness; // sphere test
		//metallic = pc.metallic; // sphere test
		
		vec3 R = reflect(-V, N);
		float prefilteredMip = pc.maxPrefilteredLod * perceptualRoughness;
		
		{
			vec3 irradiance = evaluateSH(pc.shBuffer.rCoefficients, pc.shBuffer.gCoefficients, pc.shBuffer.bCoefficients, N);
			vec3 prefiltered = textureLod(samplerCube(textures_cube[uint(sceneData.textures[2])], samplers[CUBE_SAMPLER]), R, prefilteredMip).xyz;
			vec2 brdf = texture(sampler2D(textures[uint(sceneData.textures[3])], samplers[LINEAR_CLAMP_SAMPLER]), vec2(NdotV, perceptualRoughness)).rg;
			
			//vec3 white = vec3(1.0); // sphere test
			//f0 = vec3(0.04); // sphere test
			f0 = mix(f0, albedo.xyz, metallic);
			//f0 = mix(f0, white, metallic); // sphere test
			
			F = F_SchlickRoughness(NdotV, f0, perceptualRoughness);
			kS = F;
			kD = vec3(1.0) - kS;
			kD *= 1.0 - metallic;
			
		    // TODO: we could bake the division by PI into SH
			vec3 diffuse = kD * irradiance * (1.0 / PI) * albedo.xyz;
			vec3 specular = prefiltered * (F * brdf.x + brdf.y);
			ambient = diffuse + specular;
			//outFragColor = vec4(ambient, 1.0); // sphere test
			//return; // sphere test
		}
	#endif
#else	
	color += albedo.xyz;
#endif
	
	uint cascadeIdx = 0;
	if (pc.shadows == 1)
	{
		float occluded = calculateShadow(worldPos, cascadeIdx);
		
		color *= occluded;
		
		if (pc.debugCascades == 1)
		{
			switch (cascadeIdx)
			{
				case 0:
					color *= vec3(1, 0, 0);
					break;
				case 1:
					color *= vec3(0, 1, 0);
					break;
				case 2:
					color *= vec3(0, 0, 1);
					break;
				case 3:
					color *= vec3(1, 1, 0);
					break;
			}
		}
	}
	
	color += ambient;
	
	if (pc.lightCulling == 1)
	{
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
			color += (albedo.xyz * lightColor * attenuation * NdotL);
#endif
		}
	}
	
	// TODO: refactor and use depth/stencil buffer to reject pixels in future
	if (depth == 0.0)
	{
		vec2 ndc = inUV * 2.0 - 1.0;
		ndc.y *= -1.0;
		vec3 sampleDir = vec3(inverse(sceneData.viewproj) * vec4(ndc, 0.0, 1.0));
		color = texture(samplerCube(textures_cube[uint(sceneData.textures[0])], samplers[CUBE_SAMPLER]), sampleDir).xyz;
	}
	
	if (pc.resolveTransparent == 1)
	{
		color = compositeTransparent(color);
	}
	
	outFragColor = vec4(color, 1.0);
	
	if (pc.debugShadowmap != 0)
	{
		vec2 uv = gl_FragCoord.xy / pc.screenSize;
		uint idx = pc.debugShadowmap - 1;
		vec3 depth = vec3(texture(sampler2D(textures[pc.shadowmap_id + idx], samplers[NEAREST_SAMPLER]), uv).r);
		outFragColor.xyz = depth;
	}
}