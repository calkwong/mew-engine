#version 450

#extension GL_GOOGLE_include_directive : require

#define OIT_RESOLVE

#include "bindings.glsl"
#include "buffer_references.glsl"
#include "vbuffer.glsl"
#include "pbr.glsl"
#include "sh.glsl"
#include "util.glsl"

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
	MeshletIndicesBuffer meshletIndicesBuffer;
	MeshletBuffer meshletBuffer;
	VertexBuffer vertexBuffer;
	ObjectBuffer objectBuffer;
	MaterialBuffer materialBuffer;
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
	uint debug;
} pc;

// formula is for infinite far plane, reverse-z
// returns positive z, negate depending on usage
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

#define PBR
#define GI

void main()
{
	//// HANDLE LATER, NOT SUPPORTED ON VISIBILITY PATH
	//if (pc.debugMeshlets == 1)
	//{
	//	vec3 normal = texture(sampler2D(textures[pc.normal_id], samplers[NEAREST_SAMPLER]), inUV).xyz;
	//	outFragColor = vec4(normal, 1.0);
	//	return; 
	//}
	
	uvec2 data = texture(usampler2D(textures_u[pc.gbuffer_id], samplers[NEAREST_SAMPLER]), inUV).rg; 
	
	uint drawID = data.x; 
	uint packedID = data.y;
	uint triangleID = bitfieldExtract(packedID, 25, 7);
	uint meshletID = bitfieldExtract(packedID, 0, 25);
	
	Meshlet meshlet = pc.meshletBuffer.meshlets[meshletID];
	uint triangleOffset = meshlet.dataOffset + meshlet.vertexCount;
	
	uint idx0 = pc.meshletIndicesBuffer.indices[triangleOffset + triangleID * 3 + 0];
	uint idx1 = pc.meshletIndicesBuffer.indices[triangleOffset + triangleID * 3 + 1];
	uint idx2 = pc.meshletIndicesBuffer.indices[triangleOffset + triangleID * 3 + 2];
	
	uint vertexIndex0 = pc.meshletIndicesBuffer.indices[meshlet.dataOffset + idx0]; 
	uint vertexIndex1 = pc.meshletIndicesBuffer.indices[meshlet.dataOffset + idx1]; 
	uint vertexIndex2 = pc.meshletIndicesBuffer.indices[meshlet.dataOffset + idx2]; 
	
	Vertex v0 = pc.vertexBuffer.vertices[vertexIndex0];
	Vertex v1 = pc.vertexBuffer.vertices[vertexIndex1];
	Vertex v2 = pc.vertexBuffer.vertices[vertexIndex2];
	
	mat4 worldMatrix = pc.objectBuffer.objects[drawID].worldMatrix;
	
	vec4 wp0 = worldMatrix * vec4(v0.px, v0.py, v0.pz, 1.0); 
	vec4 wp1 = worldMatrix * vec4(v1.px, v1.py, v1.pz, 1.0); 
	vec4 wp2 = worldMatrix * vec4(v2.px, v2.py, v2.pz, 1.0); 
	
	vec4 p0 = sceneData.viewproj * wp0;
	vec4 p1 = sceneData.viewproj * wp1;
	vec4 p2 = sceneData.viewproj * wp2;
	
	// vulkan top left origin, hence flipping y is necessary
	vec2 pNdc = gl_FragCoord.xy / pc.screenSize; 
	pNdc = pNdc * 2.0 - 1.0;
	pNdc.y = -pNdc.y; 
	
	BarycentricDeriv bary = calculateBarycentric(p0, p1, p2, pNdc, pc.screenSize);
	vec2 uv;
	vec2 uvDdx;
	vec2 uvDdy;
	interpolateWithDeriv(bary, vec2(v0.uv_x, v0.uv_y), vec2(v1.uv_x, v1.uv_y), vec2(v2.uv_x, v2.uv_y), uv, uvDdx, uvDdy);
	
	vec3 debugUV = vec3(uv, 0.0);
	
	vec3 worldPos = interpolate(bary, wp0.xyz, wp1.xyz, wp2.xyz);
	
	vec3 np0, np1, np2;
	vec4 tp0, tp1, tp2;
	unpackTBN(v0.normal, uint(v0.tangent), np0, tp0);
	unpackTBN(v1.normal, uint(v1.tangent), np1, tp1);
	unpackTBN(v2.normal, uint(v2.tangent), np2, tp2);
	
	// TODO: store triangle face bit in visibility buffer or recompute for masked geometry only so normals are correct? 
	
	vec3 n0 = mat3(worldMatrix) * np0;  // no transpose(inverse), no normalization
	vec3 n1 = mat3(worldMatrix) * np1;
	vec3 n2 = mat3(worldMatrix) * np2;
	
	vec3 N = normalize(interpolate(bary, n0, n1, n2)); // TODO: mikktspace convention is NOT to normalize. we normalize here as khronos sponza iirc?
	
	uint materialID = pc.objectBuffer.objects[drawID].materialID;
	MaterialData m = pc.materialBuffer.materials[materialID];
	
	vec4 albedo = m.baseColorFactor;
	
	if (m.diffuseID != 0)
	{
		albedo.xyz *= textureGrad(sampler2D(textures[m.diffuseID], samplers[LINEAR_SAMPLER]), uv, uvDdx, uvDdy).xyz;
	}
	
	vec3 color = vec3(0.0);
	vec3 ambient = albedo.xyz * 0.1; // when GI is off, likely going to look physically incorrect
	
	vec4 t0 = vec4(mat3(worldMatrix) * tp0.xyz, tp0.w);
	vec4 t1 = vec4(mat3(worldMatrix) * tp1.xyz, tp1.w);
	vec4 t2 = vec4(mat3(worldMatrix) * tp2.xyz, tp2.w);
	vec4 T = interpolate(bary, t0, t1, t2);
	T.xyz = normalize(T.xyz); // TODO: mikktspace convention is NOT to normalize. we normalize here as khronos sponza iirc?
	float sign = T.w; // sign is flipped during tangent generation so mikktspace is consistent with glTF handedness
		
#ifdef PBR
	if (m.normalID != 0)
	{
		vec3 B = sign * cross(N, T.xyz);
		vec3 shadingNormal = textureGrad(sampler2D(textures[m.normalID], samplers[LINEAR_SAMPLER]), uv, uvDdx, uvDdy).xyz;
		shadingNormal = shadingNormal * 2.0 - 1.0;
		N = normalize(shadingNormal.x * T.xyz + shadingNormal.y * B + shadingNormal.z * N);
	}
	
	float metallic = m.metallicFactor;
	float perceptualRoughness = m.roughnessFactor;
	vec2 metalRoughness = vec2(0.0);
	if (m.metalRoughnessID != 0)
	{
		metalRoughness = textureGrad(sampler2D(textures[m.metalRoughnessID], samplers[LINEAR_SAMPLER]), uv, uvDdx, uvDdy).bg;
		metallic *= metalRoughness.x;
		perceptualRoughness *= metalRoughness.y;
	}

	perceptualRoughness = max(perceptualRoughness, 0.045); // frostbite engine clamp value for analytical lights (fp32)
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
	
	vec3 lightColor = sceneData.sunlightColor.xyz;
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

			// try normal schlick and visualize difference, remove comment after
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
	color += vec4(albedo);
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
	
	float depth = texture(sampler2D(textures[pc.depth_id], samplers[NEAREST_SAMPLER]), inUV).r;
	
	if (pc.lightCulling == 1)
	{
		vec2 screenPos = vec2(gl_FragCoord.xy);
		screenPos.y = pc.screenSize.y - screenPos.y;
		
		ivec4 clusterDim = ivec4(pc.clusterSize);
		ivec2 clusterXY = ivec2(floor(screenPos.xy / clusterDim.w));
		clusterXY.x = clamp(clusterXY.x, 0, clusterDim.x - 1);
		clusterXY.y = clamp(clusterXY.y, 0, clusterDim.y - 1);
		
		float viewZ = linearizeDepthInfiniteReverse(depth); // alternative: calculate via matrix multiply, likely more expensive?
		
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
			color += albedo.xyz * lightColor * attenuation * NdotL;
#endif
		}
	}
	
	// TODO: refactor and use depth/stencil buffer to reject pixels in future
	if (depth == 0.0)
	{
		vec2 ndc = inUV * 2.0 - 1.0;
		ndc.y *= -1.0;
		vec3 sampleDir = vec3(sceneData.inverseViewproj * vec4(ndc, 0.0, 1.0));
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
	
	if (pc.debug != 0)
	{
		switch (pc.debug)
		{
		case 1: // geometry normals
			N = N * 0.5 + 0.5;
			N = srgbToLinear(N);
			outFragColor = vec4(N, 1.0);
			break;
		case 2: // tangents
			T.xyz = T.xyz * 0.5 + 0.5;
			T.xyz = srgbToLinear(T.xyz);
			outFragColor = vec4(T.xyz, 1.0);
			break;
		case 3: // tangent w
		    vec3 w = T.w > 0.0 ? vec3(1.0) : vec3(0.0);
		    w = srgbToLinear(w);
		    outFragColor = vec4(w, 1.0);
            break;
		case 4: // uv
			debugUV = srgbToLinear(debugUV);
			outFragColor = vec4(debugUV, 1.0);
			break;
		}
		
		outFragColor = depth != 0.0 ? outFragColor : vec4(0,0,0,1);
	}
}