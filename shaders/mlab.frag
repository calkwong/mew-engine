#version 450

// References
// https://dl.acm.org/doi/epdf/10.1145/2556700.2556705
// https://github.com/Devsh-Graphics-Programming/Nabla/blob/master/include/nbl/builtin/glsl/ext/OIT/insert_node.glsl
// https://interplayoflight.wordpress.com/2022/07/02/order-independent-transparency-part-2/

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_ARB_fragment_shader_interlock : require

#include "scene.glsl"
#include "samplers.glsl"
#include "mesh.glsl"

layout(set = 1, binding = 0) uniform texture2D allTextures[];
layout(set = 2, binding = 0) uniform sampler samplers[];

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec4 inTangent;
layout (location = 3) in flat uint inMaterialID;

layout (location = 0) out vec4 outFragColor;

layout(early_fragment_tests) in; // REQUIRED
layout(pixel_interlock_ordered) in; // seems to work even without, not sure if spec mandates this qualifier

const int MLAB_NODES = 4;

struct OITData
{
	uvec4 colors;
	uvec4 depths;
	vec4 transmissions;
};

layout(buffer_reference, std430) coherent buffer OITBuffer // REQUIRED to prevent WAW 
{ 
	OITData frags[];
};

layout(buffer_reference, std430) readonly buffer MaterialBuffer
{ 
	MaterialData materials[];
};

layout( push_constant ) uniform constants
{
	//ObjectBuffer objectBuffer;
	//VertexBuffer vertexBuffer;
	//MeshTaskBuffer meshTaskBuffer;
	//MeshletBuffer meshletBuffer;
	//MeshletIndicesBuffer meshletIndicesBuffer;
	//ClusterIndicesBuffer clusterIndicesBuffer; 
	uint padding[6 * 2];
	MaterialBuffer materialBuffer;
	OITBuffer oitBuffer;
	uint debugMeshlets;
} pc;

void swapNode(inout uint colorA, inout uint depthA, inout float transmissionA, inout uint colorB, inout uint depthB, inout float transmissionB)
{
	uint colorT = colorA;
	uint depthT = depthA;
	float transmissionT = transmissionA;
	
	colorA = colorB;
	depthA = depthB;
	transmissionA = transmissionB;
	
	colorB = colorT;
	depthB = depthT;
	transmissionB = transmissionT;
}

uint float2UintDepth(float depth)
{
	return uint(depth * 4294967295.0);
}

float Uint2FloatDepth(uint depth)
{
	return float(depth) / 4294967295.0;
}

void insertNode(uvec2 coords, uint color, uint depth, float transmission)
{
	uint index = coords.x + coords.y * 1700;

	OITData frag = pc.oitBuffer.frags[index];
	uvec4 colors = frag.colors;
	uvec4 depths = frag.depths;
	vec4 transmissions = frag.transmissions;
	
	bvec4 notValidMask = equal(transmissions, vec4(1.0));
	depths = floatBitsToUint(mix(uintBitsToFloat(depths), vec4(0.0), notValidMask)); // reverse z, selects a if false
	bvec4 closerMask = greaterThanEqual(uvec4(depth), depths); // potential precision issue?
	
	// swap nodes if closer
	if (closerMask[0])
		swapNode(colors[0], depths[0], transmissions[0], color, depth, transmission);
	if (closerMask[1])
		swapNode(colors[1], depths[1], transmissions[1], color, depth, transmission);
	if (closerMask[2])
		swapNode(colors[2], depths[2], transmissions[2], color, depth, transmission);
	if (closerMask[3])
		swapNode(colors[3], depths[3], transmissions[3], color, depth, transmission);
	
	// compress array if overflow
	if (!notValidMask[MLAB_NODES-1])
	{
		colors[MLAB_NODES-1] = packUnorm4x8(unpackUnorm4x8(colors[MLAB_NODES-1]) + unpackUnorm4x8(color) * transmissions[MLAB_NODES-1]);
		transmissions[MLAB_NODES-1] *= transmission;
	}
	
	// store
	pc.oitBuffer.frags[index].colors = colors;
	pc.oitBuffer.frags[index].depths = depths;
	pc.oitBuffer.frags[index].transmissions = transmissions;
}

void main()
{
	// sample pixel color
	MaterialData m = pc.materialBuffer.materials[inMaterialID];
	
	vec4 albedo = m.baseColorFactor;
	if (m.diffuseID != 0)
		albedo *= texture(sampler2D(allTextures[m.diffuseID], samplers[0]), inUV);
	
	albedo.a = 0.5; // TODO: remove, just for testing
	vec4 premultipliedAlpha = vec4(albedo.xyz * albedo.a, 1.0);
	uint color = packUnorm4x8(premultipliedAlpha);
	uint depth = floatBitsToUint(gl_FragCoord.z);
	float transmission = 1.0 - albedo.a;
	
	uvec2 screenCoords = uvec2(floor(gl_FragCoord.xy));
	
	beginInvocationInterlockARB();
	insertNode(screenCoords, color, depth, transmission);
	endInvocationInterlockARB();
}