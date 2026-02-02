#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_ARB_fragment_shader_interlock : require

// include interlock

#include "scene.glsl"
#include "samplers.glsl"
#include "mesh.glsl"

layout(set = 1, binding = 0) uniform texture2D allTextures[];
layout(set = 2, binding = 0) uniform sampler samplers[];

// TODO: clean up - lots of redundant interpolants based on old pbr code that's been put aside
layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inWorldPos;
layout (location = 2) in vec3 inViewPos;
layout (location = 3) in vec2 inUV;
layout (location = 4) in vec4 inTangent;
layout (location = 5) in flat uint inMaterialID;

layout (location = 0) out vec4 outFragColor;

const int MLAB_NODES = 4;

struct OITData
{
	uvec4 colors;
	vec4 depths;
	vec4 transmissions;
};

layout(buffer_reference, std430) readonly buffer MaterialBuffer
{ 
	MaterialData materials[];
};

layout(buffer_reference, std430) readonly buffer OITBuffer
{ 
	OITData frags[];
};

layout( push_constant ) uniform constants
{
	MaterialBuffer materialBuffer;
	OITBuffer oitBuffer;
	uint albedoID;
	
} pc;

void swapNode(inout uint colorA, inout float depthA, inout float transmissionA, inout uint colorB, inout float depthB, inout float transmissionB)
{
	uint colorT = colorA;
	float depthT = depthA;
	float transmissionT = transmissionA;
	
	colorA = colorB;
	depthA = depthB;
	transmissionA = transmissionB;
	
	colorB = colorT;
	depthB = depthT;
	transmissionB = transmissionT;
}

void insertNode(uvec2 coords, uint color, float depth, float transmission)
{
	uint index = coords.x + coords.y * coords.x;

	OITData frag = pc.oitBuffer.frags[index];
	uvec4 colors = frag.colors;
	vec4 depths = frag.depths;
	vec4 transmissions = frag.transmissions;
	
	bvec4 notValidMask = equal(transmissions, vec4(1.0));
	vec4 depthMask = mix(depths, vec4(0.0), notValidMask); // reverse z, selects a if false
	bvec4 closerMask = greaterThanEqual(vec4(depth), depthMask);
	
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
	
	// do we need a barrier or interlock has implicit guarantees?
}

void main()
{
	// sample pixel color
	MaterialData m = pc.materialBuffer.materials[inMaterialID];
	
	vec4 albedo = m.baseColorFactor;
	if (m.diffuseID != 0)
		albedo *= texture(sampler2D(allTextures[m.diffuseID], samplers[0]), inUV);
	
	// sample depth
	vec4 premultipliedAlpha = vec4(albedo.xyz * albedo.a, albedo.a);
	uint color = packUnorm4x8(premultipliedAlpha);
	float depth = gl_FragCoord.z;
	float transmission = 1.0 - albedo.a;
	
	uvec2 screenCoords = uvec2(gl_FragCoord.xy - vec2(0.5));
	
	beginInvocationInterlockARB();
	insertNode(screenCoords, color, depth, transmission);
	endInvocationInterlockARB();
}