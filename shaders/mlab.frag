#version 450

// References
// https://dl.acm.org/doi/epdf/10.1145/2556700.2556705
// https://github.com/Devsh-Graphics-Programming/Nabla/blob/master/include/nbl/builtin/glsl/ext/OIT/insert_node.glsl
// https://interplayoflight.wordpress.com/2022/07/02/order-independent-transparency-part-2/

#extension GL_GOOGLE_include_directive : require
#extension GL_ARB_fragment_shader_interlock : require

#include "bindings.glsl"
#include "buffer_references.glsl"

layout (location = 0) in vec3 in_normal;
layout (location = 1) in vec2 in_uv;
layout (location = 2) in vec4 in_tangent;
layout (location = 3) in flat uint in_material_id;

layout (location = 0) out vec4 out_color;

layout(early_fragment_tests) in; // REQUIRED
layout(pixel_interlock_ordered) in; // seems to work even without, not sure if spec mandates this qualifier

const int MLAB_NODES = 4;

layout( push_constant ) uniform constants
{
	ObjectBuffer object_buffer;
	VertexBuffer vertex_buffer;
	MeshTaskBuffer mesh_task_buffer;
	MeshletBuffer meshlet_buffer;
	MeshletIndicesBuffer meshlet_indices_buffer;
	ClusterIndicesBuffer cluster_indices_buffer; 
	MaterialBuffer material_buffer;
	OITBuffer oit_buffer;
	uint debug_meshlets;
};

void swap_node(inout uint color_a, inout uint depth_a, inout float transmission_a, inout uint color_b, inout uint depth_b, inout float transmission_b)
{
	uint colorT = color_a;
	uint depthT = depth_a;
	float transmissionT = transmission_a;
	
	color_a = color_b;
	depth_a = depth_b;
	transmission_a = transmission_b;
	
	color_b = colorT;
	depth_b = depthT;
	transmission_b = transmissionT;
}

void insert_node(uvec2 coords, uint color, uint depth, float transmission)
{
	uint index = coords.x + coords.y * 1700;

	OITData frag = oit_buffer.frags[index];
	uvec4 colors = frag.colors;
	uvec4 depths = frag.depths;
	vec4 transmissions = frag.transmissions;

	bvec4 not_valid_mask = equal(transmissions, vec4(1.0));
	depths = floatBitsToUint(mix(uintBitsToFloat(depths), vec4(0.0), not_valid_mask)); // reverse z, selects a if false
	bvec4 closer_mask = greaterThanEqual(uvec4(depth), depths); // potential precision issue?

	// swap nodes if closer
	if (closer_mask[0])
		swap_node(colors[0], depths[0], transmissions[0], color, depth, transmission);
	if (closer_mask[1])
		swap_node(colors[1], depths[1], transmissions[1], color, depth, transmission);
	if (closer_mask[2])
		swap_node(colors[2], depths[2], transmissions[2], color, depth, transmission);
	if (closer_mask[3])
		swap_node(colors[3], depths[3], transmissions[3], color, depth, transmission);

	// compress array if overflow
	if (!not_valid_mask[MLAB_NODES-1])
	{
		colors[MLAB_NODES-1] = packUnorm4x8(unpackUnorm4x8(colors[MLAB_NODES-1]) + unpackUnorm4x8(color) * transmissions[MLAB_NODES-1]);
		transmissions[MLAB_NODES-1] *= transmission;
	}

	// store
	oit_buffer.frags[index].colors = colors;
	oit_buffer.frags[index].depths = depths;
	oit_buffer.frags[index].transmissions = transmissions;
}

void main()
{
	// sample pixel color
	MaterialData m = material_buffer.materials[in_material_id];

	vec4 albedo = m.base_color_factor;
	if (m.diffuse_id != 0)
		albedo *= texture(sampler2D(textures[m.diffuse_id], samplers[0]), in_uv);

	albedo.a = 0.5; // TODO: remove, just for testing
	vec4 premultiplied_alpha = vec4(albedo.xyz * albedo.a, 1.0);
	uint color = packUnorm4x8(premultiplied_alpha);
	uint depth = floatBitsToUint(gl_FragCoord.z);
	float transmission = 1.0 - albedo.a;

	uvec2 screen_coords = uvec2(floor(gl_FragCoord.xy));

	// the same pixel cannot enter critical section; neighbouring pixels ok
	beginInvocationInterlockARB();
	insert_node(screen_coords, color, depth, transmission);
	endInvocationInterlockARB();
}