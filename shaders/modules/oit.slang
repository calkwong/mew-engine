// implements Multi-Layer Alpha Blending

// References
// https://dl.acm.org/doi/epdf/10.1145/2556700.2556705
// https://github.com/Devsh-Graphics-Programming/Nabla/blob/master/include/nbl/builtin/glsl/ext/OIT/insert_node.glsl
// https://interplayoflight.wordpress.com/2022/07/02/order-independent-transparency-part-2/

module oit;

internal static const int MLAB_NODES = 4;

public struct OITData
{
	public uint4 colors;
	public uint4 depths;
	public float4 transmissions; // could we pack this in color.a?
};

public void swap_node(inout uint color_a, inout uint depth_a, inout float transmission_a, inout uint color_b, inout uint depth_b, inout float transmission_b)
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

public void insert_node(inout OITData fragment, uint color, uint depth, float transmission)
{
	uint4 colors = fragment.colors;
	uint4 depths = fragment.depths;
	float4 transmissions = fragment.transmissions;

	bool4 not_valid_bool = transmissions == float4(1.0);

	float4 not_valid_mask = select(not_valid_bool, float4(1.0), float4(0.0));
	depths = asuint(lerp(asfloat(depths), float4(0.0), not_valid_mask)); // reverse z
	bool4 closer_mask = uint4(depth) >= depths; // potential precision issue?

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
	if (!not_valid_bool[MLAB_NODES-1])
	{
		colors[MLAB_NODES-1] = packUnorm4x8(unpackUnorm4x8ToFloat(colors[MLAB_NODES-1]) + unpackUnorm4x8ToFloat(color) * transmissions[MLAB_NODES-1]);
		transmissions[MLAB_NODES-1] *= transmission;
	}

	OITData oit_data;
	oit_data.colors = colors;
	oit_data.depths = depths;
	oit_data.transmissions = transmissions;

	fragment = oit_data;
}

public float3 composite_transparent(OITData* oit_buffer, uint index, float3 input_color, uint2 frag_coord, inout bool update)
{
	float3 color = input_color;

	OITData frags = oit_buffer[index];

	// early return if nothing stored
	if (frags.transmissions[0] == 1.0)
	{
		return color;
	}

	oit_buffer[index].transmissions = float4(1.0); // reset so we can skip vkcmdfillbuffer

	float3 composite = float3(0.);
	float t_accum = 1.0;
	for (int i = 0; i < MLAB_NODES; i++)
	{
		float t = frags.transmissions[i];
		composite = t != 1.0 ? float3(unpackUnorm4x8ToFloat(frags.colors[i]).xyz) * t_accum + composite : composite;
		t_accum *= t;
	}

	color *= t_accum;
	color += composite;

	update = true;

	return color;
}
