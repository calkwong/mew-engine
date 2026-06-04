import bindings;

// warning: untested after porting to slang
// assumes we have 256 bins

struct PushConstants
{
	uint* luminance_buffer;
	float* luminance_avg_buffer;
	float2 screen_size;
	uint image_id; // HDR color attachment
	float min_log_luminance;
	float one_over_log_luminance_range;
	uint pixel_count;
	float tau;
	float delta_time;
};

[[vk::push_constant]] PushConstants pc;

groupshared uint shared_bins[256];

float get_luminance(float3 color)
{
	return dot(color, float3(0.2125, 0.7154, 0.0721));
}

uint color_to_bin(float3 color)
{
	float luminance = get_luminance(color);

	if (luminance < 0.005)
		return 0;

	float log_luminance = clamp((log2(luminance) - pc.min_log_luminance) * pc.one_over_log_luminance_range, 0.0, 1.0); // in range [0,254]
	return uint(log_luminance * 254.0 + 1.0); // map to [1,255]
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
    uint t_index : SV_GroupIndex,
    uint3 global_id : SV_DispatchThreadID
)
{
	shared_bins[t_index] = 0;

	GroupMemoryBarrierWithGroupSync();

	int2 texel_coord = int2(global_id.xy);
	if (texel_coord.x < pc.screen_size.x && texel_coord.y < pc.screen_size.y)
	{
	    RWTexture2D tex = get_storage_image<RWTexture2D>(pc.image_id);
		float3 color = tex.Load(texel_coord).rgb;

		uint bin = color_to_bin(color);

		InterlockedAdd(shared_bins[bin], 1);
	}

	GroupMemoryBarrierWithGroupSync();
	InterlockedAdd(pc.luminance_buffer[t_index], shared_bins[t_index]);
}
