import bindings;

// warning: untested after porting to slang
// assumes we have 256 bins

struct PushConstants
{
	uint* luminance_buffer; // size: 256
	float* luminance_avg_buffer;
	float2 screen_size;
	uint image_id;
	float min_log_luminance;
	float one_over_log_luminance_range;
	uint pixel_count;
	float tau;
	float delta_time;
};

[[vk::push_constant]] PushConstants pc;

groupshared uint shared_bins[256];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
    uint t_index : SV_GroupIndex
)
{
	uint count_for_this_bin = pc.luminance_buffer[t_index];
	shared_bins[t_index] = count_for_this_bin * t_index; // bin 0 is set to 0, intentional?

	GroupMemoryBarrierWithGroupSync();

	// reset buffer
	pc.luminance_buffer[t_index] = 0;

	// TODO: reconsider placing in a for loop, dont think we actually have UB issue here
	// manual unrolling to prevent GroupMemoryBarrierWithGroupSync UB
	if (t_index < 128)
		shared_bins[t_index] += shared_bins[t_index + 128];
	GroupMemoryBarrierWithGroupSync();
	if (t_index < 64)
		shared_bins[t_index] += shared_bins[t_index + 64];
	GroupMemoryBarrierWithGroupSync();
	if (t_index < 32)
		shared_bins[t_index] += shared_bins[t_index + 32];
	GroupMemoryBarrierWithGroupSync();
	if (t_index < 16)
		shared_bins[t_index] += shared_bins[t_index + 16];
	GroupMemoryBarrierWithGroupSync();
	if (t_index < 8)
		shared_bins[t_index] += shared_bins[t_index + 8];
	GroupMemoryBarrierWithGroupSync();
	if (t_index < 4)
		shared_bins[t_index] += shared_bins[t_index + 4];
	GroupMemoryBarrierWithGroupSync();
	if (t_index < 2)
		shared_bins[t_index] += shared_bins[t_index + 2];
	GroupMemoryBarrierWithGroupSync();
	if (t_index < 1)
		shared_bins[t_index] += shared_bins[t_index + 1];
	GroupMemoryBarrierWithGroupSync();

	if (t_index == 0)
	{
		// reversing colorToBin from luminance_histogram.comp
		// weightedAverage for black pixels were effectively 0 as we multiplied black pixel count by t_index 0; below compensates for it
		float weighted_avg_log = (float(shared_bins[0]) / max(float(pc.pixel_count) - count_for_this_bin, 1.0)) - 1.0;

		// map from histogram to luminance
		float weighted_avg_luminance = exp2(((weighted_avg_log / 254.0) * 1.0 / pc.one_over_log_luminance_range) + pc.min_log_luminance);

		float last_luminance_avg = *pc.luminance_avg_buffer;
		float adapted_luminance = last_luminance_avg + (weighted_avg_luminance - last_luminance_avg) * (1.0 - exp(-pc.delta_time * pc.tau));
		*pc.luminance_avg_buffer = adapted_luminance;
	}
}
