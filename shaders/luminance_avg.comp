#version 460

#extension GL_GOOGLE_include_directive : require

#include "bindings.glsl"
#include "buffer_references.glsl"

layout (local_size_x = 16, local_size_y = 16) in;

layout( push_constant ) uniform constants
{
	LuminanceBuffer luminance_buffer;
	LuminanceAvgBuffer luminance_avg_buffer;
	vec2 screen_size;
	uint image_id;
	float min_log_luminance;
	float one_over_log_luminance_range;
	uint pixel_count;
	float tau;
	float delta_time;
};

shared uint shared_bins[256];

void main()
{
	uint thread_index = gl_LocalInvocationIndex;

	uint count_for_this_bin = luminance_buffer.bins[thread_index];
	shared_bins[thread_index] = count_for_this_bin * thread_index; // bin 0 is set to 0, intentional?

	barrier();

	// reset buffer
	luminance_buffer.bins[thread_index] = 0;

	// TODO: reconsider placing in a for loop, dont think we actually have UB issue here
	// manual unrolling to prevent barrier UB
	if (thread_index < 128)
		shared_bins[thread_index] += shared_bins[thread_index + 128];
	barrier();
	if (thread_index < 64)
		shared_bins[thread_index] += shared_bins[thread_index + 64];
	barrier();
	if (thread_index < 32)
		shared_bins[thread_index] += shared_bins[thread_index + 32];
	barrier();
	if (thread_index < 16)
		shared_bins[thread_index] += shared_bins[thread_index + 16];
	barrier();
	if (thread_index < 8)
		shared_bins[thread_index] += shared_bins[thread_index + 8];
	barrier();
	if (thread_index < 4)
		shared_bins[thread_index] += shared_bins[thread_index + 4];
	barrier();
	if (thread_index < 2)
		shared_bins[thread_index] += shared_bins[thread_index + 2];
	barrier();
	if (thread_index < 1)
		shared_bins[thread_index] += shared_bins[thread_index + 1];
	barrier();

	if (thread_index == 0)
	{
		// reversing colorToBin from luminance_histogram.comp
		// weightedAverage for black pixels were effectively 0 as we multiplied black pixel count by thread_index 0; below compensates for it
		float weighted_avg_log = (float(shared_bins[0]) / max(float(pixel_count) - count_for_this_bin, 1.0)) - 1.0;

		// map from histogram to luminance
		float weighted_avg_luminance = exp2(((weighted_avg_log / 254.0) * 1.0 / one_over_log_luminance_range) + min_log_luminance);

		float last_luminance_avg = luminance_avg_buffer.luminance_avg;
		float adapted_luminance = last_luminance_avg + (weighted_avg_luminance - last_luminance_avg) * (1.0 - exp(-delta_time * tau));
		luminance_avg_buffer.luminance_avg = adapted_luminance;
	}
}
