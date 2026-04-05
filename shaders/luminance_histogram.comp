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
	uint image_id; // HDR color attachment
	float min_log_luminance;
	float one_over_log_luminance_range;
	uint pixel_count;
	float tau;
	float delta_time;
};

shared uint shared_bins[256];

float get_luminance(vec3 color)
{
	return dot(color, vec3(0.2125, 0.7154, 0.0721));
}

uint color_to_bin(vec3 color)
{
	float luminance = get_luminance(color);

	if (luminance < 0.005)
		return 0;

	float log_luminance = clamp((log2(luminance) - min_log_luminance) * one_over_log_luminance_range, 0.0, 1.0); // in range [0,254]
	return uint(log_luminance * 254.0 + 1.0); // map to [1,255]
}

void main()
{
	shared_bins[gl_LocalInvocationIndex] = 0;

	barrier();

	ivec2 texel_coord = ivec2(gl_GlobalInvocationID.xy);
	if (texel_coord.x < screen_size.x && texel_coord.y < screen_size.y)
	{
		vec3 color = imageLoad(images_rgba32f[image_id], texel_coord).rgb;

		uint bin = color_to_bin(color);

		atomicAdd(shared_bins[bin], 1);
	}

	barrier();
	atomicAdd(luminance_buffer.bins[gl_LocalInvocationIndex], shared_bins[gl_LocalInvocationIndex]);
}
