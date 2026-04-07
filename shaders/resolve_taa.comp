// References
// https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail/
// https://alextardif.com/TAA.html
// https://github.com/playdeadgames/temporal/blob/master/Assets/Shaders/TemporalReprojection.shader
// https://advances.realtimerendering.com/s2014/#_HIGH-QUALITY_TEMPORAL_SUPERSAMPLING
// https://developer.download.nvidia.com/gameworks/events/GDC2016/msalvi_temporal_supersampling.pdf

// note: not optimized, targetting correctness
// TODO: for each tile, extend borders slightly and perform texture load once into shared memory (inc. depth)

#version 460

#extension GL_GOOGLE_include_directive : require

#include "bindings.glsl"
#include "buffer_references.glsl"

layout (local_size_x = 32, local_size_y = 32) in;

layout (push_constant) uniform constants
{
    vec4 jitter_offset;
	vec2 screen_size;
	uint current_id;
	uint history_id;
	uint resolve_id;
	uint depth_id;
	uint velocity_id;
	uint variance_clipping;
	uint history_filter;
	uint local_filter; // unused
	uint ycocg;
	uint valid_history; // valid when first_frame==false
	uint dynamic_blend;
};

// https://github.com/TheRealMJP/MSAAFilter/blob/master/MSAAFilter/Resolve.hlsl
// note: there is some scaling we may have to do
float filter_cubic(float x, float B, float C)
{
    float y = 0.0f;
    float x2 = x * x;
    float x3 = x * x * x;
    if (x < 1)
        y = (12 - 9 * B - 6 * C) * x3 + (-18 + 12 * B + 6 * C) * x2 + (6 - 2 * B);
    else if (x <= 2)
        y = (-B - 6 * C) * x3 + (6 * B + 30 * C) * x2 + (-12 * B - 48 * C) * x + (8 * B + 24 * C);

    return y / 6.0;
}

// note: there is some scaling we may have to do
// filter per axis and multiply their respective weights (see MJP link above),
// or use length? https://github.com/h3r2tic/rtoy-samples/blob/main/assets/shaders/taa.glsl
float filter_mitchell(float value)
{
	return filter_cubic(value, 1.0 / 3.0, 1.0 / 3.0);
}

// TODO: account for exposure: https://www.youtube.com/watch?v=nj4puag4hwc
// we don't want to overdarken highlights
// https://gpuopen.com/learn/optimized-reversible-tonemapper-for-resolve/
vec3 tonemap(vec3 c)
{
    float w = max(c.x, max(c.y, c.z));
    return c / (1.0 + w);
}

vec3 reverse_tonemap(vec3 c)
{
    float w = max(c.x, max(c.y, c.z));
    return c / (1.0 - w);
}

vec3 maybe_rgb_to_ycocg(vec3 c)
{
    if (ycocg == 0)
        return c;

    float y = c.x / 4.0 + c.y / 2.0 + c.z / 4.0;
    float co = c.x / 2.0 - c.z / 2.0;
    float cg = -c.x / 4.0 + c.y / 2.0 - c.z / 4.0;

    return vec3(y, co, cg);
}

// TODO: there are cases where we get black flickers and toggling YCOCG off seem to eliminate them, investigate!
// they were not nans, but pure black pixels according to renderdoc
// update: seem to no longer be able to reproduce this, may be fixed during refactor
vec3 maybe_ycocg_to_rgb(vec3 c)
{
    if (ycocg == 0)
        return c;

    float tmp = c.x - c.z;
    float r = tmp + c.y;
    float g = c.r + c.z;
    float b = tmp - c.y;

    return clamp(vec3(r,g,b), vec3(0.0), vec3(1.0));
}

vec3 sample_color(uint texture_index, uint sampler_index, vec2 uv)
{
    return tonemap(texture(sampler2D(textures[texture_index], samplers[sampler_index]), uv).xyz);
}

float luminance(vec3 c)
{
    return (ycocg == 1) ? c.x : maybe_rgb_to_ycocg(c).x;
}

vec3 reconstruct_world_pos(float depth, mat4 inverse_view_proj, vec2 uv_pos)
{
	vec2 ndc = uv_pos * 2.0 - 1.0;
	ndc.y *= -1.0; // flip as window coords are top down
	vec4 world_pos = inverse_view_proj * vec4(ndc, depth, 1.0);

	return world_pos.xyz / world_pos.w;
}

// 9 tap exploiting bilinear filtering: https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
// for more details: http://vec3.ca/bicubic-filtering-in-fewer-taps/
vec3 sample_texture_catmull_rom(vec2 uv, uint texture_index, vec2 resolution)
{
    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
    // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
    // location [1, 1] in the grid, where [0, 0] is the top left corner.
    vec2 sample_position = uv * resolution;
    vec2 tex_pos_1 = floor(sample_position - 0.5) + 0.5;

    // Compute the fractional offset from our starting texel to our original sample location, which we'll
    // feed into the Catmull-Rom spline function to get our filter weights.
    vec2 f = sample_position - tex_pos_1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);

    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    vec2 w12 = w1 + w2;
    vec2 offset_12 = w2 / w12;

    // Compute the final UV coordinates we'll use for sampling the texture
    vec2 tex_pos_0 = tex_pos_1 - 1;
    vec2 tex_pos_3 = tex_pos_1 + 2;
    vec2 tex_pos_12 = tex_pos_1 + offset_12;

    tex_pos_0 /= resolution;
    tex_pos_3 /= resolution;
    tex_pos_12 /= resolution;

    vec3 result = vec3(0);
    result += tonemap(texture(sampler2D(textures[texture_index], samplers[LINEAR_CLAMP_SAMPLER]), vec2(tex_pos_0.x, tex_pos_0.y)).rgb) * w0.x * w0.y;
    result += tonemap(texture(sampler2D(textures[texture_index], samplers[LINEAR_CLAMP_SAMPLER]), vec2(tex_pos_12.x, tex_pos_0.y)).rgb) * w12.x * w0.y;
    result += tonemap(texture(sampler2D(textures[texture_index], samplers[LINEAR_CLAMP_SAMPLER]), vec2(tex_pos_3.x, tex_pos_0.y)).rgb) * w3.x * w0.y;
    result += tonemap(texture(sampler2D(textures[texture_index], samplers[LINEAR_CLAMP_SAMPLER]), vec2(tex_pos_0.x, tex_pos_12.y)).rgb) * w0.x * w12.y;
    result += tonemap(texture(sampler2D(textures[texture_index], samplers[LINEAR_CLAMP_SAMPLER]), vec2(tex_pos_12.x, tex_pos_12.y)).rgb) * w12.x * w12.y;
    result += tonemap(texture(sampler2D(textures[texture_index], samplers[LINEAR_CLAMP_SAMPLER]), vec2(tex_pos_3.x, tex_pos_12.y)).rgb) * w3.x * w12.y;
    result += tonemap(texture(sampler2D(textures[texture_index], samplers[LINEAR_CLAMP_SAMPLER]), vec2(tex_pos_0.x, tex_pos_3.y)).rgb) * w0.x * w3.y;
    result += tonemap(texture(sampler2D(textures[texture_index], samplers[LINEAR_CLAMP_SAMPLER]), vec2(tex_pos_12.x, tex_pos_3.y)).rgb) * w12.x * w3.y;
    result += tonemap(texture(sampler2D(textures[texture_index], samplers[LINEAR_CLAMP_SAMPLER]), vec2(tex_pos_3.x, tex_pos_3.y)).rgb) * w3.x * w3.y;

    return result;
}

// Optimized ver from Inside/playdeadgames
vec4 clip_aabb(vec3 aabb_min, vec3 aabb_max, vec4 q, float average_alpha)
{
	// note: only clips towards aabb center (but fast!)
	vec3 p_clip = 0.5 * (aabb_max + aabb_min);
	vec3 e_clip = 0.5 * (aabb_max - aabb_min) + 0.00000001;

	vec4 v_clip = q - vec4(p_clip, average_alpha);
	vec3 v_unit = v_clip.xyz / e_clip;
	vec3 a_unit = abs(v_unit);
	float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));

	if (ma_unit > 1.0)
		return vec4(p_clip, average_alpha) + v_clip / ma_unit;
	else
		return q; // point inside aabb
}

void main()
{
    ivec2 texel_coord = ivec2(gl_GlobalInvocationID.xy);

    if (texel_coord.x >= screen_size.x || texel_coord.y >= screen_size.y)
        return;

	vec2 uv = vec2(texel_coord) + vec2(0.5);
	uv /= screen_size;

	// frame 0 copy
	if (valid_history == 0)
	{
		vec3 history_color = texture(sampler2D(textures[current_id], samplers[NEAREST_SAMPLER]), uv).xyz;
		imageStore(images_rgba32f[resolve_id], texel_coord, vec4(history_color, 1.0));
		return;
	}

	// neighbourhood sampling
	// TODO: try mitchell filter
	// note: ignore or accumulate clamped edges for variance clipping?
	vec3 min_color = vec3(99999.0);
	vec3 max_color = vec3(-99999.0);
	vec3 m1 = vec3(0.); // used in variance clipping
	vec3 m2 = vec3(0.); // used in variance clipping
	vec3 current_color;
	for (int x = -1; x <= 1; x++)
	{
	    for (int y = -1; y <= 1; y++)
    	{
            vec3 col = sample_color(current_id, NEAREST_SAMPLER, uv + (vec2(x, y) / screen_size));
            col = maybe_rgb_to_ycocg(col);

            if (x == 0 && y == 0)
                current_color = col;

            // note: when mitchell is implemented, remember to ignore weights when computing min/max_color
           	min_color = min(col, min_color);
            max_color = max(col, max_color);
            m1 += col;
            m2 += (col * col);
    	}
	}

	// depth dilation
	float closest_depth = 0.0;
	vec2 closest_uv_offset;
	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			vec2 uv_offset = vec2(x, y) / screen_size;
			float neighbour_depth = texture(sampler2D(textures[depth_id], samplers[NEAREST_SAMPLER]), uv + uv_offset).r;
			if (neighbour_depth > closest_depth)
			{
				closest_depth = neighbour_depth;
				closest_uv_offset = uv_offset;
			}
		}
	}

	// note: our scene is entirely static now, so we recompute it here
	float current_depth = texture(sampler2D(textures[depth_id], samplers[NEAREST_SAMPLER]), uv).r;
	vec2 current_ndc = uv * 2.0 - 1.0;
	current_ndc.y *= -1.0;
	vec4 world_pos = uniforms.inverse_view_proj * vec4(current_ndc, current_depth, 1.0);
	world_pos /= world_pos.w;
	vec4 previous_clip = uniforms.prev_view_proj * world_pos;
	vec2 previous_ndc = previous_clip.xy / previous_clip.w;
	current_ndc -= jitter_offset.xy;
	previous_ndc -= jitter_offset.zw;
	vec2 camera_velocity = (current_ndc - previous_ndc) * vec2(0.5, -0.5);

	// important! without this, setting blend_factor to 1.0 is not sufficient as nan has propogated to the blending eq
	camera_velocity = current_depth != 0.0 ? camera_velocity : vec2(0.0);

	// TODO: object motion
	// vec2 object_velocity = texture(sampler2D(textures[velocity_id], samplers[NEAREST_SAMPLER]), uv + closest_uv_offset).rg; // uv space
    vec2 reprojected_uv = uv - camera_velocity; // - object_velocity;

	// alternative: catmull 5-taps from COD?
	vec3 history_color;
	if (history_filter == 1)
		history_color = sample_texture_catmull_rom(reprojected_uv, history_id, screen_size);
	else
		history_color = sample_color(history_id, LINEAR_CLAMP_SAMPLER, reprojected_uv);

	history_color = maybe_rgb_to_ycocg(history_color);

	// https://developer.download.nvidia.com/gameworks/events/GDC2016/msalvi_temporal_supersampling.pdf
	vec3 history_color_clamped;
	if (variance_clipping == 1)
	{
		float rcp_samples = 1.0 / 9.0;
		// large gamma - temporally stable results at the cost of increased ghosting; small gamma - lose ability to integrate data over time
		float gamma = 1.0;
		vec3 mu = m1 * rcp_samples;
		vec3 sigma = sqrt(abs((m2 * rcp_samples) - (mu * mu)));
		vec3 minc = mu - gamma * sigma;
		vec3 maxc = mu + gamma * sigma;
		history_color_clamped = clip_aabb(minc, maxc, vec4(clamp(history_color, min_color, max_color), 1.0), 1.0).xyz;
	}
	else
	{
	    // should not be disabled; neighbourhood clamping trades off ghosting for flickering - turn this off to see without flickering
		history_color_clamped = history_color;
	}

	float blend_factor = 0.05;

	// TODO: explore other blending heuristics; a straightforward one is when reprojected uv is OOB
	// https://www.youtube.com/watch?v=WzpLWzGvFK4&t=18m
	if (dynamic_blend == 1)
	{
    	float min_blend_factor = 0.03;
    	float max_blend_factor = 0.12;

        float luma_history = luminance(history_color_clamped);
        float luma_current = luminance(current_color);

        float diff = 1.0 - abs(luma_history - luma_current) / max(max(luma_history, luma_current), 0.2);
        float w = diff * diff; // w == 0, low similarity -> lower history contribution; w == 1, high similarity -> higher history contribution

        blend_factor = max_blend_factor + (min_blend_factor - max_blend_factor) * w;
	}

	vec3 final_color = history_color_clamped + (current_color - history_color_clamped) * blend_factor;

	final_color = maybe_ycocg_to_rgb(final_color);
	final_color = reverse_tonemap(final_color);

	// TODO: we can optimize by storing tonemapped value for next frame's history, skipping tonemapping for reprojected colors
	// but this is an inconvenience for experimenting with other post-fx for now, so we store reverse tonemapped values
	imageStore(images_rgba32f[resolve_id], texel_coord, vec4(final_color, 1.0));
}
