// References
// https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail/
// https://alextardif.com/TAA.html
// https://github.com/playdeadgames/temporal/blob/master/Assets/Shaders/TemporalReprojection.shader
// https://advances.realtimerendering.com/s2014/#_HIGH-QUALITY_TEMPORAL_SUPERSAMPLING
// https://developer.download.nvidia.com/gameworks/events/GDC2016/msalvi_temporal_supersampling.pdf

#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "scene.glsl"
#include "samplers.glsl"

layout(set = 1, binding = 0) uniform texture2D allTextures[];
layout(set = 2, binding = 0) uniform sampler samplers[];

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

layout( push_constant ) uniform constants
{
	vec2 screenSize;
	vec2 currentJitter;
	uint color_id;
	uint accum_id;
	uint depth_id;
	uint velocity_id;
	uint variance_clipping;
	uint history_filter;
	uint local_filter;
	uint ycocg;
	uint depth_dilation;
} pc;

// https://github.com/TheRealMJP/MSAAFilter/blob/master/MSAAFilter/Resolve.hlsl
float filterCubic(float x, float B, float C)
{
    float y = 0.0f;
    float x2 = x * x;
    float x3 = x * x * x;
    if (x < 1)
        y = (12 - 9 * B - 6 * C) * x3 + (-18 + 12 * B + 6 * C) * x2 + (6 - 2 * B);
    else if (x <= 2)
        y = (-B - 6 * C) * x3 + (6 * B + 30 * C) * x2 + (-12 * B - 48 * C) * x + (8 * B + 24 * C);

    return y / 6.0f;
}

float filterMitchell(float value)
{
	return filterCubic(value, 1.0 / 3.0, 1.0 / 3.0);
}

vec3 rgb_to_ycocg(vec3 c)
{
    float y = c.x / 4.0 + c.y / 2.0 + c.z / 4.0;
    float co = c.x / 2.0 - c.z / 2.0;
    float cg = -c.x / 4.0 + c.y / 2.0 - c.z / 4.0;

    return vec3(y, co, cg);
}

vec3 ycocg_to_rgb(vec3 c)
{
    float tmp = c.x - c.z;
    float r = tmp + c.y;
    float g = c.r + c.z;
    float b = tmp - c.y;

    return clamp(vec3(r,g,b), vec3(0.0), vec3(1.0));
}

vec3 sample_color(uint texture_index, uint sampler_index, vec2 uv)
{
    vec3 color = texture(sampler2D(allTextures[texture_index], samplers[sampler_index]), uv).xyz;

    if (pc.ycocg == 1)
        color = rgb_to_ycocg(color);

    return color;
}

// Samples a texture with Catmull-Rom filtering, using 9 texture fetches instead of 16.
// See http://vec3.ca/bicubic-filtering-in-fewer-taps/ for more details
vec3 sample_texture_catmull_rom(vec2 uv, uint texture_index, vec2 resolution) {
    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
    // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
    // location [1, 1] in the grid, where [0, 0] is the top left corner.
    vec2 sample_position = uv * resolution;
    vec2 tex_pos_1 = floor(sample_position - 0.5f) + 0.5f;

    // Compute the fractional offset from our starting texel to our original sample location, which we'll
    // feed into the Catmull-Rom spline function to get our filter weights.
    vec2 f = sample_position - tex_pos_1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    vec2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    vec2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    vec2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    vec2 w3 = f * f * (-0.5f + 0.5f * f);

    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    vec2 w12 = w1 + w2;
    vec2 offset_12 = w2 / (w1 + w2);

    // Compute the final UV coordinates we'll use for sampling the texture
    vec2 tex_pos_0 = tex_pos_1 - 1;
    vec2 tex_pos_3 = tex_pos_1 + 2;
    vec2 tex_pos_12 = tex_pos_1 + offset_12;

    tex_pos_0 /= resolution;
    tex_pos_3 /= resolution;
    tex_pos_12 /= resolution;

    vec3 result = vec3(0);
    result += texture(sampler2D(allTextures[texture_index], samplers[LINEAR_SAMPLER]), vec2(tex_pos_0.x, tex_pos_0.y)).rgb * w0.x * w0.y;
    result += texture(sampler2D(allTextures[texture_index], samplers[LINEAR_SAMPLER]), vec2(tex_pos_12.x, tex_pos_0.y)).rgb * w12.x * w0.y;
    result += texture(sampler2D(allTextures[texture_index], samplers[LINEAR_SAMPLER]), vec2(tex_pos_3.x, tex_pos_0.y)).rgb * w3.x * w0.y;
    result += texture(sampler2D(allTextures[texture_index], samplers[LINEAR_SAMPLER]), vec2(tex_pos_0.x, tex_pos_12.y)).rgb * w0.x * w12.y;
    result += texture(sampler2D(allTextures[texture_index], samplers[LINEAR_SAMPLER]), vec2(tex_pos_12.x, tex_pos_12.y)).rgb * w12.x * w12.y;
    result += texture(sampler2D(allTextures[texture_index], samplers[LINEAR_SAMPLER]), vec2(tex_pos_3.x, tex_pos_12.y)).rgb * w3.x * w12.y;
    result += texture(sampler2D(allTextures[texture_index], samplers[LINEAR_SAMPLER]), vec2(tex_pos_0.x, tex_pos_3.y)).rgb * w0.x * w3.y;
    result += texture(sampler2D(allTextures[texture_index], samplers[LINEAR_SAMPLER]), vec2(tex_pos_12.x, tex_pos_3.y)).rgb * w12.x * w3.y;
    result += texture(sampler2D(allTextures[texture_index], samplers[LINEAR_SAMPLER]), vec2(tex_pos_3.x, tex_pos_3.y)).rgb * w3.x * w3.y;

    if (pc.ycocg == 1)
        result = rgb_to_ycocg(result);

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
	vec2 uv = inUV;
	
	// depth dilation
	vec2 velocityUV;
	if (pc.depth_dilation == 1)
	{
		float closestDepth = 0.0;
		vec2 closestUVOffset;
		for (int x = -1; x <= 1; x++)
		{
			for (int y = -1; y <= 1; y++)
			{
				vec2 uvOffset = vec2(x, y) / pc.screenSize;
				float neighbourDepth = texture(sampler2D(allTextures[pc.depth_id], samplers[NEAREST_SAMPLER]), uv + uvOffset).r;
				if (neighbourDepth > closestDepth)
				{
					closestDepth = neighbourDepth;
					closestUVOffset = uvOffset;
				}
			}
		}
	
		velocityUV = texture(sampler2D(allTextures[pc.velocity_id], samplers[NEAREST_SAMPLER]), uv + closestUVOffset).rg;
	}
	else
	{
		velocityUV = texture(sampler2D(allTextures[pc.velocity_id], samplers[NEAREST_SAMPLER]), uv).rg; 
	}
	
	// alternative: blend this with a 5 taps '+' pattern per Karis UE4
	// no need max(sample, 0.0) to ensure no garbage values?
	vec3 ctl = sample_color(pc.color_id, NEAREST_SAMPLER, uv + (vec2(-1, 1) / pc.screenSize));
	vec3 ctc = sample_color(pc.color_id, NEAREST_SAMPLER, uv + (vec2( 0, 1) / pc.screenSize));
	vec3 ctr = sample_color(pc.color_id, NEAREST_SAMPLER, uv + (vec2( 1, 1) / pc.screenSize));
	vec3 cml = sample_color(pc.color_id, NEAREST_SAMPLER, uv + (vec2(-1, 0) / pc.screenSize));
	vec3 cmc = sample_color(pc.color_id, NEAREST_SAMPLER, uv + (vec2( 0, 0) / pc.screenSize));
	vec3 cmr = sample_color(pc.color_id, NEAREST_SAMPLER, uv + (vec2( 1, 0) / pc.screenSize));
	vec3 cbl = sample_color(pc.color_id, NEAREST_SAMPLER, uv + (vec2(-1,-1) / pc.screenSize));
	vec3 cbc = sample_color(pc.color_id, NEAREST_SAMPLER, uv + (vec2( 0,-1) / pc.screenSize));
	vec3 cbr = sample_color(pc.color_id, NEAREST_SAMPLER, uv + (vec2( 1,-1) / pc.screenSize));
	vec3 minColor = vec3(9999.0);
	vec3 maxColor = vec3(-9999.0);
	minColor = min(ctl, min(ctc, min(ctr, min(cml, min(cmc, min(cmr, min(cbl, min(cbc, cbr))))))));
	maxColor = max(ctl, max(ctc, max(ctr, max(cml, max(cmc, max(cmr, max(cbl, max(cbc, cbr))))))));

	// alternative: blackman, mitchell filter or average/weighted neighbourhood?
    vec3 currentColor = cmc;

	// alternative: catmull 5-taps from COD?
	vec2 reprojectedUV = uv - velocityUV;
	vec3 previousColor;
	if (pc.history_filter == 1)
		previousColor = sample_texture_catmull_rom(reprojectedUV, pc.accum_id, pc.screenSize);
	else
		previousColor = sample_color(pc.accum_id, LINEAR_SAMPLER, reprojectedUV);
	vec3 previousColorClamped;
	
	// https://developer.download.nvidia.com/gameworks/events/GDC2016/msalvi_temporal_supersampling.pdf
	if (pc.variance_clipping == 1)
	{
		vec3 m1 = ctl + ctc + ctr + cml + cmc + cmr + cbl + cbc + cbr;
		vec3 m2 = ctl * ctl + ctc * ctc + ctr * ctr + cml * cml + cmc * cmc + cmr * cmr + cbl * cbl + cbc * cbc + cbr * cbr;

		float rcpSamples = 1.0 / 9.0;
		// large gamma - temporally stable results at the cost of increased ghosting; small gamma - lose ability to integrate data over time
		float gamma = 1.0;
		vec3 mu = m1 * rcpSamples;
		vec3 sigma = sqrt(abs((m2 * rcpSamples) - (mu * mu)));
		vec3 minc = mu - gamma * sigma;
		vec3 maxc = mu + gamma * sigma;
		previousColorClamped = clip_aabb(minc, maxc, vec4(clamp(previousColor, minColor, maxColor), 1.0), 1.0).xyz; 
	}
	else
	{
		previousColorClamped = clamp(previousColor, minColor, maxColor);
	}

    // TODO: fix flickering

	vec3 finalColor = currentColor * 0.1 + previousColorClamped * 0.9;

	if (pc.ycocg == 1)
	    finalColor = ycocg_to_rgb(finalColor);

	outFragColor = vec4(finalColor, 1.0);
}