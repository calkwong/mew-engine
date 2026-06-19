// currently unused and may be incompatible with current build
// will refactor if temporal filter is its own pass

import bindings;
import volumetrics;

struct PushConstants
{
    float4x4 inverse_view_proj;
    float4x4 previous_view_proj;

    uint3 froxel_dims;
    uint first_frame;

    float near;
    float far;
    uint history_tex;
    uint input_tex;
    uint output_tex;
    float volumetrics_scale;
    float volumetrics_bias;
    float reprojection_factor;

    float2 halton;
    uint blue_noise_tex;
    uint current_frame;
};

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
    uint3 global_id: SV_DispatchThreadID
)
{
    uint3 froxel_coord = global_id;

    if (any(froxel_coord >= pc.froxel_dims))
        return;

    // TODO: use correct sampler
    SamplerState samp = get_sampler<SamplerState>(LINEAR_CLAMP_SAMPLER);
    Texture3D<float4> history_tex = get_sampled_image<Texture3D<float4>>(pc.history_tex);
    Texture3D<float4> current_tex = get_sampled_image<Texture3D<float4>>(pc.input_tex);
    RWTexture3D<float4> output_tex = get_storage_image<RWTexture3D<float4>>(pc.output_tex);

    float3 uvw = (float3(froxel_coord) + float3(0.5)) / float3(pc.froxel_dims);

    float4 scattering_extinction = current_tex.SampleLevel(samp, uvw, 0.0);
    if (pc.first_frame == 1)
    {
        output_tex.Store(froxel_coord, scattering_extinction);
        return;
    }

    float temporal_reprojection_jitter_scale = 0.5;
    float3 world_pos_no_jitter = world_from_froxel(froxel_coord, pc.froxel_dims, pc.inverse_view_proj, pc.near, pc.far, pc.halton, temporal_reprojection_jitter_scale, pc.blue_noise_tex, pc.current_frame);
    float4 screen_space_center_last = mul(pc.previous_view_proj, float4(world_pos_no_jitter, 1.0));
    float3 ndc = screen_space_center_last.xyz / screen_space_center_last.w;

    // compute volume z slice to index into
    float linear_depth = pc.near / ndc.z; // infinite plane + reverse z
    uint slice = uint(floor(log(linear_depth) * pc.volumetrics_scale - pc.volumetrics_bias));
    uint max_slice = pc.froxel_dims.z - 1;
    slice = clamp(slice, 0, max_slice);
    float w = float(slice) / float(max_slice);

    ndc.xy += pc.halton;
    float2 history_uv = ndc.xy * float2(0.5, -0.5) + float2(0.5);
    float3 history_uvw = float3(history_uv, w);

    if (all(history_uvw >= 0.0) && all(history_uvw <= 1.0))
    {
        float4 history = history_tex.SampleLevel(samp, history_uvw, 0.0);
        history = max(history, scattering_extinction);
        scattering_extinction.rgb = lerp(history.rgb, scattering_extinction.rgb, pc.reprojection_factor);
        scattering_extinction.a = lerp(history.a, scattering_extinction.a, pc.reprojection_factor);
    }

    output_tex.Store(froxel_coord, scattering_extinction);
}
