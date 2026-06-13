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
    uint previous_scattering_extinction_tex;
    uint scattering_extinction_tex; //unused

    uint rw_scattering_extinction_tex;
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
    SamplerState samp = get_sampler<SamplerState>(NEAREST_SAMPLER);
    Texture3D previous_scattering_extinction_tex = get_sampled_image<Texture3D>(pc.previous_scattering_extinction_tex);
    RWTexture3D rw_scattering_extinction_tex = get_storage_image<RWTexture3D>(pc.rw_scattering_extinction_tex);

    float4 scattering_extinction = rw_scattering_extinction_tex.Load(froxel_coord);

    float3 world_pos_no_jitter = world_from_froxel_no_jitter(froxel_coord, pc.froxel_dims, pc.inverse_view_proj, pc.near, pc.far);
    // float temporal_reprojection_jitter_scale = 0.5;
    // float3 world_pos_no_jitter = world_from_froxel(froxel_coord, pc.froxel_dims, pc.inverse_view_proj, pc.near, pc.far, pc.halton, temporal_reprojection_jitter_scale, pc.blue_noise_tex, pc.current_frame);
    float4 screen_space_center_last = mul(pc.previous_view_proj, float4(world_pos_no_jitter, 1.0));
    float3 ndc = screen_space_center_last.xyz / screen_space_center_last.w;

    // compute volume z slice to index into
    float linear_depth = pc.near / ndc.z; // infinite plane + reverse z
    uint slice = uint(floor(log(linear_depth) * pc.volumetrics_scale - pc.volumetrics_bias));
    uint max_slice = pc.froxel_dims.z - 1;
    slice = clamp(slice, 0, max_slice);
    float w = float(slice) / float(max_slice);

    ndc.xy = ndc.xy * float2(0.5, -0.5) + float2(0.5);
    float3 history_uvw = float3(ndc.xy, w);

    if (all(history_uvw >= 0.0) && all(history_uvw <= 1.0))
    {
        // bug: previous light scattering? or filtered (so we are correct here)?
        float4 history = previous_scattering_extinction_tex.SampleLevel(samp, history_uvw, 0.0);
        history = max(history, scattering_extinction);

        scattering_extinction.rgb = lerp(history.rgb, scattering_extinction.rgb, pc.reprojection_factor);
        scattering_extinction.a = lerp(history.a, scattering_extinction.a, pc.reprojection_factor);
    }

    rw_scattering_extinction_tex.Store(froxel_coord, scattering_extinction);
}
