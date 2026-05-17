module bindings;

// TODO: can we not import this entire module for a struct?
import oit;

public struct UniformBuffer
{
    public float4x4 view;
	public float4x4 proj;
	public float4x4 view_proj;
	public float4x4 inverse_view_proj;
	public float4x4 prev_view_proj;
	public float4x4 light_rot;
	public float4x4 shadow_transforms[4];
	public float4 cascade_splits;
	public float4 camera_pos;
	public float4 sunlight_color;
	public float4 sunlight_dir; // w for sun power
	public float4 textures; // cubemap/skybox, irradiance, prefiltered, brdf
	public float4x4 shadow_views[4]; // for shadow cull
	public float4 shadow_widths; // for shadow cull, each channel for a diff cascade; TODO: move to pc?
}

[[vk::binding(0, 0)]] public ConstantBuffer<UniformBuffer> uniforms;
[[vk::binding(1, 0)]] public globallycoherent RasterizerOrderedStructuredBuffer<OITData> oit_buffer;
[[vk::binding(2, 0)]] public RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0, 2)]][vk::image_format("r32f")] public RWTexture2D<float> images_r32f[];
[[vk::binding(0, 2)]][vk::image_format("rgba32f")] public RWTexture2DArray<float4> images_array_rgba32f[];
[[vk::binding(0, 2)]][vk::image_format("rgba32f")] public RWTexture2D<float4> images_rgba32f[];
[[vk::binding(0, 2)]][vk::image_format("rg16f")] public RWTexture2D<float2> images_rg16f[];

[[vk::binding(0, 1)]] public Texture2D<float4> textures[];
[[vk::binding(0, 1)]] public TextureCube<float4> textures_cube[];
[[vk::binding(0, 1)]] public Texture2D<uint4> textures_u[];

public static const uint LINEAR_SAMPLER = 0;
public static const uint CUBE_SAMPLER = 1;
public static const uint SHADOW_SAMPLER = 2;
public static const uint DEPTH_REDUCTION_SAMPLER = 3;
public static const uint NEAREST_CLAMP_BORDER_SAMPLER = 4;
public static const uint NEAREST_SAMPLER = 5; // clamp to edge
public static const uint LINEAR_CLAMP_SAMPLER = 6; // clamp to edge

[[vk::binding(0, 3)]] public SamplerState samplers[];
