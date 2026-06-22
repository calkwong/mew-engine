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
	public uint skybox_id;
	public uint irradiance_id;
	public uint prefiltered_id;
	public uint brdf_id;
	public float4x4 shadow_views[4]; // for shadow cull
	public float4 shadow_widths; // for shadow cull, each channel for a diff cascade; TODO: move to pc?
}

// REQUIRES MAX_FRAMES_IN_FLIGHT = 2
[[vk::binding(0, 0)]] public ConstantBuffer<UniformBuffer> uniforms[];
[[vk::binding(2, 0)]] public globallycoherent RasterizerOrderedStructuredBuffer<OITData> oit_buffer;
[[vk::binding(3, 0)]] public RaytracingAccelerationStructure acceleration_structure;

public static const uint LINEAR_SAMPLER = 0;
public static const uint CUBE_SAMPLER = 1;
public static const uint SHADOW_SAMPLER = 2;
public static const uint DEPTH_REDUCTION_SAMPLER = 3;
public static const uint NEAREST_CLAMP_BORDER_SAMPLER = 4;
public static const uint NEAREST_SAMPLER = 5; // clamp to edge
public static const uint LINEAR_CLAMP_SAMPLER = 6; // clamp to edge
public static const uint NEAREST_REPEAT_SAMPLER = 7; // nearest repeat

// https://shader-slang.org/slang/user-guide/convenience-features.html#descriptorhandle-for-bindless-descriptor-access
[vk::binding(0, 1)]
__DynamicResource<__DynamicResourceKind.General> storage_images[];

[vk::binding(0, 2)]
__DynamicResource<__DynamicResourceKind.General> sampled_images[];

[vk::binding(0, 3)]
__DynamicResource<__DynamicResourceKind.Sampler> samplers[];

public T get_storage_image<T>(uint handle) where T : IOpaqueDescriptor
{
    return storage_images[handle].asOpaqueDescriptor<T>();
}

public T get_sampled_image<T>(uint handle) where T : IOpaqueDescriptor
{
    return sampled_images[handle].asOpaqueDescriptor<T>();
}

public T get_sampler<T>(uint handle) where T : IOpaqueDescriptor
{
    return samplers[handle].asOpaqueDescriptor<T>();
}
