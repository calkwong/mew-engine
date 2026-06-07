// References
// https://dl.acm.org/doi/epdf/10.1145/2556700.2556705
// https://github.com/Devsh-Graphics-Programming/Nabla/blob/master/include/nbl/builtin/glsl/ext/OIT/insert_node.glsl
// https://interplayoflight.wordpress.com/2022/07/02/order-independent-transparency-part-2/

import bindings;
import types;
import util;
import prefix_sum;
import oit;
import pbr;
import sh;
import transmission;

struct Interp
{
    float4 position : SV_Position;
    float3 world_pos;
    float3 view_pos;
    float3 normal;
    float2 uv;
    float4 tangent;
    nointerpolation uint material_id;
    nointerpolation uint transmission;
    nointerpolation float scale;
};

struct PushConstants
{
	ObjectData* object_buffer;
	Vertex* vertex_buffer;
	Meshlet* meshlet_buffer;
	uint* meshlet_indices_buffer;
	uint* cluster_indices_buffer;
	MaterialData* material_buffer;
	PrefixSum* prefix_sum_buffer;
	SH* sh_buffer;
	PointLight* light_buffer;
	uint* light_index_buffer;
	LightGrid* light_grid_buffer;
	float2 cluster_size;
	uint2 screen_size;
	float max_prefiltered_lod;
	uint framebuffer_id;
	uint volume;
	uint point_lights;
	uint scale;
	uint bias;
};

[[vk::push_constant]] PushConstants pc;

[shader("vertex")]
Interp vs_main(
    uint instance_id : SV_VulkanInstanceID,
    uint vertex_id : SV_VulkanVertexID
)
{
    Interp interp;

	ObjectData o = pc.object_buffer[instance_id]; // instance_id from drawIndirectCommand
	Vertex v = pc.vertex_buffer[vertex_id];

	float3 pos = float3(v.px, v.py, v.pz);
	float4 world_pos = float4(rotate_quat(pos, o.orientation) * o.scale + o.translation, 1.0);
	float4 view_pos = mul(uniforms.view, world_pos);
    float4 clip_pos = mul(uniforms.view_proj, world_pos);
	interp.position = clip_pos;
	interp.world_pos = world_pos.xyz;
	interp.view_pos = view_pos.xyz;

	float3 normal;
	float4 tangent;
	unpack_tbn(v.normal, uint(v.tangent), normal, tangent);

	interp.normal = rotate_quat(normal, o.orientation);
	interp.tangent = float4(rotate_quat(tangent.xyz, o.orientation), tangent.w);

	interp.uv = float2(v.uv_x, v.uv_y);
	interp.material_id = o.material_id;

	interp.transmission = (o.post_pass == 3) ? 1 : 0;
	interp.scale = o.scale;

	return interp;
}

// note: no triangle culling as VK_CULL_MODE_NONE
#define WG_SIZE_X 128

[shader("mesh")]
[numthreads(WG_SIZE_X, 1, 1)]
[outputtopology("triangle")]
void mesh_main(
    uint3 wg_id : SV_GroupID,
    uint t_index : SV_GroupIndex,
    out vertices Interp verts[64],
    out indices uint3 tris[124]
)
{
	uint id = wg_id.x;

	uint64_t counter = get_prefix_sum_counter(pc.prefix_sum_buffer);
	uint total_meshlet_count = uint(counter);
	uint target = pc.cluster_indices_buffer[id];

	uint prefix_sum_idx = prefix_sum_binary_search(pc.prefix_sum_buffer, target);
    PrefixSumData prefix_sum_data = pc.prefix_sum_buffer.data[prefix_sum_idx];
    uint meshlet_lod_offset = prefix_sum_data.lod_offset;

    // we do some packing hence this is essential!
    meshlet_lod_offset = bitfieldExtract(meshlet_lod_offset, 0u, 31u);

    uint object_id = prefix_sum_data.instance_id;

   	bool overflow = prefix_sum_idx == total_meshlet_count;

    uint inclusive_sum = overflow ? total_meshlet_count : pc.prefix_sum_buffer.data[prefix_sum_idx + 1].sum;
    uint exclusive_sum = prefix_sum_data.sum;
    uint local_meshlet_count = inclusive_sum - exclusive_sum;

    // actual meshlet index
    uint local_meshlet_index = target - (inclusive_sum - local_meshlet_count);

    uint meshlet_offset = local_meshlet_index + meshlet_lod_offset;
	Meshlet meshlet = pc.meshlet_buffer[meshlet_offset];

	uint vertex_count = meshlet.vertex_count;
	uint triangle_count = meshlet.triangle_count;

	SetMeshOutputCounts(vertex_count, triangle_count);

	ObjectData o = pc.object_buffer[object_id];

	uint m_vertex_offset = meshlet.data_offset; // offset into meshlet_indices_buffer
	uint m_triangle_offset = m_vertex_offset + vertex_count;

	for (uint i = t_index; i < vertex_count; i += WG_SIZE_X)
	{
		uint vertex_index = pc.meshlet_indices_buffer[m_vertex_offset + i];
		Vertex v = pc.vertex_buffer[vertex_index];

		float3 pos = float3(v.px, v.py, v.pz);
		float4 world_pos = float4(rotate_quat(pos, o.orientation) * o.scale + o.translation, 1.0);
		float4 view_pos = mul(uniforms.view, world_pos);
	    float4 clip_pos = mul(uniforms.view_proj, world_pos);
		verts[i].position = clip_pos;
		verts[i].world_pos = world_pos.xyz;
		verts[i].view_pos = view_pos.xyz;

		float3 normal;
		float4 tangent;
		unpack_tbn(v.normal, uint(v.tangent), normal, tangent);

		verts[i].normal = rotate_quat(normal, o.orientation);
		verts[i].tangent = float4(rotate_quat(tangent.xyz, o.orientation), tangent.w);

		verts[i].uv = float2(v.uv_x, v.uv_y);
		verts[i].material_id = o.material_id;

		verts[i].transmission = (o.post_pass == 3) ? 1 : 0;
		verts[i].scale = o.scale;
	}

#ifdef TRIANGLE_CULLING
    GroupMemoryBarrierWithGroupSync();
#endif

	for (uint i = t_index; i < triangle_count; i += WG_SIZE_X)
	{
		uint idx0 = pc.meshlet_indices_buffer[m_triangle_offset + i * 3 + 0];
		uint idx1 = pc.meshlet_indices_buffer[m_triangle_offset + i * 3 + 1];
		uint idx2 = pc.meshlet_indices_buffer[m_triangle_offset + i * 3 + 2];

		tris[i] = uint3(idx0, idx1, idx2);
	}
}

// notes:
// - for alphacoverage, flickering occurs when nodes overflow as the draw calls in the indirect buffer we build
// are unsorted and vary each frame; to fix either implement GPU sorting of indirect buffer, or reserve a spot in
// the indirect buffer for each mesh and mark empty draw calls via instanceCount = 1/0 accordingly (command parser hates this)
// - alphacoverage and transmission do not currently layer on top of one another properly
//
// future work:
// - shadows
// - punctual lights
// - KHR_materials_volume

[earlydepthstencil]
[shader("pixel")]
void ps_main(
    Interp in
)
{
    SamplerState samp = get_sampler<SamplerState>(LINEAR_SAMPLER);

	MaterialData m = pc.material_buffer[in.material_id];

	float4 albedo = m.base_color_factor;
	if (m.diffuse_id != 0)
		albedo *= get_sampled_image<Texture2D>(m.diffuse_id).Sample(samp, in.uv);

	float3 N = in.normal;
	if (m.normal_id != 0)
	{
		float4 T = in.tangent; // mikktspace
		float sign = in.tangent.w; // sign is flipped during tangent generation so mikktspace is consistent with glTF handedness

		float3 shading_normal = get_sampled_image<Texture2D>(m.normal_id).Sample(samp, in.uv).xyz;
		shading_normal = shading_normal * 2.0 - 1.0;
		N = tangent_to_world_space(shading_normal, T, N);
	}
	else
	{
        N = normalize(N);
	}

	float metallic = m.metallic_factor;
	float perceptual_roughness = m.roughness_factor;
	if (m.metalroughness_id != 0)
	{
		float2 metal_roughness = get_sampled_image<Texture2D>(m.metalroughness_id).Sample(samp, in.uv).bg;
		metallic *= metal_roughness.x;
		perceptual_roughness *= metal_roughness.y;
	}
	perceptual_roughness = max(perceptual_roughness, 0.045);
	float roughness = perceptual_roughness * perceptual_roughness;

	float transmission_factor = 0.0;
	bool material_transmission = (in.transmission == 1);
	if (material_transmission)
	{
	    transmission_factor = m.transmission_factor;
    	if (m.transmission_id != 0)
        {
            transmission_factor *= get_sampled_image<Texture2D>(m.transmission_id).Sample(samp, in.uv).r;
        }
	}

	float3 Fr = float3(0.0);

	float3 L = normalize(uniforms.sunlight_dir.xyz);
	float3 V = normalize(uniforms.camera_pos.xyz - in.world_pos);
	float3 H = normalize(L + V);

	float3 light_color = uniforms.sunlight_color.xyz;

	BRDFData brdf_data = prepare_brdf_data(N, L, V, albedo.xyz, roughness, metallic);
	float3 direct_lighting;
    float thickness = m.thickness_factor;
    float3 attenuation_color = m.attenuation_color;
    float3 attenuation_distance = m.attenuation_distance;
    float ior = 1.5;
    float scale = in.scale;
    if (pc.volume == 0)
    {
        thickness = 0.0;
        attenuation_color = float3(1.0);
        attenuation_distance = 0.0; // infinity
    }
	if (material_transmission)
	{
    	// note: using lambertian here as I think burley's implicit Fresnel modulation breaks the model?
        float3 l_diffuse = eval_diffuse_brdf(brdf_data, albedo.xyz, roughness, metallic) * light_color * brdf_data.NdotL;

        if (thickness != 0.0 && m.thickness_id != 0)
        {
            thickness *= get_sampled_image<Texture2D>(m.thickness_id).Sample(samp, in.uv).g;
        }

        // if light ray travels through geometry, use the point it exits the geometry again
        // this changes angle to light source if material refracts the light ray
        float3 transmission_ray = get_volume_transmission_ray(brdf_data, thickness, ior, scale);
        // note: original L is stored in BRDFData
        L -= transmission_ray;
        L = normalize(L);

        float3 transmitted_light = eval_punctual_radiance_transmission(brdf_data, L, albedo.xyz, roughness, ior) * light_color;
        if (pc.volume == 1)
            transmitted_light = apply_volume_attenuation(transmitted_light, length(transmission_ray), m.attenuation_color, m.attenuation_distance);

        l_diffuse = lerp(l_diffuse, transmitted_light, transmission_factor);
        float3 F;
        float3 l_specular = eval_specular_brdf(brdf_data, roughness, F) * light_color * brdf_data.NdotL;
        direct_lighting = l_diffuse * (1.0 - F) + l_specular;

        // note: likely to have correctness issues for transmission/volume model
        #if 1
        if (pc.point_lights == 1)
        {
            float view_z = in.view_pos.z; // TODO: can we do better than passing another matrix as interpolant?
    		uint slice = get_cluster_z_slice(view_z, pc.scale, pc.bias);
            float2 frag_coord = in.position.xy;
    		uint cluster_index = get_cluster_index(float2(frag_coord.x, pc.screen_size.y - frag_coord.y), pc.cluster_size, slice);

    		LightGrid grid = pc.light_grid_buffer[cluster_index];
    		uint offset = grid.offset;
    		uint count = grid.count;

            for (int i = 0; i < count; i++)
            {
                uint index = pc.light_index_buffer[offset + i];

          		PointLight point_light = pc.light_buffer[index];
          		float3 light_pos = point_light.pos.xyz;
          		float light_radius = point_light.pos.w;
          		float3 light_color = point_light.color.xyz;

          		light_pos = mul(uniforms.light_rot, float4(light_pos, 1.0)).xyz;
          		float3 point_to_light = light_pos - in.world_pos;

                update_brdf_data(brdf_data, normalize(point_to_light));
                float attenuation = get_square_falloff_attenuation(point_to_light, light_radius);

                float3 l_diffuse = eval_diffuse_brdf(brdf_data, albedo.xyz, roughness, metallic) * light_color * brdf_data.NdotL * attenuation;

                // VERIFY: transmission ray unchanged
                point_to_light -= transmission_ray;
                L = normalize(point_to_light);

                float3 transmitted_light = eval_punctual_radiance_transmission(brdf_data, L, albedo.xyz, roughness, ior) * light_color * attenuation;
                if (pc.volume == 1)
                    transmitted_light = apply_volume_attenuation(transmitted_light, length(transmission_ray), m.attenuation_color, m.attenuation_distance);

                l_diffuse = lerp(l_diffuse, transmitted_light, transmission_factor);
                float3 F;
                // update attenuation to account for adjusted point_to_light
                attenuation = get_square_falloff_attenuation(point_to_light, light_radius);
                float3 l_specular = eval_specular_brdf(brdf_data, roughness, F) * light_color * brdf_data.NdotL * attenuation;
                direct_lighting += l_diffuse * (1.0 - F) + l_specular;
            }
        }
        #endif
	}
	else
	{
	    // TODO: account for point lights
	    direct_lighting = eval_brdf(brdf_data, albedo.xyz, roughness, metallic) * light_color * brdf_data.NdotL;
	}

	// TODO: double check reflect
	float3 R = reflect(-V, N);
	float prefiltered_mip = pc.max_prefiltered_lod * perceptual_roughness;

	SamplerState cube_samp = get_sampler<SamplerState>(CUBE_SAMPLER);
	SamplerState linear_clamp_samp = get_sampler<SamplerState>(LINEAR_CLAMP_SAMPLER);

	float3 irradiance = evaluate_sh(pc.sh_buffer->r_coefficients, pc.sh_buffer->g_coefficients, pc.sh_buffer->b_coefficients, N);
	float3 radiance = get_sampled_image<TextureCube>(uniforms.prefiltered_id).SampleLevel(cube_samp, R, prefiltered_mip).xyz;
	float2 brdf_lut = get_sampled_image<Texture2D>(uniforms.brdf_id).SampleLevel(linear_clamp_samp, float2(brdf_data.NdotV, perceptual_roughness), 0.0).rg;

	float3 indirect_lighting;
	if (material_transmission)
	{
    	float3 f_diffuse = irradiance * albedo.xyz;
        float max_extent = float(max(pc.screen_size.x, pc.screen_size.y));
        float3 f_specular_transmission = eval_ibl_volume_refraction(brdf_data, albedo.xyz, perceptual_roughness, in.world_pos, uniforms.view_proj, pc.framebuffer_id, max_extent, thickness, m.attenuation_color, m.attenuation_distance, ior, scale);
        f_diffuse = lerp(f_diffuse, f_specular_transmission, transmission_factor);

        // for simplicity, use single scattering for transmission path
        float3 f_specular = (brdf_data.f0 * brdf_lut.x + brdf_lut.y) * radiance;
        indirect_lighting = f_diffuse + f_specular;
	}
	else
	    indirect_lighting = eval_indirect_multiscattering(brdf_data, irradiance, radiance, brdf_lut, albedo.xyz, perceptual_roughness, metallic);

	float3 outgoing_radiance = direct_lighting + indirect_lighting;

	float4 premultiplied_alpha = float4(outgoing_radiance * albedo.a, 1.0);
	uint color = packUnorm4x8(premultiplied_alpha);
	uint depth = asuint(in.position.z);
	// this should not be called transmission, it is only related to alpha coverage
	float transmission = 1.0 - albedo.a;

	uint2 screen_coords = uint2(floor(in.position.xy));

	uint index = screen_coords.x + screen_coords.y * pc.screen_size.x;
	OITData fragment = oit_buffer[index];
	insert_node(fragment, color, depth, transmission);

	oit_buffer[index] = fragment;
}
