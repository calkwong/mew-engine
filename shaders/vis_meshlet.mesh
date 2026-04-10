#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_mesh_shader : require

#include "bindings.glsl"
#include "buffer_references.glsl"
#include "util.glsl"
#include "prefix_sum.glsl"

layout(local_size_x = 128) in;
layout(triangles, max_vertices = 64, max_primitives = 124) out; // not 126?

layout (constant_id = 0) const int OPAQUE = 1;

layout(push_constant) uniform constants
{
    ObjectBuffer object_buffer;
    VertexBuffer vertex_buffer;
    MeshletBuffer meshlet_buffer;
    MeshletIndicesBuffer meshlet_indices_buffer;
    ClusterIndicesBuffer cluster_indices_buffer;
    MaterialBuffer material_buffer;
    OITBuffer oit_buffer;
    PrefixSumBuffer prefix_sum_buffer;
    vec4 jitter_offset;
};

layout (location = 0) perprimitiveEXT out uint out_draw_id[];
layout (location = 1) perprimitiveEXT out uint out_triangle_id[];
layout (location = 2) out vec2 out_uv[];
layout (location = 3) out uint out_material_id[];
// layout (location = 4) out vec4 out_clip_pos[];
// layout (location = 5) out vec4 out_prev_clip_pos[];

#define TRIANGLE_CULLING

#ifdef TRIANGLE_CULLING
shared vec3 vertex_clip[64];
#endif

void main()
{
    uint id = gl_WorkGroupID.x;

    uint64_t counter = get_prefix_sum_counter(prefix_sum_buffer);
	uint total_meshlet_count = uint(counter);
    uint target = cluster_indices_buffer.indices[id];

    uint prefix_sum_idx = prefix_sum_binary_search(prefix_sum_buffer, target);
    PrefixSumData prefix_sum_data = prefix_sum_buffer.prefix_sum.data[prefix_sum_idx];
    uint meshlet_lod_offset = prefix_sum_data.lod_offset;
    meshlet_lod_offset = bitfieldExtract(meshlet_lod_offset, 0, 31); // we do some packing hence this is essential!
    uint object_id = prefix_sum_data.instance_id;

   	bool overflow = prefix_sum_idx == total_meshlet_count;

	uint inclusive_sum = overflow ? total_meshlet_count : prefix_sum_buffer.prefix_sum.data[prefix_sum_idx + 1].sum;
	uint exclusive_sum = prefix_sum_data.sum;
	uint local_meshlet_count = inclusive_sum - exclusive_sum;

	// actual meshlet index
	uint local_meshlet_index = target - (inclusive_sum - local_meshlet_count);

	uint meshlet_offset = local_meshlet_index + meshlet_lod_offset;
	Meshlet meshlet = meshlet_buffer.meshlets[meshlet_offset];

    uint vertex_count = meshlet.vertex_count;
    uint triangle_count = meshlet.triangle_count;

    SetMeshOutputsEXT(vertex_count, triangle_count);

    ObjectData o = object_buffer.objects[object_id];

    uint m_vertex_offset = meshlet.data_offset; // offset into meshlet_indices_buffer
    uint m_triangle_offset = m_vertex_offset + vertex_count;

    uint ti = gl_LocalInvocationIndex;

    for (uint i = ti; i < vertex_count; i += gl_WorkGroupSize.x)
    {
        uint vertex_index = meshlet_indices_buffer.indices[m_vertex_offset + i]; // vertex indices for meshlets are global, really need to document this
        Vertex v = vertex_buffer.vertices[vertex_index];

        vec3 pos = vec3(v.px, v.py, v.pz);
        vec4 world_pos = vec4(rotate_quat(pos, o.orientation) * o.scale + o.translation, 1.0);

        out_uv[i] = vec2(v.uv_x, v.uv_y);
        out_material_id[i] = o.material_id; // only used for alpha clipping

        vec4 clip_pos = uniforms.view_proj * world_pos;

        // TODO: only account for screen size if doing small primitive culling
#ifdef TRIANGLE_CULLING
        if (OPAQUE == 1)
        {
            vertex_clip[i] = vec3(clip_pos.xy / clip_pos.w * 0.5 + 0.5, clip_pos.w);
        }
#endif
        gl_MeshVerticesEXT[i].gl_Position = clip_pos;
    }

    barrier();

    for (uint i = ti; i < triangle_count; i += gl_WorkGroupSize.x)
    {
        out_draw_id[i] = object_id;

        // WARNING: unsafe if meshlet_offset is large
        out_triangle_id[i] = (i << 25) | meshlet_offset;

        uint idx0 = meshlet_indices_buffer.indices[m_triangle_offset + i * 3 + 0];
        uint idx1 = meshlet_indices_buffer.indices[m_triangle_offset + i * 3 + 1];
        uint idx2 = meshlet_indices_buffer.indices[m_triangle_offset + i * 3 + 2];

        gl_PrimitiveTriangleIndicesEXT[i] = uvec3(idx0, idx1, idx2);

#ifdef TRIANGLE_CULLING
        // backface + zero area culling
        if (OPAQUE == 1)
        {
            vec2 va = vertex_clip[idx0].xy;
            vec2 vb = vertex_clip[idx1].xy;
            vec2 vc = vertex_clip[idx2].xy;

            vec2 eb = vb - va;
            vec2 ec = vc - va;

            // triangle backface + zero area culling
            bool culled = eb.x * ec.y <= eb.y * ec.x;

            // triangle culling only works if all triangles are in front of camera
            culled = culled && (vertex_clip[idx0].z > 0 && vertex_clip[idx1].z > 0 && vertex_clip[idx2].z > 0);

            // this cannot be in a runtime branch! compiler likely sets this to a default value after dead code removal if it cannot find gl_CullPrimitiveEXT
            // if it exists and we place it in a branch, we get flickering artifacts all over
            gl_MeshPrimitivesEXT[i].gl_CullPrimitiveEXT = culled;
        }
#endif
    }
}
