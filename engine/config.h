#pragma once

#define SHADOW_MAP_SIZE 4096
#define NUMBER_OF_CASCADES 4

#define DEBUG_COUNT 10

// for shadow pass, we set to 200000; for geometry pass, raised to 1000000 to allow testing millions of mesh instances
#define MAX_OPAQUE_DRAWS 200000
#define MAX_ALPHACLIP_DRAWS 200000
#define MAX_MESH_DRAWS 1000000

#define WARP_SIZE 32
#define LUMINANCE_BINS 16 // 16*16=256
#define CULL_WGSIZE 256

// queries
#define QUERY_COUNT 50
#define TIMESTAMP_QUERIES 30
#define PIPELINE_QUERIES 8

// clustered shading
#define CLUSTER_DEPTH_SLICES 24
#define MAX_POINT_LIGHTS 1000
#define CLUSTER_X 16
#define CLUSTER_Y 8
#define CLUSTER_Z 4 // only for light_culling

// meshlets
#define MESHLET_MAX_VERTICES 64
#define MESHLET_MAX_TRIANGLES 124
#define MESHLET_LIMIT (1 << 24) // max VISIBLE meshlets, ~16.7m meshlets for ~64mb. validate with frame 0 capture for test scenes.

// samplers
#define LINEAR_SAMPLER 0
#define CUBE_SAMPLER 1
#define SHADOW_SAMPLER 2
#define DEPTH_REDUCTION_SAMPLER 3
#define NEAREST_CLAMP_BORDER_SAMPLER 4
#define NEAREST_SAMPLER 5 // clamp to edge
#define LINEAR_CLAMP_SAMPLER 6 // clamp to edge
