#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

struct PrefixSumData
{
    uint instance_id;
    uint sum;
    uint lod_offset;
};

struct PrefixSum
{
    uint64_t counter;
    PrefixSumData data[1000000];
};

layout(buffer_reference, std430) buffer PrefixSumBuffer
{
	PrefixSum prefix_sum;
};

// TODO: handle lod offset
void prefix_sum_inclusive_append(PrefixSumBuffer buf, uint index, uint value)
{
    uint64_t count = uint64_t(1) << uint64_t(32);
    uint64_t sum_and_counter = atomicAdd(buf.prefix_sum.counter, count | uint64_t(value));
    uint instance_index = uint(sum_and_counter >> 32);

    buf.prefix_sum.data[instance_index].instance_id = index;
    buf.prefix_sum.data[instance_index].sum = uint(sum_and_counter);
}

PrefixSumData prefix_sum_binary_search(PrefixSumBuffer buf, uint target)
{
    // extract the number of entries/instances
    uint count = uint(buf.prefix_sum.counter >> 32);
    uint first = 0;

    while (count > 0)
    {
        uint step = count / 2;
        uint current = first + step;
        bool greater = target >= buf.prefix_sum.data[current].sum;
        first =  greater ? (current + 1) : first; // left or right path
        count = greater ? (count - (step + 1)) : step; // size of left or right path
    }

    return buf.prefix_sum.data[first];
}
