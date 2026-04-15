// https://github.com/Devsh-Graphics-Programming/Nabla/blob/8da4b980b5617802ea4e96bf101ddbfe94721a51/include/nbl/builtin/glsl/scanning_append/scanning_append.glsl
// https://github.com/expenses/lighthugger/blob/main/src/shaders/common/prefix_sum.glsl

module prefix_sum;

public struct PrefixSumData
{
    public uint instance_id;
    public uint sum;
    public uint lod_offset;
};

public struct PrefixSum
{
    public uint64_t counter;
    public PrefixSumData data[1000000]; // TODO: hardcoded
};

// note: offset can be packed with bit flags as we are unlikely to even use half its bits
// currently msb is set to mesh visibility last frame for late pass
public void prefix_sum_inclusive_append(PrefixSum* buf, uint index, uint value, uint offset)
{
    uint64_t count = uint64_t(1) << uint64_t(32);
    // uint64_t sum_and_counter = atomicAdd(buf.prefix_sum.counter, count | uint64_t(value));
    uint64_t sum_and_counter;
    InterlockedAdd(buf.counter, count | uint64_t(value), sum_and_counter);
    uint instance_index = uint(sum_and_counter >> 32);

    buf.data[instance_index].instance_id = index;
    // buf.data[instance_index].sum = uint(sum_and_counter) + value; // inclusive
    buf.data[instance_index].sum = uint(sum_and_counter); // exclusive
    buf.data[instance_index].lod_offset = offset;
}

public uint prefix_sum_binary_search(PrefixSum* buf, uint target)
{
    // extract the number of entries/instances
    uint count = uint(buf.counter >> 32);
    uint first = 0;

    while (count > 0)
    {
        uint step = count / 2;
        uint current = first + step;
        bool greater = target >= buf.data[current].sum;
        first =  greater ? (current + 1) : first; // left or right path
        count = greater ? (count - (step + 1)) : step; // size of left or right path
    }

    return first - 1; // remove -1 for inclusive
}

public uint64_t get_prefix_sum_counter(PrefixSum* buf)
{
    return buf.counter;
}
