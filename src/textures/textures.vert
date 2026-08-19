#version 460
struct TexturesStyleData
{
    vec2 uv_offset;
    vec2 uv_size;
    uint tint_color;
    float z_order;
    uint packed_id;
    uint flags;
};

layout(std430, set = 0, binding = 0) readonly buffer RectBuffer { vec4 rects[]; };
layout(std430, set = 0, binding = 1) readonly buffer StyleBuffer { TexturesStyleData styles[]; };

layout(push_constant) uniform Push { vec2 window_size; };

layout(location = 0) out vec2 out_uv;
layout(location = 1) flat out vec4 out_tint_color;
layout(location = 2) flat out uint out_texture_id;
layout(location = 3) flat out uint out_clip_id;
layout(location = 4) flat out uint out_flags;
layout(location = 5) out vec2 out_pixel_pos;

vec2 get_quad_uv(uint vertex_index)
{
    const vec2 uv[6] =
        vec2[6](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));
    return uv[vertex_index];
}

void main()
{
    vec4 rect = rects[gl_InstanceIndex];
    TexturesStyleData style = styles[gl_InstanceIndex];
    vec2 uv = get_quad_uv(uint(gl_VertexIndex));

    vec2 pixel_pos = rect.xy + uv * rect.zw;
    vec2 ndc = vec2((pixel_pos.x / window_size.x) * 2.0 - 1.0, (pixel_pos.y / window_size.y) * 2.0 - 1.0);

    gl_Position = vec4(ndc, style.z_order, 1.0);

    out_uv = style.uv_offset + uv * style.uv_size;
    out_tint_color = unpackUnorm4x8(style.tint_color);
    out_texture_id = style.packed_id & 0xFFFFu;
    out_clip_id = style.packed_id >> 16u;
    out_flags = style.flags;
    out_pixel_pos = pixel_pos;
}
