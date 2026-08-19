#version 460

struct QuadsStyleData
{
    uint background_color;
    uint border_color;
    float border_radius;
    float border_thickness;
    float z_order;
    uint mask;
};

layout(std430, set = 0, binding = 0) readonly buffer RectBuffer { vec4 rects[]; };
layout(std430, set = 0, binding = 1) readonly buffer StyleBuffer { QuadsStyleData styles[]; };

layout(push_constant) uniform Push { vec2 window_size; };

layout(location = 0) out vec2 out_local_pos;
layout(location = 1) flat out vec2 out_size;
layout(location = 2) flat out vec4 out_background_color;
layout(location = 3) flat out vec4 out_border_color;
layout(location = 4) flat out float out_border_radius;
layout(location = 5) flat out float out_border_thickness;
layout(location = 6) flat out uint out_corner_mask;
layout(location = 7) flat out uint out_flags;
layout(location = 8) flat out uint out_clip_id;
layout(location = 9) out vec2 out_pixel_pos;

vec2 get_quad_uv(uint vertex_index)
{
    const vec2 uv[6] =
        vec2[6](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));
    return uv[vertex_index];
}

void main()
{
    vec4 rect = rects[gl_InstanceIndex];
    QuadsStyleData style = styles[gl_InstanceIndex];
    vec2 uv = get_quad_uv(uint(gl_VertexIndex));

    vec2 pixel_pos = rect.xy + uv * rect.zw;
    vec2 ndc = vec2((pixel_pos.x / window_size.x) * 2.0 - 1.0, (pixel_pos.y / window_size.y) * 2.0 - 1.0);

    gl_Position = vec4(ndc, style.z_order, 1.0);

    out_local_pos = (uv - 0.5) * rect.zw;
    out_pixel_pos = pixel_pos;
    out_size = rect.zw;
    out_background_color = unpackUnorm4x8(style.background_color);
    out_border_color = unpackUnorm4x8(style.border_color);
    out_border_radius = style.border_radius;
    out_border_thickness = style.border_thickness;
    const uint style_mask = style.mask >> 16u;
    out_clip_id = style.mask & 0xFFFFu;
    out_corner_mask = style_mask & 0xFu;
    out_flags = style_mask >> 4u;
}
