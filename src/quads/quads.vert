#version 460

struct QuadsInstanceData
{
    vec2 position;
    vec2 size;
    uint background_color;
    uint border_color;
    float border_radius;
    float border_thickness;
    float z_order;
    uint mask;
};

layout(std430, set = 0, binding = 0) readonly buffer QuadsBuffer { QuadsInstanceData instances[]; };

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
    QuadsInstanceData instance = instances[gl_InstanceIndex];
    vec2 uv = get_quad_uv(uint(gl_VertexIndex));

    vec2 pixel_pos = instance.position + uv * instance.size;
    vec2 ndc = vec2((pixel_pos.x / window_size.x) * 2.0 - 1.0, (pixel_pos.y / window_size.y) * 2.0 - 1.0);

    gl_Position = vec4(ndc, instance.z_order, 1.0);

    out_local_pos = (uv - 0.5) * instance.size;
    out_pixel_pos = pixel_pos;
    out_size = instance.size;
    out_background_color = unpackUnorm4x8(instance.background_color);
    out_border_color = unpackUnorm4x8(instance.border_color);
    out_border_radius = instance.border_radius;
    out_border_thickness = instance.border_thickness;
    const uint style_mask = instance.mask >> 16u;
    out_clip_id = instance.mask & 0xFFFFu;
    out_corner_mask = style_mask & 0xFu;
    out_flags = style_mask >> 4u;
}
