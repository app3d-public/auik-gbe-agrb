#version 460

struct PickerStyleData
{
    uint widget_id;
    uint tag_id;
    uint element_id;
    float hit_depth;
    float depth;
    uint clip_and_flags;
};

layout(std430, set = 0, binding = 0) readonly buffer RectBuffer { vec4 rects[]; };
layout(std430, set = 0, binding = 1) readonly buffer StyleBuffer { PickerStyleData styles[]; };

layout(push_constant) uniform Push { vec2 window_size; };

layout(location = 0) flat out uint out_widget_id;
layout(location = 1) flat out uint out_tag_id;
layout(location = 2) flat out uint out_element_id;
layout(location = 3) flat out uint out_clip_and_flags;
layout(location = 4) out vec2 out_pixel_pos;
layout(location = 5) out vec2 out_local_pos;
layout(location = 6) flat out vec2 out_rect_size;

#define AUIK_HITBOX_PAD 4.0

vec2 get_quad_uv(uint vertex_index)
{
    const vec2 uv[6] =
        vec2[6](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));
    return uv[vertex_index];
}

void main()
{
    vec4 rect = rects[gl_InstanceIndex];
    PickerStyleData style = styles[gl_InstanceIndex];
    vec2 uv = get_quad_uv(uint(gl_VertexIndex));
    bool is_hitbox = (((style.clip_and_flags >> 16u) & 0x1u) != 0u);
    vec2 draw_pos = rect.xy;
    vec2 draw_size = rect.zw;
    if (is_hitbox)
    {
        draw_pos -= vec2(AUIK_HITBOX_PAD);
        draw_size += vec2(AUIK_HITBOX_PAD * 2.0);
    }

    vec2 pixel_pos = draw_pos + uv * draw_size;
    vec2 ndc = vec2((pixel_pos.x / window_size.x) * 2.0 - 1.0, (pixel_pos.y / window_size.y) * 2.0 - 1.0);

    gl_Position = vec4(ndc, style.hit_depth, 1.0);
    out_widget_id = style.widget_id;
    out_tag_id = style.tag_id;
    out_element_id = style.element_id;
    out_clip_and_flags = style.clip_and_flags;
    out_pixel_pos = pixel_pos;
    out_local_pos = pixel_pos - (rect.xy + rect.zw * 0.5);
    out_rect_size = rect.zw;
}
