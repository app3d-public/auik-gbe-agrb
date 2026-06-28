#version 460

struct PickerInstanceData
{
    uint widget_id;
    uint tag_id;
    uint element_id;
    float hit_depth;
    vec2 position;
    vec2 size;
    float depth;
    uint clip_and_flags;
};

layout(std430, set = 0, binding = 0) readonly buffer PickerBuffer { PickerInstanceData instances[]; };

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
    PickerInstanceData instance = instances[gl_InstanceIndex];
    vec2 uv = get_quad_uv(uint(gl_VertexIndex));
    bool is_hitbox = (((instance.clip_and_flags >> 16u) & 0x1u) != 0u);
    vec2 draw_pos = instance.position;
    vec2 draw_size = instance.size;
    if (is_hitbox)
    {
        draw_pos -= vec2(AUIK_HITBOX_PAD);
        draw_size += vec2(AUIK_HITBOX_PAD * 2.0);
    }

    vec2 pixel_pos = draw_pos + uv * draw_size;
    vec2 ndc = vec2((pixel_pos.x / window_size.x) * 2.0 - 1.0, (pixel_pos.y / window_size.y) * 2.0 - 1.0);

    gl_Position = vec4(ndc, instance.hit_depth, 1.0);
    out_widget_id = instance.widget_id;
    out_tag_id = instance.tag_id;
    out_element_id = instance.element_id;
    out_clip_and_flags = instance.clip_and_flags;
    out_pixel_pos = pixel_pos;
    out_local_pos = pixel_pos - (instance.position + instance.size * 0.5);
    out_rect_size = instance.size;
}
