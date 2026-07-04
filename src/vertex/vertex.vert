#version 460

layout(location = 0) in vec4 in_pos_z;
layout(location = 1) in uvec2 in_data;

layout(set = 0, binding = 1) readonly buffer Offsets
{
    vec2 offsets[];
};

layout(push_constant) uniform Push { vec2 window_size; };

layout(location = 0) out vec4 out_color;
layout(location = 1) flat out uint out_clip_id;
layout(location = 2) out vec2 out_pixel_pos;

void main()
{
    uint batch_id = uint(in_pos_z.w + 0.5);
    vec2 pixel_pos = in_pos_z.xy + offsets[batch_id];
    vec2 ndc = vec2((pixel_pos.x / window_size.x) * 2.0 - 1.0, (pixel_pos.y / window_size.y) * 2.0 - 1.0);
    gl_Position = vec4(ndc, in_pos_z.z, 1.0);
    out_color = unpackUnorm4x8(in_data.x);
    out_clip_id = in_data.y;
    out_pixel_pos = pixel_pos;
}
