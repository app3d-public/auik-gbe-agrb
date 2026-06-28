#version 460

layout(location = 0) in vec4 in_pos_z;
layout(location = 1) in uvec2 in_data;

layout(push_constant) uniform Push { vec2 window_size; };

layout(location = 0) out vec4 out_color;
layout(location = 1) flat out uint out_clip_id;
layout(location = 2) out vec2 out_pixel_pos;

void main()
{
    vec2 ndc = vec2((in_pos_z.x / window_size.x) * 2.0 - 1.0, (in_pos_z.y / window_size.y) * 2.0 - 1.0);
    gl_Position = vec4(ndc, in_pos_z.z, 1.0);
    out_color = unpackUnorm4x8(in_data.x);
    out_clip_id = in_data.y;
    out_pixel_pos = in_pos_z.xy;
}
