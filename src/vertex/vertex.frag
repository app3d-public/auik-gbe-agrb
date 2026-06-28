#version 460
#include <common/clip.glsl>

layout(location = 0) in vec4 in_color;
layout(location = 1) flat in uint in_clip_id;
layout(location = 2) in vec2 in_pixel_pos;

layout(location = 0) out vec4 out_color;

layout(std430, set = 0, binding = 0) readonly buffer ClipRectsBuffer { vec4 clip_rects[]; };

void main()
{
    if (is_clipped(in_pixel_pos, clip_rects[in_clip_id])) discard;
    out_color = in_color;
}
