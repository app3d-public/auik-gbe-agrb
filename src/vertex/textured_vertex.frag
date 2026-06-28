#version 460
#extension GL_EXT_nonuniform_qualifier : require
#include <common/clip.glsl>

layout(location = 0) in vec2 in_uv;
layout(location = 1) flat in uint in_clip_id;
layout(location = 2) in vec2 in_pixel_pos;

layout(location = 0) out vec4 out_color;

layout(std430, set = 0, binding = 0) readonly buffer ClipRectsBuffer { vec4 clip_rects[]; };
layout(set = 1, binding = 0) uniform sampler2D ui_textures[];

layout(push_constant) uniform Push
{
    vec2 window_size;
    uint texture_id;
    uint flags;
};

void main()
{
    if (is_clipped(in_pixel_pos, clip_rects[in_clip_id])) discard;
    vec4 sampled = texture(ui_textures[nonuniformEXT(texture_id)], in_uv);
    out_color = ((flags & 1u) != 0u) ? vec4(1.0, 1.0, 1.0, sampled.r) : sampled;
}
