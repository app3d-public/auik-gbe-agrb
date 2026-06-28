#version 460
#extension GL_EXT_nonuniform_qualifier : require
#include <common/clip.glsl>

layout(location = 0) in vec2 in_uv;
layout(location = 1) flat in vec4 in_tint_color;
layout(location = 2) flat in uint in_texture_id;
layout(location = 3) flat in uint in_clip_id;
layout(location = 4) flat in uint in_flags;
layout(location = 5) in vec2 in_pixel_pos;

layout(location = 0) out vec4 out_color;

layout(std430, set = 0, binding = 1) readonly buffer ClipRectsBuffer { vec4 clip_rects[]; };
layout(set = 1, binding = 0) uniform sampler2D ui_textures[];

#define TEXTURE_INSTANCE_TEXT_BIT 0x1u

void main()
{
    if (is_clipped(in_pixel_pos, clip_rects[in_clip_id])) discard;
    vec4 sampled = texture(ui_textures[nonuniformEXT(in_texture_id)], in_uv);
    if ((in_flags & TEXTURE_INSTANCE_TEXT_BIT) != 0u) out_color = vec4(in_tint_color.rgb, in_tint_color.a * sampled.r);
    else out_color = sampled;
}
