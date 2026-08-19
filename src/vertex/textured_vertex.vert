#version 460

layout(location = 0) in vec4 in_pos_z;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in uint in_clip_id;

struct TexturedVertexBatchStyle
{
    uint texture_id;
    uint flags;
};

layout(set = 0, binding = 1) readonly buffer Offsets
{
    vec2 offsets[];
};

layout(set = 0, binding = 2) readonly buffer Styles
{
    TexturedVertexBatchStyle styles[];
};

layout(push_constant) uniform Push
{
    vec2 window_size;
};

layout(location = 0) out vec2 out_uv;
layout(location = 1) flat out uint out_clip_id;
layout(location = 2) out vec2 out_pixel_pos;
layout(location = 3) flat out uint out_texture_id;
layout(location = 4) flat out uint out_flags;

void main()
{
    uint batch_id = uint(in_pos_z.w + 0.5);
    TexturedVertexBatchStyle style = styles[batch_id];
    vec2 pixel_pos = in_pos_z.xy + offsets[batch_id];
    vec2 ndc = vec2((pixel_pos.x / window_size.x) * 2.0 - 1.0, (pixel_pos.y / window_size.y) * 2.0 - 1.0);
    gl_Position = vec4(ndc, in_pos_z.z, 1.0);
    out_uv = in_uv;
    out_clip_id = in_clip_id;
    out_pixel_pos = pixel_pos;
    out_texture_id = style.texture_id;
    out_flags = style.flags;
}
