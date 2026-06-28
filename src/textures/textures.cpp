#include <acul/scalars.hpp>
#include <acul/string/string.hpp>
#include <auik-gbe-agrb/shaders.h>
#include <auik/gbe/agrb/textures_pipeline.hpp>
#include <auik/pipelines.hpp>
#include "../instance_stream.hpp"

namespace auik::detail
{
    using TexturesStream = InstanceStream<TexturesInstanceData>;

    void render_textures_stream(DrawStream *stream, void *render_ctx, GPUContext *gpu_context, u32 frame_id)
    {
        if (stream->draw_sizes[frame_id] == 0) return;
        auto *ctx = get_agrb_context(gpu_context);
        auto &gpu_data = static_cast<TexturesStream *>(stream->stream_instances)[frame_id];
        if (!ctx->bindless_texture_set)
            return;
        if (!update_instance_descriptor_set(stream, gpu_data, gpu_context, frame_id))
            return;
        auto *pipeline = stream->pipeline;
        auto &device = get_agrb_device(gpu_context);
        auto &cmd = *static_cast<vk::CommandBuffer *>(render_ctx);
        auto &loader = device.loader;
        const vk::DescriptorSet descriptor_sets[] = {gpu_data.descriptor_set, ctx->bindless_texture_set};
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->handle, loader);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline->layout, 0, 2, descriptor_sets, 0, nullptr,
                               loader);
        cmd.pushConstants(pipeline->layout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(amal::vec2),
                          &get_display_size(), loader);
        cmd.draw(6, stream->draw_sizes[frame_id], 0, 0, loader);
    }

    void init_textures_pipeline_calls(StreamGPUDispatch &dispatch)
    {
        dispatch.push_data_to_stream = &push_data_to_instance_stream<TexturesInstanceData>;
        dispatch.push_data_batch_to_stream = &push_data_batch_to_instance_stream<TexturesInstanceData>;
        dispatch.update_stream_data = &update_instance_stream_data<TexturesInstanceData>;
        dispatch.update_stream_data_batch = &update_instance_stream_data_batch<TexturesInstanceData>;
        dispatch.invalidate_stream_data = &invalidate_instance_stream_data<TexturesInstanceData>;
        dispatch.clear_stream = &clear_instance_stream<TexturesStream>;
        dispatch.copy_stream_frame_data = &copy_instance_stream_frame_data<TexturesStream>;
        dispatch.sync_stream_cache = &sync_instance_stream_cache<TexturesStream>;
        dispatch.render_stream = &render_textures_stream;
        dispatch.create_stream_gpu_data = &create_instance_stream_gpu_data<TexturesStream>;
        dispatch.destroy_stream_gpu_data = &destroy_instance_stream_gpu_data<TexturesStream>;
    }
} // namespace auik::detail

namespace auik
{
    bool construct_textures_pipeline(DrawPipeline &pipeline, agrb::device &device)
    {
        pipeline.descriptor_set_layout =
            agrb::descriptor_set_layout::builder()
                .add_binding(0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eVertex)
                .add_binding(1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eFragment)
                .build(device);
        if (!pipeline.descriptor_set_layout) return false;

        auto *ctx = detail::get_agrb_context(detail::get_context().gpu_ctx);
        assert(ctx && ctx->bindless_texture_layout);
        const vk::DescriptorSetLayout set_layouts[] = {pipeline.descriptor_set_layout->layout(),
                                                       ctx->bindless_texture_layout->layout()};
        const vk::PushConstantRange push_constant{vk::ShaderStageFlagBits::eVertex, 0, sizeof(amal::vec2)};
        vk::PipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.setLayoutCount = 2;
        pipeline_layout_info.pSetLayouts = set_layouts;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_constant;
        pipeline.layout = device.vk_device.createPipelineLayout(pipeline_layout_info, nullptr, device.loader);
        return pipeline.layout != nullptr;
    }

    bool configure_textures_pipeline(agrb::graphics_pipeline_batch::artifact &artifact, vk::RenderPass render_pass,
                                     DrawPipeline &pipeline, agrb::device &device)
    {
        auto *tmp = static_cast<agrb::graphics_pipeline_batch::artifact::custom_data_t<u32> *>(artifact.tmp);
        if (!tmp) return false;

        artifact.config.load_defaults().enable_alpha_blending();
        auto &blend = artifact.config.color_blend_attachment;
        blend.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
        blend.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
        blend.colorBlendOp = vk::BlendOp::eAdd;
        blend.srcAlphaBlendFactor = vk::BlendFactor::eOne;
        blend.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
        blend.alphaBlendOp = vk::BlendOp::eAdd;
        artifact.config.depth_stencil_info.setDepthTestEnable(true).setDepthWriteEnable(true).setDepthCompareOp(
            vk::CompareOp::eGreaterOrEqual);
        artifact.config.render_pass = render_pass;
        artifact.config.pipeline_layout = pipeline.layout;
        artifact.config.subpass = tmp->value;

        auto *ctx = detail::get_agrb_context(detail::get_context().gpu_ctx);
        if (!ctx) return false;

        const auto &path = detail::get_shader_library_path();
        vk::ShaderModule shaders[2];
        auto vs =ctx->shader_cache.get_shader(AS_AUIK_TEXTURES_VS, shaders[0], device, path);
        if (!vs.success()) return false;
        auto fs = ctx->shader_cache.get_shader(AS_AUIK_TEXTURES_FS, shaders[1], device, path);
        if (!fs.success()) return false;
        agrb::prepare_base_graphics_pipeline(artifact, shaders, device);
        return true;
    }
} // namespace auik
