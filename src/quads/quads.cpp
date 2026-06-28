#include <acul/scalars.hpp>
#include <acul/string/string.hpp>
#include <auik-gbe-agrb/shaders.h>
#include <auik/gbe/agrb/quads_pipeline.hpp>
#include <auik/pipelines.hpp>
#include "../instance_stream.hpp"

namespace auik::detail
{
    using QuadsStream = InstanceStream<QuadsInstanceData>;

    void render_quads_stream(DrawStream *stream, void *render_ctx, GPUContext *gpu_context, u32 frame_id)
    {
        if (stream->draw_sizes[frame_id] == 0) return;
        auto &gpu_data = static_cast<QuadsStream *>(stream->stream_instances)[frame_id];
        if (!update_instance_descriptor_set(stream, gpu_data, gpu_context, frame_id)) return;
        auto *pipeline = stream->pipeline;
        auto &device = get_agrb_device(gpu_context);
        auto &cmd = *static_cast<vk::CommandBuffer *>(render_ctx);
        auto &loader = device.loader;
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->handle, loader);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline->layout, 0, 1, &gpu_data.descriptor_set, 0,
                               nullptr, loader);
        cmd.pushConstants(pipeline->layout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(amal::vec2),
                          &get_display_size(), loader);
        cmd.draw(6, stream->draw_sizes[frame_id], 0, 0, loader);
    }

    void init_quads_pipeline_calls(StreamGPUDispatch &dispatch)
    {
        dispatch.push_data_to_stream = &push_data_to_instance_stream<QuadsInstanceData>;
        dispatch.push_data_batch_to_stream = &push_data_batch_to_instance_stream<QuadsInstanceData>;
        dispatch.update_stream_data = &update_instance_stream_data<QuadsInstanceData>;
        dispatch.update_stream_data_batch = &update_instance_stream_data_batch<QuadsInstanceData>;
        dispatch.invalidate_stream_data = &invalidate_instance_stream_data<QuadsInstanceData>;
        dispatch.clear_stream = &clear_instance_stream<QuadsStream>;
        dispatch.copy_stream_frame_data = &copy_instance_stream_frame_data<QuadsStream>;
        dispatch.sync_stream_cache = &sync_instance_stream_cache<QuadsStream>;
        dispatch.render_stream = &render_quads_stream;
        dispatch.create_stream_gpu_data = &create_instance_stream_gpu_data<QuadsStream>;
        dispatch.destroy_stream_gpu_data = &destroy_instance_stream_gpu_data<QuadsStream>;
    }

} // namespace auik::detail

namespace auik
{
    bool construct_quads_pipeline(DrawPipeline &pipeline, agrb::device &device)
    {
        pipeline.descriptor_set_layout =
            agrb::descriptor_set_layout::builder()
                .add_binding(0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eVertex)
                .add_binding(1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eFragment)
                .build(device);
        if (!pipeline.descriptor_set_layout) return false;

        const vk::DescriptorSetLayout set_layouts[] = {pipeline.descriptor_set_layout->layout()};
        const vk::PushConstantRange push_constant{vk::ShaderStageFlagBits::eVertex, 0, sizeof(amal::vec2)};
        vk::PipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = set_layouts;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_constant;
        pipeline.layout = device.vk_device.createPipelineLayout(pipeline_layout_info, nullptr, device.loader);
        return pipeline.layout != nullptr;
    }

    bool configure_quads_pipeline(agrb::graphics_pipeline_batch::artifact &artifact, vk::RenderPass render_pass,
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
        auto vs = ctx->shader_cache.get_shader(AS_AUIK_QUADS_VS, shaders[0], device, path);
        if (!vs.success()) return false;
        auto fs = ctx->shader_cache.get_shader(AS_AUIK_QUADS_FS, shaders[1], device, path);
        if (!fs.success()) return false;
        agrb::prepare_base_graphics_pipeline(artifact, shaders, device);
        return true;
    }
} // namespace auik
