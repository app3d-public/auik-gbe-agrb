#pragma once

#include <agrb/descriptors.hpp>
#include <agrb/device.hpp>
#include <agrb/pipeline.hpp>
#include <auik-gbe-agrb/symbol_export.h>
#include <auik/detail/fwd.hpp>
#include <auik/detail/ids.hpp>

namespace auik
{
    AUIK_GBE_AGRB_EXPORT detail::GPUContext *create_agrb_backend(agrb::device &device, agrb::descriptor_pool *descriptor_pool);
    AUIK_GBE_AGRB_EXPORT TextureID add_agrb_texture(vk::Sampler sampler, vk::ImageView image_view,
                                           vk::ImageLayout image_layout = vk::ImageLayout::eShaderReadOnlyOptimal);
    AUIK_GBE_AGRB_EXPORT bool remove_agrb_texture(vk::ImageView image_view);
    AUIK_GBE_AGRB_EXPORT u32 get_agrb_texture_bind_slot(vk::ImageView image_view);
    AUIK_GBE_AGRB_EXPORT void clear_shader_cache(agrb::device &device);
    AUIK_GBE_AGRB_EXPORT bool configure_service_pipelines(agrb::graphics_pipeline_batch &batch, DrawPipeline *pipelines);
    AUIK_GBE_AGRB_EXPORT bool configure_default_streams(agrb::graphics_pipeline_batch &batch, DrawPipeline *pipelines,
                                               DrawStream *streams, u32 subpass, vk::RenderPass render_pass);

    struct DrawPipeline
    {
        vk::Pipeline handle = nullptr;
        vk::PipelineLayout layout = nullptr;
        acul::shared_ptr<agrb::descriptor_set_layout> descriptor_set_layout = nullptr;
    };

    inline void construct_pipeline_artifact(agrb::graphics_pipeline_batch::artifact &artifact, u32 subpass,
                                            DrawPipeline *pipeline)
    {
        artifact.tmp = acul::alloc<agrb::graphics_pipeline_batch::artifact::custom_data_t<u32>>(subpass);
        artifact.commit = [pipeline](vk::Pipeline handle) { pipeline->handle = handle; };
    }

    AUIK_GBE_AGRB_EXPORT void destroy_draw_pipeline(DrawPipeline &pipeline, agrb::device &device);
} // namespace auik
