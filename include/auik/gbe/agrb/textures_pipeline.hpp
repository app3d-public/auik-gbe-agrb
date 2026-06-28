#pragma once

#include "agrb.hpp"
#include <agrb/pipeline.hpp>

namespace auik
{
    AUIK_GBE_AGRB_EXPORT bool construct_textures_pipeline(DrawPipeline &pipeline, agrb::device &device);
    AUIK_GBE_AGRB_EXPORT bool configure_textures_pipeline(agrb::graphics_pipeline_batch::artifact &artifact,
                                                vk::RenderPass render_pass, DrawPipeline &pipeline,
                                                agrb::device &device);
}
