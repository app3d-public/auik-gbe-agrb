#pragma once

#include "agrb.hpp"
#include <agrb/pipeline.hpp>

namespace auik
{
    AUIK_GBE_AGRB_EXPORT bool construct_picker_pipeline(DrawPipeline &pipeline, agrb::device &device,
                                                        bool bind_pipeline = true);
    AUIK_GBE_AGRB_EXPORT bool configure_picker_pipeline(agrb::graphics_pipeline_batch::artifact &artifact,
                                                        DrawPipeline &pipeline);
} // namespace auik
