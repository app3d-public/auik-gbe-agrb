#pragma once

#include <acul/io/path.hpp>
#include <acul/vector.hpp>
#include <agrb/descriptors.hpp>
#include <agrb/device.hpp>
#include <agrb/pipeline.hpp>
#include <agrb/vector.hpp>
#include <amal/vector.hpp>
#include <auik/detail/gpu_context.hpp>
#include "picker/picker.hpp"

namespace auik::detail
{
    struct AgrbContext final : GPUContext
    {
        agrb::device &device;
        agrb::descriptor_pool *descriptor_pool = nullptr;
        agrb::shader_cache shader_cache;
        acul::shared_ptr<agrb::descriptor_set_layout> bindless_texture_layout = nullptr;
        vk::DescriptorSet bindless_texture_set = nullptr;
        acul::vector<vk::DescriptorImageInfo> bindless_textures;
        agrb::vector<amal::vec4> *clip_rects = nullptr;
        bool *clip_rects_reallocated = nullptr;
        acul::unique_ptr<class GPUPicker> picker;

        AgrbContext(agrb::device &device, agrb::descriptor_pool *descriptor_pool)
            : device(device), descriptor_pool(descriptor_pool)
        {
        }
    };

    inline AgrbContext *get_agrb_context(GPUContext *gpu_ctx)
    {
        assert(gpu_ctx && "gpu_backend is null");
        return static_cast<AgrbContext *>(gpu_ctx);
    }

    inline agrb::device &get_agrb_device(GPUContext *gpu_backend) { return get_agrb_context(gpu_backend)->device; }

    inline const acul::path &get_shader_library_path()
    {
        static acul::path path = acul::path(AUIK_SHADERS_OUTPUT_DIR) / "auik.umlib";
        return path;
    }
} // namespace auik::detail
