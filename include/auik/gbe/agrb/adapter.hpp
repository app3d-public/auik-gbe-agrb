#pragma once

#include <agrb/device_adapter.hpp>

namespace auik
{
    inline bool initialize_agrb_adopted_device(agrb::device &device, agrb::adopted_device_create_info &create_info)
    {
        create_info.set_create_command_pools(true)
            .set_graphics_command_buffers(2, 0)
            .set_create_fence_pool(true)
            .set_fence_pool_size(2);
        agrb::initialize_adopted_device(device, create_info);
        if (!device.allocator) return agrb::create_adopted_allocator(device);
        return true;
    }
} // namespace auik
