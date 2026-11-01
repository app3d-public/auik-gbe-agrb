#include <auik/gbe/agrb/agrb.hpp>
#include <auik/detail/gpu_context.hpp>
#include <cassert>
#include <string>

int main(int argc, char **argv)
{
    assert(argc == 4);
    agrb::init_library();
    umbf::registry::HashResolver resolver;
    umbf::insert_default_segment_codecs(resolver);
    resolver.block_streams[static_cast<u32>(AGRB_SIGN_ID_SHADER)] = &agrb::streams::shader;
    auto *previous = umbf::registry::resolver;
    umbf::registry::resolver = &resolver;
    {
        agrb::device device;
        for (int i = 1; i <= 2; ++i)
        {
            bool rejected = false;
            try
            {
                auto *backend = auik::create_agrb_backend(device, nullptr, acul::path(argv[i]));
                backend->destroy_context(backend);
            }
            catch (const std::exception &e)
            {
                rejected = true;
                assert(std::string(e.what()).find(argv[i]) != std::string::npos);
            }
            assert(rejected);
        }
        // Loading and teardown need no GPU operations and work outside the build tree.
        auto *backend = auik::create_agrb_backend(device, nullptr, acul::path(argv[3]));
        assert(backend);
        backend->destroy_context(backend);
    }
    umbf::registry::resolver = previous;
    agrb::destroy_library();
}
