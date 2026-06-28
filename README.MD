# Auik Agrb GPU Backend

`auik-gbe-agrb` is the Agrb GPU backend for [Auik](../auik).
It connects Auik resources to the Agrb rendering layer.

## Features
- Default Auik draw stream pipelines
- GPU hit testing and picker support
- Auik shader compilation and packing through Agrb shader tools
- Single Draw Instancing (SDI)
- SDF-based rendering
- Bindless texture model

## Building

`auik-gbe-agrb` is distributed as a CMake module and is intended to be added after `auik` and `agrb`.

Before adding this module, the host project must define:

```cmake
set(AUIK_SHADERS_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
```

`AUIK_SHADERS_OUTPUT_DIR` is the directory where the generated Auik shader library is written.
At runtime, the backend loads:

```text
${AUIK_SHADERS_OUTPUT_DIR}/auik.umlib
```

Example:

```cmake
add_subdirectory(modules/auik)
add_subdirectory(modules/agrb)

set(AUIK_SHADERS_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
add_subdirectory(modules/auik-gbe-agrb)
```

## Usage

Create an Auik GPU backend from an initialized Agrb device:

```cpp
auto *gpu_backend = auik::create_agrb_backend(device, descriptor_pool);
```

Pass it to Auik through `auik::CreateInfo::set_gpu_backend()`.

## Dependencies

- [auik](../auik)
- [agrb](../agrb)
- [acul](../acul)
- [amal](../amal)

## License
This project is licensed under the [MIT License](LICENSE).

## Contacts
For any questions or feedback, you can reach out via [email](mailto:wusikijeronii@gmail.com) or open a new issue.
