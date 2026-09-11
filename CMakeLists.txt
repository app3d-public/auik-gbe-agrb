cmake_minimum_required(VERSION 3.17)

if(NOT DEFINED ACBT_PROJECT_LOADED)
    include(cmake/project.cmake)
endif()

project(auik-gbe-agrb
    VERSION 0.0.1
    LANGUAGES CXX C
    DESCRIPTION "Auik agrb GPU Backend")

if(NOT DEFINED ACBT_LOADED)
    include(cmake/utils.cmake)
endif()

include(${CMAKE_SOURCE_DIR}/modules/agrb/tools/shader_tools.cmake)

create_symbol_export_file()
add_files(${PROJECT_NAME} RECURSE "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_library(${PROJECT_NAME} SHARED ${AUIK_GBE_AGRB_SRC})

if(NOT DEFINED AUIK_SHADERS_OUTPUT_DIR)
    message(FATAL_ERROR "Shaders output dir must be set")
endif()

add_shaders(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/env.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders.yaml"
    "${AUIK_SHADERS_OUTPUT_DIR}"
    TARGET_PREFIX AUIK_GBE_AGRB_SHADERS
    BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/src/shaders"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src"
    S2U_TARGET agrb_s2u
)
set(AUIK_GBE_AGRB_SHADER_HEADER_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/src/shaders/auik/shaders.h")
set(AUIK_GBE_AGRB_SHADER_HEADER "${CMAKE_CURRENT_BINARY_DIR}/include/auik-gbe-agrb/shaders.h")
add_custom_command(
    OUTPUT "${AUIK_GBE_AGRB_SHADER_HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/include/auik-gbe-agrb"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${AUIK_GBE_AGRB_SHADER_HEADER_SOURCE}"
            "${AUIK_GBE_AGRB_SHADER_HEADER}"
    DEPENDS "${AUIK_GBE_AGRB_SHADER_HEADER_SOURCE}"
    COMMENT "Generating auik-gbe-agrb shader header"
    VERBATIM
)
add_custom_target(AUIK_GBE_AGRB_SHADER_HEADER DEPENDS "${AUIK_GBE_AGRB_SHADER_HEADER}")
add_dependencies(${PROJECT_NAME} AUIK_GBE_AGRB_SHADERS AUIK_GBE_AGRB_SHADER_HEADER)

set_target_properties(${PROJECT_NAME} PROPERTIES
    DEFINE_SYMBOL AUIK_GBE_AGRB_BUILD
    CXX_VISIBILITY_PRESET hidden
)
file(TO_CMAKE_PATH "${AUIK_SHADERS_OUTPUT_DIR}" AUIK_SHADERS_OUTPUT_DIR_CMAKE)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    "AUIK_SHADERS_OUTPUT_DIR=\"${AUIK_SHADERS_OUTPUT_DIR_CMAKE}\""
)

target_include_directories(${PROJECT_NAME} PUBLIC
    "${CMAKE_CURRENT_BINARY_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${AUIK_GBE_AGRB_CONFIGURE_PATH}"
)
target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/src/shaders"
    "${CMAKE_SOURCE_DIR}/modules/auik/src"
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC auik agrb
    PRIVATE acul amal
)

if(WIN32)
    # Manifest
    gen_manifest_lib(lib${PROJECT_NAME} ${PROJECT_VERSION})
endif()

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    CXX_EXTENSIONS YES
)

option(AUIK_GBE_AGRB_BUILD_TESTS "Build backend cache tests" OFF)
if(AUIK_GBE_AGRB_BUILD_TESTS)
    enable_testing()
    add_executable(auik_picker_cache_test tests/picker_cache.cpp src/picker/picker.cpp)
    target_include_directories(auik_picker_cache_test PRIVATE "${CMAKE_SOURCE_DIR}/modules/auik/src")
    target_link_libraries(auik_picker_cache_test PRIVATE auik-gbe-agrb acul amal)
    target_compile_options(auik_picker_cache_test PRIVATE -UNDEBUG)
    target_compile_definitions(auik_picker_cache_test PRIVATE
        "AUIK_SHADERS_OUTPUT_DIR=\"${AUIK_SHADERS_OUTPUT_DIR_CMAKE}\"")
    add_dependencies(auik_picker_cache_test AUIK_GBE_AGRB_SHADER_HEADER)
    add_test(NAME auik_picker_cache COMMAND auik_picker_cache_test)
endif()
