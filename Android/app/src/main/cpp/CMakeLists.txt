cmake_minimum_required(VERSION 3.22.1)

project("brush_bridge")

# Build the JNI bridge library
add_library(brush_bridge SHARED
    brush_bridge.cpp
)

find_library(log-lib log)

# Link against system libraries only.
# libbrush_c.so is resolved dynamically at runtime via dlopen/dlsym,
# completely avoiding any host machine absolute paths in DT_NEEDED.
target_link_libraries(brush_bridge
    ${log-lib}
    dl
    m
)
