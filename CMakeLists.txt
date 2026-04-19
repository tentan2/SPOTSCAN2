cmake_minimum_required(VERSION 3.16)
project(Spotscan_CPP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Get the absolute path of the current directory
get_filename_component(CURRENT_DIR_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

if("${CMAKE_CURRENT_SOURCE_DIR}" MATCHES "infinityintegral")
    set(Torch_DIR "/usr/local/libtorch/share/cmake/Torch")
endif()

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(nlohmann_json REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

# Add executable
add_executable(spotscan_cpp
    src/main.cpp
    src/model_manager.cpp
    src/image_processor.cpp
    src/analyzers/food_detector.cpp
    src/analyzers/nutrition_analyzer.cpp
    src/analyzers/vit_analyzer.cpp
    src/ui/main_window.cpp
    src/ui/analysis_panel.cpp
)

# Link libraries
target_link_libraries(spotscan_cpp
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    nlohmann_json::nlohmann_json
)

# Compiler options
target_compile_options(spotscan_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
    $<$<COMPILE_LANGUAGE:CXX>:/O3>
)

# Copy models directory to build
file(COPY ${CMAKE_SOURCE_DIR}/models DESTINATION ${CMAKE_BINARY_DIR}/models)

# Install targets
install(TARGETS spotscan_cpp DESTINATION bin)
install(DIRECTORY models DESTINATION bin/models)
