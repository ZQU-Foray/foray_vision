cmake_minimum_required(VERSION 3.8)
project(foray_vision)

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS
        ${ONNXRUNTIME_DIR}/include
        $ENV{ONNXRUNTIME_DIR}/include
        $ENV{HOME}/vision_learn/onnxruntime_include
)
find_library(ONNXRUNTIME_LIB onnxruntime
    HINTS
        ${ONNXRUNTIME_DIR}/lib
        $ENV{ONNXRUNTIME_DIR}/lib
        $ENV{HOME}/yolo_env/lib/python3.10/site-packages/onnxruntime/capi
)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime not found; set ONNXRUNTIME_DIR to its installation root")
endif()

add_library(${PROJECT_NAME} SHARED
    src/pnp_solver.cpp
    src/armor_pipeline.cpp
    src/pattern_match.cpp
    src/dual_detector.cpp
    src/sp_lightbar_geometry.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen ${OpenCV_LIBS} ${ONNXRUNTIME_LIB} Threads::Threads)
get_filename_component(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_LIB}" DIRECTORY)
set_target_properties(${PROJECT_NAME} PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_LIBRARY_DIR}"
)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

install(TARGETS ${PROJECT_NAME}
    EXPORT export_${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 OpenCV)

if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(test_pnp test/test_pnp.cpp)
    target_link_libraries(test_pnp ${PROJECT_NAME})

    ament_add_gtest(test_armor_pipeline test/test_armor_pipeline.cpp)
    target_link_libraries(test_armor_pipeline ${PROJECT_NAME})

    ament_add_gtest(test_pattern_match test/test_pattern_match.cpp)
    target_link_libraries(test_pattern_match ${PROJECT_NAME})

    ament_add_gtest(test_dual_detector test/test_dual_detector.cpp)
    target_link_libraries(test_dual_detector ${PROJECT_NAME})

    ament_add_gtest(test_sp_lightbar_geometry test/test_sp_lightbar_geometry.cpp)
    target_link_libraries(test_sp_lightbar_geometry ${PROJECT_NAME})

endif()

ament_package()
