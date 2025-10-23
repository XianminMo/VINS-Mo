# cmake_minimum_required(VERSION 2.8.3)
cmake_minimum_required(VERSION 3.5)

project(vins_estimator)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "-std=c++14")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    visualization_msgs
)

# --- 查找 ONNX Runtime ---
# ${PROJECT_SOURCE_DIR} 指向 vins_estimator 目录
set(ONNX_RUNTIME_DIR ${PROJECT_SOURCE_DIR}/third_party/onnx_runtime)

# 将 ONNX 头文件目录添加到 include 路径
include_directories(
    ${ONNX_RUNTIME_DIR}/include
)

# 将 ONNX 库目录添加到 link 路径
link_directories(
    ${ONNX_RUNTIME_DIR}/lib
)

# find_package(OpenCV REQUIRED)
find_package(OpenCV 4 REQUIRED)

# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

find_package(Ceres REQUIRED)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

catkin_package()

add_executable(vins_estimator
    src/estimator_node.cpp
    src/parameters.cpp
    src/estimator.cpp
    src/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projection_factor.cpp
    src/factor/projection_td_factor.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    src/initial/depth_estimator.cpp
    src/initial/initial_fast_mono.cpp
    )


target_link_libraries(vins_estimator 
    ${catkin_LIBRARIES} 
    ${OpenCV_LIBS} 
    ${CERES_LIBRARIES}
    ${EIGEN3_LIBS}
    "onnxruntime"
    ${CMAKE_DL_LIBS}
) 
