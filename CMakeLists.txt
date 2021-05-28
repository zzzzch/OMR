cmake_minimum_required(VERSION 3.9)
project(OMR)

#set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_STANDARD 17)
set(CPP_VERSION 17)

#set(USE_ICP_GPU "ON")
#if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
#    if(${USE_ICP_GPU} STREQUAL "ON")
#        find_package(PCL 1.9 QUIET REQUIRED
#                COMPONENTS common kdtree octree search surface io ml sample_consensus filters geometry 2d features segmentation visualization registration gpu_features)
#    else()
#        find_package(PCL 1.9 QUIET REQUIRED
#                COMPONENTS common kdtree octree search surface io ml sample_consensus filters geometry 2d features segmentation visualization registration)
#    endif()
#else ()
#    find_package(PCL 1.9 QUIET REQUIRED
#            COMPONENTS common kdtree octree search surface io ml sample_consensus filters geometry 2d features segmentation registration)
#endif ()

find_package(Boost REQUIRED QUIET)
#find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
#find_package(GLEW REQUIRED)
#find_package(realsense2 2.39 REQUIRED)
#find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

## glog
find_package(Glog REQUIRED)
if(NOT GLOG_INCLUDE_DIR)
    set(GLOG_INCLUDE_DIR /usr/lib/x86_64-linux-gnu/)
    set(GLOG_LIBRARY /usr/local/lib/libglog.so)
endif(NOT GLOG_INCLUDE_DIR)


## gflags
find_package(Gflags REQUIRED)
if(NOT GFLAGS_INCLUDE_DIR)
    message("Gflags path is wrong , can't find it" )
endif(NOT GFLAGS_INCLUDE_DIR)

#if (POLICY CMP0072)
#    set(OpenGL_GL_PREFERENCE GLVND)
#endif()

## opengl
#find_package(OpenGL REQUIRED)
#if(NOT (OPENGL_INCLUDE_DIR OR OPENGL_LIBRARIES))
#    message("opengl path is wrong , can't find it" )
#    message("opengl include is " ${OPENGL_INCLUDE_DIR})
#    message("opengl lab is " ${OPENGL_LIBRARIES})
#endif(NOT (OPENGL_INCLUDE_DIR OR OPENGL_LIBRARIES))
#
#
## glut
#find_package(GLUT REQUIRED)
#if(NOT (GLUT_INCLUDE_DIR OR GLUT_LIBRARIES))
#    message("glut path is wrong , can't find it" )
#    message("glut include is " ${GLUT_INCLUDE_DIR})
#    message("glut lab is " ${GLUT_LIBRARIES})
#endif(NOT (GLUT_INCLUDE_DIR OR GLUT_LIBRARIES))

# nolman json
#find_package(nlohmann_json REQUIRED)

include_directories(
#        ${PCL_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIR}
        ${GFLAGS_INCLUDE_DIR}
        ${EIGEN_INCLUDE_DIRS}
        )

#add_subdirectory(base)
add_subdirectory(music_recognized)
add_subdirectory(Optical_Music_Recognition)
#add_subdirectory(imgui)
#add_subdirectory(FindCloud)
#add_subdirectory(localization)
#add_subdirectory(realsense_grap)
#add_subdirectory(test_gpu)
#add_subdirectory(cuda)

