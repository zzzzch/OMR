project(Optical_Music_Recognition)


add_executable(Optical_Music_Recognition
        src/staff_split.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}
        #        ${CMAKE_SOURCE_DIR}/base/src
        #        ${CMAKE_SOURCE_DIR}/base/include
        )

target_include_directories(Optical_Music_Recognition
        PUBLIC
        #        ${CMAKE_SOURCE_DIR}/base/include
        ${EIGEN_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${OPENCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${GLOG_INCLUDE_DIR}
        ${GFLAGS_INCLUDE_DIR}
        )


target_link_libraries(Optical_Music_Recognition
        PUBLIC

        #        base
        ${Eigen_LIBRARIES}
        ${OpenCV_LIBS}
        ${GLOG_LIBRARY}
        ${gflags_LIBRARIES}
        )


