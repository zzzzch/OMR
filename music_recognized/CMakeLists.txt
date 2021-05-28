project(music_recognized)

link_directories(
        ${Boost_LIBRARY_DIRS}
#        ${CMAKE_SOURCE_DIR}/base/include
)

add_executable(staff_detect
        src/staff_detect.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}
#        ${CMAKE_SOURCE_DIR}/base/src
#        ${CMAKE_SOURCE_DIR}/base/include
        )

target_include_directories(staff_detect
        PUBLIC
#        ${CMAKE_SOURCE_DIR}/base/include
        ${EIGEN_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${OPENCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${GLOG_INCLUDE_DIR}
        ${GFLAGS_INCLUDE_DIR}
        )


target_link_libraries(staff_detect
        PUBLIC
#        base
        ${Eigen_LIBRARIES}
        ${OpenCV_LIBS}
        ${GLOG_LIBRARY}
        ${gflags_LIBRARIES}
        )



