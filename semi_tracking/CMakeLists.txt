cmake_minimum_required(VERSION 2.8)
project(semi_tracking)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/cmake_modules/")
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_CFLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


enable_testing()

if (APPLE)
  add_definitions(-DAPPLE)
endif (APPLE)

find_package(Eigen3 3.1.3 REQUIRED)
find_package(GoogleLibs REQUIRED)
find_package(PNG REQUIRED STATIC)
find_package(JPEG REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV 3.4 REQUIRED) 
 # Optional : for features only.

set(SRC_DIR src)
set(INC_DIR include)
include_directories(${INC_DIR})
include_directories(${SRC_DIR})
include_directories(${GLOG_INCLUDE})
include_directories(${GFLAGS_INCLUDE})
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PNG_INCLUDE_DIR})
include_directories(${JPEG_INCLUDE_DIR})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(.)

#----------------------------------------
# monoslam libraries
#----------------------------------------
add_library(semi_tracking
	src/image_file.cc src/image_util.cc src/viewer.cc 
	src/semi_tracker.cc src/frame.cc src/semi_mapper.cc
  	src/optimizer.cc src/evaluator.cc
)
target_link_libraries(semi_tracking
    ${PNG_LIBRARY} ${CERES_LIBRARIES} ${GOOGLE_LIBRARIES} pthread
    ${OpenCV_LIBS} # Optional libraries.
)


#----------------------------------------
# monoslam Executables
#----------------------------------------
if (OpenCV_FOUND)
  add_executable(main src/main.cc)
	target_link_libraries(main semi_tracking)

   add_executable(euroc src/euroc_main.cc)
   target_link_libraries(euroc semi_tracking)

   add_executable(kitti src/kitti_main.cc)
   target_link_libraries(kitti semi_tracking)
endif (OpenCV_FOUND)


