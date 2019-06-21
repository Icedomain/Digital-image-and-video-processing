
# 声明一个 cmake 工程
project(image-processing)

# 设置编译模式
set(CMAKE_BUILD_TYPE "Released")
# set(CMAKE_FALGS_CXX "-std=c++11 -O")

# opencv 
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# head file path，头文件目录

#include_directories(include)


# 添加一个可执行程序
add_executable(hxy hxy.cpp)
#add_executable(hxy test.cpp)

# 将库文件链接到可执行程序上
target_link_libraries(hxy ${OpenCV_LIBS})



