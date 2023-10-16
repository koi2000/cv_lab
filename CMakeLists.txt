cmake_minimum_required(VERSION 3.17)
project(cv)

set(CMAKE_CXX_STANDARD 11)
# 设置OpenCV目录所在
set(OpenCV_DIR "F:\\C++\\OpenCV\\opencv34\\opencv\\builds")
# 搜索OpenCV目录
find_package(OpenCV REQUIRED)

# 添加OpenCV头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cvlab1_1 lab1/lab1_1.cpp)
add_executable(cvlab1_2 lab1/lab1_2.cpp)
add_executable(cvlab2 lab2/lab2.cpp)
add_executable(cvlab3 lab2/lab2.cpp)
link_directories(${OpenCV_LIBRARY_DIRS})
# 链接OpenCV库文件
target_link_libraries(cvlab1_1 ${OpenCV_LIBS})
target_link_libraries(cvlab1_2 ${OpenCV_LIBS})
target_link_libraries(cvlab2 ${OpenCV_LIBS})
target_link_libraries(cvlab3 ${OpenCV_LIBS})