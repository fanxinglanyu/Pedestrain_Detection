cmake_minimum_required(VERSION 3.13)
#引入项目
project(pedestrian_detection)

set(CMAKE_CXX_STANDARD 14)

#找到opencv的包
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

#添加运行文件
add_executable( CropImage   test/detection_crop.cpp )
target_link_libraries( CropImage ${OpenCV_LIBS} )

add_executable( Detection test/detection.cpp  test/detection.h )
target_link_libraries( Detection ${OpenCV_LIBS} )

add_executable( Find_Save_Hardexample  test/find_save_hardexample.cpp)
target_link_libraries(Find_Save_Hardexample ${OpenCV_LIBS} )

add_executable(TestSvmModel test/test_main.cpp)
target_link_libraries(TestSvmModel ${OpenCV_LIBS} )

add_executable(DetectionCrop test/detection_crop.cpp)
target_link_libraries(DetectionCrop ${OpenCV_LIBS} )


#add_executable(Test test/test.cpp)
#target_link_libraries(Test ${OpenCV_LIBS} )

#动态链接
#target_link_libraries(pedestrian_detection ${OpenCV_LIBS} )
#target_link_libraries(test ${OpenCV_LIBS} )

