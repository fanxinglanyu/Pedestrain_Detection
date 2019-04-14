这是采用Hog和SVM的行人检测。
步骤
1,运行crop_image
剪裁负样本的图片

2，运行detection.cpp
进行HOG的提取和SVM的训练

3，运行find_save_hardexample训练hardneg

4，运行detection.cpp
生成最终的SVM模型

5，如果需要检测生成模型的xml,可以运训test_main.cpp

本机测试运行环境：mac+clion+opencv4.0
