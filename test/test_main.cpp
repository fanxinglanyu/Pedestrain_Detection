#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <sys/time.h>
#include "detection.h"


using namespace std;
using namespace cv;
using namespace cv::ml;

//读取HOG特征的xml文件
#define SvmLisrFile "/Users/macbookpro/CLionProjects/pedestrian_detection/data/SVM_HOG2.xml"
//读取测试的图片
#define TestImgFile "/Users/macbookpro/CLionProjects/pedestrian_detection/data/images4.jpg"

int main(){
    //检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
    HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);
//    int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
    Ptr<SVM> svm = SVM::create();// 创建分类器

    svm = SVM::load(SvmLisrFile);

    int svdim = svm ->getVarCount();//特征向量的维数，即HOG描述子的维数
    //支持向量的个数
    Mat svecsmat = svm ->getSupportVectors();//svecsmat元素的数据类型为float
    int numofsv = svecsmat.rows;

    // Mat alphamat = Mat::zeros(numofsv, svdim, CV_32F);//alphamat和svindex必须初始化，否则getDecisionFunction()函数会报错
    Mat alphamat = Mat::zeros(numofsv, svdim, CV_32F);
    Mat svindex = Mat::zeros(1, numofsv,CV_64F);
    cout << "after initialize the value of alphamat is  " << alphamat.size()  << endl;

    Mat Result;
    double rho = svm ->getDecisionFunction(0, alphamat, svindex);

    cout << "the value of rho is  " << rho << endl;
    alphamat.convertTo(alphamat, CV_32F);//将alphamat元素的数据类型重新转成CV_32F
    cout << "the value of alphamat is  " << alphamat << endl;
    cout << "the size of alphamat is  " << alphamat.size() << endl;
    cout << "the size of svecsmat is  " << svecsmat.size() << endl;

    //计算-(alphaMat * supportVectorMat),结果放到resultMat中
    Result = -1 * alphamat * svecsmat;//float

    cout << "the value of svdim is  " << svdim << endl;

    //得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
    vector<float> vec;
    //将resultMat中的数据复制到数组vec中
    for (int i = 0; i < svdim; ++i)
    {
        vec.push_back(Result.at<float>(0, i));
    }
    vec.push_back(rho);

    /*********************************Testing**************************************************/
    HOGDescriptor hog_test;
    hog_test.setSVMDetector(vec);

    // Mat src = imread("../person_and_bike_177b.png");
    Mat src = imread(TestImgFile);
    vector<Rect> found, found_filtered;
    hog_test.detectMultiScale(src, found, 0, Size(8,8), Size(32,32), 1.05, 2);

    cout<<"found.size : "<<found.size()<<endl;

    //找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中
    for(int i=0; i < found.size(); i++)
    {
        Rect r = found[i];
        int j=0;
        for(; j < found.size(); j++)
            if(j != i && (r & found[j]) == r)
                break;
        if( j == found.size())
            found_filtered.push_back(r);
    }


    //画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
    for(int i=0; i<found_filtered.size(); i++)
    {
        Rect r = found_filtered[i];
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
        rectangle(src, r.tl(), r.br(), Scalar(0,255,0), 3);
    }

    imwrite("ImgProcessed.jpg",src);
    namedWindow("src",0);
    imshow("src",src);
    waitKey(0);

    /******************读入单个64*128的测试图并对其HOG描述子进行分类*********************/
    ////读取测试图片(64*128大小)，并计算其HOG描述子
//    Mat testImg = imread("/Users/macbookpro/CLionProjects/pedestrian_detection/data/2.jpg");
//   // Mat testImg = imread("/Users/macbookpro/CLionProjects/pedestrian_detection/data/images5.jpg");
//    vector<float> descriptor;
//    hog.compute(testImg,descriptor,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
//    Mat testFeatureMat = Mat::zeros(1,3780,CV_32FC1);//测试样本的特征向量矩阵
//    //将计算好的HOG描述子复制到testFeatureMat矩阵中
//
//    for(int i=0; i<descriptor.size(); i++)
//    	testFeatureMat.at<float>(0,i) = descriptor[i];
//
//    //用训练好的SVM分类器对测试图片的特征向量进行分类
//    int result = svm->predict(testFeatureMat);//返回类标
//    cout<<"分类结果："<<result<<endl;

}