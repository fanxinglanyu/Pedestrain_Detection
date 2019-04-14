//
// Created by MacBook Pro on 2019-04-08.
//

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
//#include "dataset.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include<iomanip>

using namespace std;
using namespace cv;
using namespace cv::ml;

int HardExampleCount = 1;
#define cropHardNegNum 10896

#define NegFile "/Users/macbookpro/CLionProjects/pedestrian_detection/img_dir/sample_neg.txt"
#define HardFile "/Users/macbookpro/CLionProjects/pedestrian_detection/img_dir/hard_neg4.txt"
#define SvmLoadFile "/Users/macbookpro/CLionProjects/pedestrian_detection/data/SVM_HOG7.xml"
#define NegImageFile "/Users/macbookpro/CLionProjects/pedestrian_detection/normalized_images/train/neg/"

class MySVM : public  ml::SVM
{
public:
    //获得SVM的决策函数中的alpha数组
    double get_svm_rho()
    {
        return this->getDecisionFunction(0, svm_alpha, svm_svidx);
    }

    //获得SVM的决策函数中的rho参数,即偏移量

    vector<float> svm_alpha;
    vector<float> svm_svidx;
    float  svm_rho;

};

int main(int argc, char** argv)
{
    Mat src;
    string ImgName;

    char saveName[256];//找出来的HardExample图片文件名
    //打开原始负样本图片文件列表
    ifstream fin(NegFile);

    ofstream fout(HardFile,ios::trunc);//加路径
   // int num = 1;
    //检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
    //HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);//HOG检测器，用来计算HOG描述子的
    int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
    ///MySVM svm;//SVM分类器
   // Ptr<SVM> svm = SVM::create();// 创建分类器
    ///svm = svm::load("SVM_HOG.xml");
    //svm ->save("/Users/macbookpro/CLionProjects/pedestrian_detection/data/SVM_HOG7.xml");

    Ptr<SVM> svm = Algorithm::load<SVM>(SvmLoadFile);

    /*************************************************************************************************
      线性SVM训练完成后得到的XML文件里面，有一个数组，叫做support vector，还有一个数组，叫做alpha,有一个浮点数，叫做rho;
      将alpha矩阵同support vector相乘，注意，alpha*supportVector,将得到一个列向量。之后，再该列向量的最后添加一个元素rho。
      如此，变得到了一个分类器，利用该分类器，直接替换opencv中行人检测默认的那个分类器（cv::HOGDescriptor::setSVMDetector()），
      就可以利用你的训练样本训练出来的分类器进行行人检测了。
      ***************************************************************************************************/
    DescriptorDim = svm->getVarCount();//特征向量的维数，即HOG描述子的维数
    Mat supportVector = svm->getSupportVectors();//支持向量的个数
    int supportVectorNum = supportVector.rows;
    //cout<<"支持向量个数："<<supportVectorNum<<endl;

    vector<float> svm_alpha;
    vector<float> svm_svidx;
    float  svm_rho;
    svm_rho = svm->getDecisionFunction(0, svm_alpha, svm_svidx);

    Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数
    Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵
    Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果


    //将支持向量的数据复制到supportVectorMat矩阵中

    supportVectorMat = supportVector;

    //将alpha向量的数据复制到alphaMat中
    ///double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量
    for(int i = 0; i < supportVectorNum; i++)
    {
        alphaMat.at<float>(0,i) = svm_alpha[i];
    }

    //计算-(alphaMat * supportVectorMat),结果放到resultMat中
    //gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);
    resultMat = -1 * alphaMat * supportVectorMat;

    //得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
    vector<float> myDetector;
    //将resultMat中的数据复制到数组myDetector中
    for(int i=0; i<DescriptorDim; i++)
    {
        myDetector.push_back(resultMat.at<float>(0,i));
    }

    //最后添加偏移量rho，得到检测子
   /// myDetector.push_back(svm.get_rho());
    myDetector.push_back(svm_rho);
    cout<<"检测子维数："<<myDetector.size()<<endl;

    //设置HOGDescriptor的检测子
    HOGDescriptor myHOG;
    myHOG.setSVMDetector(myDetector);
    //myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

//	//保存检测子参数到文件
//	ofstream fout("HOGDetectorForOpenCV.txt");
//	for(int i=0; i<myDetector.size(); i++)
//	{
//		fout<<myDetector[i]<<endl;
//	}

//  namedWindow("people detector", 1);

    while(getline(fin,ImgName))
    {
        cout<<"处理："<<ImgName<<endl;
        ImgName = NegImageFile + ImgName;
        src = imread(ImgName,1);//读取图片

        Mat img = src.clone();//复制原图

        vector<Rect> found, found_filtered;
        //double t = (double)getTickCount();
        // run the detector with default parameters. to get a higher hit-rate
        // (and more false alarms, respectively), decrease the hitThreshold and
        // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).

        //对负样本原图进行多尺度检测，检测出的都是误报
        myHOG.detectMultiScale(src, found, 0, Size(8,8), Size(32,32), 1.05, 2);
        //t = (double)getTickCount() - t;
        //printf("tdetection time = %gms\n", t*1000./cv::getTickFrequency());


        //遍历从图像中检测出来的矩形框，得到hard example
        size_t i, j;
        for( i = 0; i < found.size(); i++ )
        {
            Rect r = found[i];
            for( j = 0; j < found.size(); j++ )
                if( j != i && (r & found[j]) == r)
                    break;
            if( j == found.size() )
                found_filtered.push_back(r);
        }

        for( i = 0; i < found_filtered.size(); i++ )
        {
            Rect r = found_filtered[i];
            // the HOG detector returns slightly larger rectangles than the real objects.
            // so we slightly shrink the rectangles to get a nicer output.
            //r.x += cvRound(r.width*0.1);
            //r.width = cvRound(r.width*0.8);
            //r.y += cvRound(r.height*0.07);
            //r.height = cvRound(r.height*0.8);

            //检测出来的很多矩形框都超出了图像边界，将这些矩形框都强制规范在图像边界内部
            if(r.x < 0)
                r.x = 0;
            if(r.y < 0)
                r.y = 0;
            if(r.x + r.width > src.cols)
                r.width = src.cols - r.x;
            if(r.y + r.height > src.rows)
                r.height = src.rows - r.y;

            //从原图上截取矩形框大小的图片
            Mat imgROI = src(Rect(r.x, r.y, r.width, r.height));

            //将剪裁出来的图片缩放为64*128大小
            resize(imgROI,imgROI,Size(64,128));

            //生成hard example图片的文件名
            sprintf(saveName,"/Users/macbookpro/CLionProjects/pedestrian_detection/normalized_images/train/hard_neg4/hardexample%06d.jpg",HardExampleCount);

            //保存文件
            imwrite(saveName,imgROI);

//            //保存裁剪得到的图片名称到txt文件，换行分隔

                fout <<"hardexample" << setw(6)<<setfill('0') << HardExampleCount  << ".jpg"<< endl;

//            //num++;
            HardExampleCount++;

            //rectangle(src, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
        }
        //imshow("people detector", src);
        //waitKey(0);

    }
    fout.close();

    cout<<"HardExampleCount: "<<HardExampleCount - 1<<endl;

    return 0;
}
