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


/*********************************    随机剪裁负样本   *******************************************/


int main()
{

    //检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
    HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);
    int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
    Ptr<SVM> svm = SVM::create();// 创建分类器

    if(TRAIN)//若TRAIN为true，重新训练分类器
    {
        string ImgName;//图片名(绝对路径)
        //正样本图片的文件名列表
        ifstream finPos(PosSamListFile);
        // ifstream finNeg("../sample_neg.txt");
        //负样本图片的文件名列表
        ifstream finNeg(NegSamListFile);
        //HardExample负样本的文件名列表
        ifstream finHardNeg(NegHardListFile);

        if(HARDNEG) {
            if (!finPos || !finNeg || !finHardNeg){
                cout << "Pos/Neg/hardNeg imglist reading failed..." << endl;
                return 1;
            }
        }else{
            if (!finPos || !finNeg)
            {
                cout << "Pos/Neg/hardNeg imglist reading failed..." << endl;
                return 1;
            }
        }

        Mat sampleFeatureMat;
        Mat sampleLabelMat;

        //loading original positive examples...
        for(int num=0; num < PosSamNO && getline(finPos,ImgName); num++)
        {
            cout <<"Now processing original positive image: " << ImgName << endl;
            ImgName = PosImageFile + ImgName;
            Mat src = imread(ImgName);//读取图片

            if(CENTRAL_CROP)//true:训练时，对96*160的INRIA正样本图片剪裁出中间的64*128大小人体
                if(src.cols >= 96 && src.rows >= 160)
                //resize(src,src,Size(64,128));
                    src = src(Rect(16,16,64,128));
               // else cout << "error" << endl; //测试

            vector<float> descriptors;//HOG描述子向量
            hog.compute(src, descriptors, Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
            //cout<<"描述子维数："<<descriptors.size()<<endl;


            //处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
            if(num == 0 )
            {
                DescriptorDim = descriptors.size();//HOG描述子的维数
                //初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
                sampleFeatureMat = Mat::zeros(PosSamNO +AugPosSamNO +NegSamNO +HardExampleNO, DescriptorDim, CV_32FC1);//CV_32FC1：CvMat数据结构参数
                //初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
                sampleLabelMat = Mat::zeros(PosSamNO +AugPosSamNO +NegSamNO +HardExampleNO, 1, CV_32SC1);//sampleLabelMat的数据类型必须为有符号整数型
            }

            //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
            for(int i=0; i<DescriptorDim; i++)
                sampleFeatureMat.at<float>(num,i) = descriptors[i];//第num个样本的特征向量中的第i个元素
            sampleLabelMat.at<int>(num,0) = 1;//正样本类别为1，有人
        }
        finPos.close();


        //依次读取负样本图片，生成HOG描述子
        for(int num = 0; num < NegSamNO && getline(finNeg,ImgName); num++)
        {
            cout<<"Now processing original negative image: "<<ImgName<<endl;
            // ImgName = "../normalized_images/train/neg/" + ImgName;
            //加上负样本的路径名
            ImgName = NegImageFile + ImgName;
            Mat src = imread(ImgName);//读取图片

            vector<float> descriptors;//HOG描述子向量
            hog.compute(src,descriptors,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)

            //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
            for(int i=0; i<DescriptorDim; i++)
                sampleFeatureMat.at<float>(num+PosSamNO+AugPosSamNO,i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
            sampleLabelMat.at<int>(num +PosSamNO +AugPosSamNO, 0) = -1;//负样本类别为-1，无人

        }
        finNeg.close();

        //依次读取HardExample负样本图片，生成HOG描述子
        if(HARDNEG){
            for(int num = 0; num < HardExampleNO && getline(finHardNeg,ImgName); num++)
        {
            cout<<"Now processing original hard negative image: "<<ImgName<<endl;
            // ImgName = "../normalized_images/train/neg/" + ImgName;
            //加上负样本的路径名
            ImgName = HardNegImageFile + ImgName;
            Mat src = imread(ImgName);//读取图片

            vector<float> descriptors;//HOG描述子向量
            hog.compute(src,descriptors,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
            //cout<<"描述子维数："<<descriptors.size()<<endl;

            //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
            for(int i=0; i<DescriptorDim; i++)
                sampleFeatureMat.at<float>(num+ PosSamNO + NegSamNO + AugPosSamNO,i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
            sampleLabelMat.at<int>(num + PosSamNO + NegSamNO + AugPosSamNO, 0) = -1;//负样本类别为-1，无人

        }
        }
        finHardNeg.close();

        //指定svm种类
        svm ->setType(SVM::C_SVC);
        //CvSVM::C_SVC : C类支持向量分类机。 n类分组  (n≥2)，允许用异常值惩罚因子C进行不完全分类。

        //CvSVM::NU_SVC : 类支持向量分类机。n类似然不完全分类的分类器。
        // 参数为取代C（其值在区间【0，1】中，nu越大，决策边界越平滑）

        //CvSVM::ONE_CLASS : 单分类器，所有的训练数据提取自同一个类里，
        // 然后SVM建立了一个分界线以分割该类在特征空间中所占区域和其它类在特征空间中所占区域。

        //CvSVM::EPS_SVR : 类支持向量回归机。
        // 训练集中的特征向量和拟合出来的超平面的距离需要小于p。异常值惩罚因子C被采用

        //CvSVM::NU_SVR : 类支持向量回归机。 代替了p。



        svm ->setC(0.01);//惩罚因子

        svm ->setGamma(1.0);//gamma参数

        //C是惩罚系数，即对误差的宽容度。
        // c越高，说明越不能容忍出现误差,容易过拟合。
        // C越小，容易欠拟合。C过大或过小，泛化能力变差

        svm ->setKernel(SVM::LINEAR);//设置核函数
        // LINEAR：线性核函数；

        // POLY:多项式核函数；
        /// -d用来设置多项式核函数的最高此项次数；默认值是3
        /// -r用来设置核函数中的coef0，也就是公式中的第二个r，默认值是0。
        // 一般选择1-11：1 3 5 7 9 11，也可以选择2,4，6…

        // RBF:径向机核函数【高斯核函数】；
        /// -g用来设置核函数中的gamma参数设置，默认值是1/k（k是类别数）
        //gamma是选择RBF函数作为kernel后，该函数自带的一个参数。
        // 隐含地决定了数据映射到新的特征空间后的分布，
        // gamma越大，支持向量越少，gamma值越小，支持向量越多。
        // 支持向量的个数影响训练与预测的速度。

        // SIGMOID:神经元的非线性作用函数核函数；
        /// -g用来设置核函数中的gamma参数设置，默认值是1/k（k是类别数）
        /// -r用来设置核函数中的coef0，也就是公式中的第二个r，默认值是0

        // PRECOMPUTED：用户自定义核函数

        //SVM的迭代训练过程的中止条件
        svm ->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 50000, FLT_EPSILON));

        cout<<"Starting training..."<<endl;
        //svm->train(trainingDataMat, cv::ml::SampleTypes::ROW_SAMPLE, labelsMat);
        svm ->train(sampleFeatureMat, ROW_SAMPLE, sampleLabelMat);

        // svm ->trainAuto(); //svm自动优化参数
        cout<<"Finishing training..."<<endl;

        //储存 SVM 分类器
        svm ->save(SvmListFile);

        //SVM的改进算法
        //J.Platt的SMO算法、
        //T.Joachims的SVM、
        //C.J.C.Burges等的PCGC、
        //张学工的CSVM
        //以及O.L.Mangasarian等的SOR算法

    }
    else {
        svm = SVM::load(SvmListFile);
    }
    cout << "loaded SVM_HOG.xml file"  << endl;

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

    cout << "going to write the HOGDetectorForOpenCV.txt file"  << endl;

    //保存HOG特征到HOGDetectorForOpenCV.txt
    ofstream fout(HogDetectorListFile);
    for (int i = 0; i < vec.size(); ++i)
    {
        fout << vec[i] << endl;
    }
    fout.close();//关闭文件


    /*********************************Testing**************************************************/
    HOGDescriptor hog_test;
    hog_test.setSVMDetector(vec);


    Mat src = imread("/Users/macbookpro/CLionProjects/pedestrian_detection/data/Test.jpg");
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
    //Mat testImg = imread("person014142.jpg");
    //Mat testImg = imread("noperson000026.jpg");
    //vector<float> descriptor;
    //hog.compute(testImg,descriptor,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
    //Mat testFeatureMat = Mat::zeros(1,3780,CV_32FC1);//测试样本的特征向量矩阵
    //将计算好的HOG描述子复制到testFeatureMat矩阵中
    //for(int i=0; i<descriptor.size(); i++)
    //	testFeatureMat.at<float>(0,i) = descriptor[i];

    //用训练好的SVM分类器对测试图片的特征向量进行分类
    //int result = svm.predict(testFeatureMat);//返回类标
    //cout<<"分类结果："<<result<<endl;


    return 0;
}


