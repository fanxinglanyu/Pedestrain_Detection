#include <iostream>
#include <iostream>
#include <fstream>
#include <stdlib.h> //srand()和rand()函数
#include <time.h> //time()函数
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <sys/time.h>

//原始负样本图片文件列表
#define INRIANegativeImageList "/Users/macbookpro/CLionProjects/pedestrian_detection/img_dir/sample_neg.txt"
//剪裁后的需要写入的负样本图片的文件名列表
#define NegSamListFile "/Users/macbookpro/CLionProjects/pedestrian_detection/img_dir/sample_new_neg.txt"


#define NegSamNO 12140 // 剪裁后的负样本数6070
#define cropNegNum 1214  //原始负样本数

using namespace std;
using namespace cv;

int CropImageCount = 0; //裁剪出来的负样本图片个数
long long  sumCount = 0;
int needCropNum = 10;

int main()
{
    Mat src;
    string imgName;

    char saveName[256];//裁剪出来的负样本图片文件名
    ifstream fin(INRIANegativeImageList);//打开原始负样本图片文件列表
    ofstream fout(NegSamListFile,ios::trunc);//加路径
    int num = 0;
    //ifstream fin("subset.txt");


    for (int i = 0;i < cropNegNum && getline(fin, imgName); i++)
    {
        imgName = "/Users/macbookpro/CLionProjects/pedestrian_detection/normalized_images/train/neg/" + imgName;
        //IMREAD_UNCHANGED ：不进行转化，比如保存为了16位的图片，读取出来仍然为16位。
        Mat img = imread(imgName, IMREAD_UNCHANGED);
        //Linux时间函数
        //tv_sec;        /* Seconds. */
        //tv_usec;  /* Microseconds. */
        struct timeval tv;
        if (img.empty())//如果图片不存在，则输出
        {
            cout << "can not load the image:" << imgName << endl;
            continue;
        }
        if (img.cols >= 64 && img.rows >= 128)//如果图片尺寸大于64或者128时
        {
            num = 0;
            //从每张图片中随机剪裁20张64*128的负样本
            for (int j = 0;j < needCropNum;j++)
            {

                //gettimeofday()会把目前的时间用tv 结构体返回
                gettimeofday(&tv,NULL);
                srand(tv.tv_usec);//利用系统时间（微妙），设置随机数种子

                int x = rand() % (img.cols - 64); //左上角x， 范围为[0,cols - 64)
                int y = rand() % (img.rows - 128); //左上角y， 范围为[0,rows - 64)
                cout << "x:" << x << "y:" << y <<endl;
                Mat src = img(Rect(x, y, 64, 128));//Rect(x,y,64,128)从左上角坐标为(x,y)位置剪裁一个宽64，高128的矩形
                //把剪裁后的图片名称存入svaeName变量中
                sprintf(saveName, "/Users/macbookpro/CLionProjects/pedestrian_detection/normalized_images/train/new_neg/neg%dCropped%d.png",i, num);
                //把剪裁后的图片src，另存为名字为svaeName的图片
                imwrite(saveName,src);

                //保存裁剪得到的图片名称到txt文件，换行分隔
                if(i<(cropNegNum-1)){
                    fout <<"neg" << i << "Cropped"<< num++ << ".png"<< endl;
                }
                else if(i==(cropNegNum-1) && j< needCropNum - 1){
                    fout <<"neg" << i << "Cropped"<< num++ <<  ".png"<< endl;
                }
                else{
                    fout <<"neg" << i << "Cropped"<< num++ << ".png";
                }
                sumCount++;
            }
        }
    }
    fout.close();//关闭文件

    cout<<"总共裁剪出"<< sumCount <<"张图片"<<endl;

}