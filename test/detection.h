#ifndef PEDESTRIAN_DETECTION_DETECTION_H
#define PEDESTRIAN_DETECTION_DETECTION_H

//训练开关
#define TRAIN true//是否进行训练,true表示重新训练，false表示读取xml文件中的SVM模型
#define CENTRAL_CROP true //true:训练时，对96*160的INRIA正样本图片剪裁出中间的64*128大小人体
#define HARDNEG true //是否使用hardneg，true表示使用

//样本数量
#define PosSamNO 2416  //原始正样本数2416 , 3542
#define NegSamNO 12140 // 剪裁后的负样本数6070
#define cropNegNum 1214  //原始负样本数
#define HardExampleNO 4081 // hardneg的样本数量 //10896
#define AugPosSamNO 0 //Aug positive num

//正样本图片的文件名列表
#define PosSamListFile "/Users/macbookpro/CLionProjects/pedestrian_detection/img_dir/pos1.txt"
//负样本图片的文件名列表
#define NegSamListFile "/Users/macbookpro/CLionProjects/pedestrian_detection/img_dir/sample_new_neg.txt"
//hard负样本图片的文件列表
#define NegHardListFile  "/Users/macbookpro/CLionProjects/pedestrian_detection/img_dir/hard_neg4.txt"
//训练的HOG特征，svm分类时使用
#define SvmListFile "/Users/macbookpro/CLionProjects/pedestrian_detection/data/SVM_HOG8.xml"
//训练的HOG特征，resultMat结果
#define HogDetectorListFile "/Users/macbookpro/CLionProjects/pedestrian_detection/data/HOGDetectorForOpenCV7.txt"

//读入的pos图片路径
#define PosImageFile "/Users/macbookpro/CLionProjects/pedestrian_detection/normalized_images/train/pos1/"
//读入的neg图片路径
#define NegImageFile "/Users/macbookpro/CLionProjects/pedestrian_detection/normalized_images/train/new_neg/"
//读入的hardneg图片路径
#define HardNegImageFile "/Users/macbookpro/CLionProjects/pedestrian_detection/normalized_images/train/hard_neg4/"

#endif //PEDESTRIAN_DETECTION_DETECTION_H
