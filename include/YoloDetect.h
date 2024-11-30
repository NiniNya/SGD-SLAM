//
// yolov5部分
//
#ifndef YOLO_DETECT_H
#define YOLO_DETECT_H

// OpenCV相关引用
#include <opencv2/opencv.hpp>
// torch相关引用
#include <torch/script.h>

#include <algorithm>
#include <iostream>
#include <utility>
#include <time.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
using namespace std;

class YoloDetection
{
public:
    YoloDetection();
    ~YoloDetection();
    void GetImage(cv::Mat& RGB);
    void ClearImage();
    bool Detect();
    void ClearArea();
    vector<cv::Rect2i> mvPersonArea = {};
    vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float score_thresh=0.5, float iou_thresh=0.5);

public:
    cv::Mat mRGB;
    torch::jit::script::Module mModule;
    std::vector<std::string> mClassnames;
    
    vector<string> mvDynamicNames;
    vector<string> mvStaticNames;
    vector<string> mvMovableNames;
    vector<cv::Rect2i> mvDynamicArea;
    vector<cv::Rect2i> mvStaticArea;
    vector<cv::Rect2i> mvMovableArea;
    // cv::Mat mask;  
    map<string, vector<cv::Rect2i>> mmDetectMap;

};


#endif //YOLO_DETECT_H
