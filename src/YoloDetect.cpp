//
// yolov5目标检测调用部分
//
#include <YoloDetect.h>

YoloDetection::YoloDetection()
{
    torch::jit::setTensorExprFuserEnabled(false);

    // 加载模型
    mModule = torch::jit::load("yolov5s.torchscript.pt");

    //加载类别
    std::ifstream f("coco.names");
    std::string name = "";
    while (std::getline(f, name))
    {
        mClassnames.push_back(name);
    }
    //mvDynamicNames = {"person","bird", "cat", "dog", "horse", "sheep", "crow", "bear","bicycle", "motorbike", "bus", "train", "truck", "boat", "backpack", "umbrella", "remote", 
                        //"bottle", "wine glass", "cup", "chair", "laptop", "mouse","keyboard","book","cell phone"};
    mvDynamicNames = {"person","bird", "cat", "dog", "horse", "sheep", "crow", "bear"};
    mvStaticNames = {"traffic light", "fire hydrant", "stop sign", "sofa", "parking meter", "bench", "couch", "dining table", "toilet",
                      "microwave", "oven", "toaster", "sink", "refrigerator", "tvmonitor", "keyboard"};
    mvMovableNames = {"bicycle", "motorbike", "bus", "train", "truck", "boat", "backpack", "umbrella", "remote", 
                         "bottle", "wine glass", "cup", "chair", "mouse","book","laptop","cell phone", "frisbee"};
    //mvMovableNames = { };
}

YoloDetection::~YoloDetection()
{

}

bool YoloDetection::Detect()
{

    cv::Mat img;

    if(mRGB.empty())
    {
        std::cout << "Read RGB failed!" << std::endl;
        return false;
    }

    // Preparing input tensor 前处理
    // 缩放至指定大小
    cv::resize(mRGB, img, cv::Size(640, 380));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

     // 将OpenCV的Mat类型构造成Tensor
    torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols,3},torch::kByte);
    imgTensor = imgTensor.permute({2,0,1});
    imgTensor = imgTensor.toType(torch::kFloat);
    imgTensor = imgTensor.div(255);
    imgTensor = imgTensor.unsqueeze(0);

    // yolov5模型识别
    torch::Tensor preds = mModule.forward({imgTensor}).toTuple()->elements()[0].toTensor();
    //后处理
    std::vector<torch::Tensor> dets = YoloDetection::non_max_suppression(preds, 0.4, 0.5);
    if (dets.size() > 0)
    {
        // Visualize result
        //可视化结果（方框）
        for (size_t i=0; i < dets[0].sizes()[0]; ++ i)
        {
            float left = dets[0][i][0].item().toFloat() * mRGB.cols / 640;
            float top = dets[0][i][1].item().toFloat() * mRGB.rows / 384;
            float right = dets[0][i][2].item().toFloat() * mRGB.cols / 640;
            float bottom = dets[0][i][3].item().toFloat() * mRGB.rows / 384;
            int classID = dets[0][i][5].item().toInt(); //识别出的对象类别id


            cv::Rect2i DetectArea(left, top, right - left, bottom - top);
            // cv::Rect2i DetectArea(left-10, top-10, (right - left)+20, (bottom - top)+20);
            mmDetectMap[mClassnames[classID]].push_back(DetectArea);
            // cv::Mat mask1 = cv::Mat::zeros(480,640,CV_8U);   
            // cv::Mat mask = cv::Mat::zeros(480,640,CV_8U);  
            // cv::Mat bgModel,fgModel;
            // cv::Mat foreground(mRGB.size(), CV_8UC3, cv::Scalar(255, 255, 255));

            if (count(mvDynamicNames.begin(), mvDynamicNames.end(), mClassnames[classID])) //识别出的对象是否是动态对象
            {
                // cv::Rect2i DynamicArea(left-10, top-10, (right - left)+20, (bottom - top)+20); 
                cv::Rect2i DynamicArea(left, top, right - left, bottom - top); 
                mvDynamicArea.push_back(DynamicArea); // 是的话，将该区域放入动态区域
                // cv::grabCut(mRGB, mask1, DynamicArea, bgModel, fgModel, 5, cv::GC_INIT_WITH_RECT);  //使用grabCut进行背景分离
                // cv::compare(mask1, cv::GC_PR_FGD, mask1, cv::CMP_EQ);    //根据结果合成掩码
                //cv::compare(mask1, cv::GC_FGD, mask1, cv::CMP_EQ);    //根据结果合成掩码
                // cv::bitwise_or(mask, mask1, mask);   //将掩码叠加
                // mRGB.copyTo(foreground, mask1); // 复制前景图像
                // cv::imshow("Foreground",foreground);
	        
            }

            if (count(mvStaticNames.begin(), mvStaticNames.end(), mClassnames[classID])) //识别出的对象是否是静态对象
            {
                cv::Rect2i StaticArea(left, top, right - left, bottom - top); 
                // cv::Rect2i StaticArea(left-10, top-10, (right - left)+20, (bottom - top)+20); 
                mvStaticArea.push_back(StaticArea);    
            }

            if (count(mvMovableNames.begin(), mvMovableNames.end(), mClassnames[classID])) //识别出的对象是否是可动对象
            {
                cv::Rect2i MovableArea(left, top, right - left, bottom - top); 
                // cv::Rect2i MovableArea(left-10, top-10, (right - left)+20, (bottom - top)+20); 
                mvMovableArea.push_back(MovableArea);    
            }

                    //定义：std::vector<cv::Rect2i> mvDynamicArea;
                    //typedef struct CvRect 
                    // 　　{ 
                    // 　　int x; /* 方形的左上角的x-坐标 */ 
                    // 　　int y; /* 方形的左上角的y-坐标*/ 
                    // 　　int width; /* 宽 */ 
                    // 　　int height; /* 高 */ 
                    // 　　}

        }
        if (mvDynamicArea.size() == 0)
        {
            cv::Rect2i tDynamicArea(1, 1, 1, 1);
            mvDynamicArea.push_back(tDynamicArea);
        }
    }
    return true;
}


// 后处理
vector<torch::Tensor> YoloDetection::non_max_suppression(torch::Tensor preds, float score_thresh, float iou_thresh)
{
    std::vector<torch::Tensor> output;
    for (size_t i=0; i < preds.sizes()[0]; ++i)
    {
        torch::Tensor pred = preds.select(0, i);

        // Filter by scores
        torch::Tensor scores = pred.select(1, 4) * std::get<0>( torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));
        pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
        if (pred.sizes()[0] == 0) continue;

        // (center_x, center_y, w, h) to (left, top, right, bottom)
        pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
        pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
        pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
        pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

        // Computing scores and classes
        std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
        pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
        pred.select(1, 5) = std::get<1>(max_tuple);

        torch::Tensor  dets = pred.slice(1, 0, 6);

        torch::Tensor keep = torch::empty({dets.sizes()[0]});
        torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
        std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
        torch::Tensor v = std::get<0>(indexes_tuple);
        torch::Tensor indexes = std::get<1>(indexes_tuple);
        int count = 0;
        while (indexes.sizes()[0] > 0)
        {
            keep[count] = (indexes[0].item().toInt());
            count += 1;

            // Computing overlaps
            torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1);
            for (size_t i=0; i<indexes.sizes()[0] - 1; ++i)
            {
                lefts[i] = std::max(dets[indexes[0]][0].item().toFloat(), dets[indexes[i + 1]][0].item().toFloat());
                tops[i] = std::max(dets[indexes[0]][1].item().toFloat(), dets[indexes[i + 1]][1].item().toFloat());
                rights[i] = std::min(dets[indexes[0]][2].item().toFloat(), dets[indexes[i + 1]][2].item().toFloat());
                bottoms[i] = std::min(dets[indexes[0]][3].item().toFloat(), dets[indexes[i + 1]][3].item().toFloat());
                widths[i] = std::max(float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
                heights[i] = std::max(float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());
            }
            torch::Tensor overlaps = widths * heights;


            torch::Tensor ious = overlaps / (areas.select(0, indexes[0].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);
            indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
        }
        keep = keep.toType(torch::kInt64);
        output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));
    }
    return output;
}

void YoloDetection::GetImage(cv::Mat &RGB) //获取图像
{
    mRGB = RGB;
}

void YoloDetection::ClearImage()
{
    mRGB = 0;
}

void YoloDetection::ClearArea()
{
    mvPersonArea.clear();
}

