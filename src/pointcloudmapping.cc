/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

// PointcloudMapping.cc
#include "pointcloudmapping.h"

#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "Converter.h"
#include <pcl/visualization/cloud_viewer.h>

namespace ORB_SLAM3 {

PointCloudMapping::PointCloudMapping(bool is_pc_reconstruction, double resolution)
{
    mResolution = resolution;
    mCx = 0;
    mCy = 0;
    mFx = 0;
    mFy = 0;
    mbShutdown = false;
    mbFinish = false;
    if(is_pc_reconstruction)
    {
        voxel.setLeafSize( resolution, resolution, resolution);
        statistical_filter.setMeanK(50);
        statistical_filter.setStddevMulThresh(1.0); // The distance threshold will be equal to: mean + stddev_mult * stddev

        mPointCloud = boost::make_shared<PointCloud>();  // 用boost::make_shared<>
        viewerThread = std::make_shared<std::thread>(&PointCloudMapping::showPointCloud, this);  // make_unique是c++14的
    }
    else mbFinish = true;

}

PointCloudMapping::~PointCloudMapping()
{
    viewerThread->join();
}

void PointCloudMapping::requestFinish()
{
    {
        unique_lock<mutex> locker(mKeyFrameMtx);
        mbShutdown = true;
    }
    mKeyFrameUpdatedCond.notify_one();
}

bool PointCloudMapping::isFinished()
{
    return mbFinish;
}

void PointCloudMapping::insertKeyFrame(KeyFrame* kf, const cv::Mat& color, const cv::Mat& depth)
{
    unique_lock<mutex> locker(mKeyFrameMtx);
    mvKeyFrames.push(kf);
    mvColorImgs.push( color.clone() );  // clone()函数进行Mat类型的深拷贝，为什幺深拷贝？？
    mvDepthImgs.push( depth.clone() );

    mKeyFrameUpdatedCond.notify_one();
    // cout << "receive a keyframe, id = " << kf->mnId << endl;
}

void PointCloudMapping::showPointCloud() 
{
    pcl::visualization::CloudViewer viewer("Dense pointcloud viewer");
    while(true) {   
        KeyFrame* kf;
        cv::Mat colorImg, depthImg;

        {
            std::unique_lock<std::mutex> locker(mKeyFrameMtx);
            while(mvKeyFrames.empty() && !mbShutdown){  // !mbShutdown为了防止所有关键帧映射点云完成后进入无限等待
                mKeyFrameUpdatedCond.wait(locker); 
            }            
            
            if (!(mvDepthImgs.size() == mvColorImgs.size() && mvKeyFrames.size() == mvColorImgs.size())) {
                std::cout << "这是不应该出现的情况！" << std::endl;
                continue;
            }

            if (mbShutdown && mvColorImgs.empty() && mvDepthImgs.empty() && mvKeyFrames.empty()) {
                break;
            }

            kf = mvKeyFrames.front();
            colorImg = mvColorImgs.front();    
            depthImg = mvDepthImgs.front();    
            mvKeyFrames.pop();
            mvColorImgs.pop();
            mvDepthImgs.pop();
        }

        if (mCx==0 || mCy==0 || mFx==0 || mFy==0) {
            mCx = kf->cx;
            mCy = kf->cy;
            mFx = kf->fx;
            mFy = kf->fy;
        }

        
        {
            std::unique_lock<std::mutex> locker(mPointCloudMtx);
            //cv::Mat cvpose(3,3,CV_32FC1,kf->GetPose().rotationMatrix().data());
            generatePointCloud(colorImg, depthImg, kf, kf->mnId);
            viewer.showCloud(mPointCloud);
        }
        
        // std::cout << "show point cloud, size=" << mPointCloud->points.size() << std::endl;
    }

    // 存储点云
    //string save_path = "11.pcd";
    if (!mPointCloud->points.empty())
    {
    pcl::io::savePCDFile("pointcloudmap.pcd", *mPointCloud);
    }
    cout << "save pcd files success!"<< endl;
    mbFinish = true;
}

void PointCloudMapping::generatePointCloud(const cv::Mat &imRGB, const cv::Mat &imD, KeyFrame *kf, int nId)
{ 
    // std::cout << "Converting image: " << nId;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();     
    PointCloud::Ptr current(new PointCloud);

    vector<float> meanDep; //获取平均深度
    for (int n = 0; n < kf -> mvDynamicArea.size(); n ++){
        float meanDep_ = 0;
        meanDep_ = get_meanDep(kf -> mvDynamicArea[n], imD, imRGB); //特征点个数和平均深度
        //meanDep.push_back(meanDep_);
        meanDep.push_back(meanDep_-0.2); //将均值拉到更接近人物的部分
        //cout << meanDep_<<' '<<kf -> KPinfo[n].second<<' ';
    }
    //cout << endl;

    for(size_t v = 0; v < imRGB.rows ; v+=3){
        for(size_t u = 0; u < imRGB.cols ; u+=3){
            cv::Point2i pt(u, v);
            bool IsDynamic = false;

            float d = imD.ptr<float>(v)[u];
            if(d <0.01 || d>10){ // 深度值为0 表示测量失败
                continue;
            }

            for (int n = 0; n < kf ->mvDynamicArea.size(); n ++){
                float depErr = d - kf -> KPinfo[n].second;
                //float depErr = abs(meanDep[n] - d);
                //cout << d <<' ' << kf -> KPinfo[n].second <<' ' <<depErr <<endl;
                if (kf ->mvDynamicArea[n].contains(pt) && depErr < 1) {
                //if (kf ->mvDynamicArea[n].contains(pt) ) {
                    IsDynamic = true;
                    break;
                }
            }

            if(IsDynamic == true){ // 动态对象内，不进行建图
              continue;
            }  

            PointT p;
            p.z = d;
            p.x = ( u - mCx) * p.z / mFx;
            p.y = ( v - mCy) * p.z / mFy;

            p.b = imRGB.ptr<uchar>(v)[u*3];
            p.g = imRGB.ptr<uchar>(v)[u*3+1];
            p.r = imRGB.ptr<uchar>(v)[u*3+2];

            current->points.push_back(p);

            
        }        
    }

    Eigen::Isometry3d T = Converter::toSE3Quat(kf->GetPose());
    PointCloud::Ptr tmp(new PointCloud);
    // tmp为转换到世界坐标系下的点云
    pcl::transformPointCloud(*current, *tmp, T.inverse().matrix()); 

    // depth filter and statistical removal，离群点剔除
    statistical_filter.setInputCloud(tmp);  
    statistical_filter.filter(*current);   
    (*mPointCloud) += *current;

    pcl::transformPointCloud(*mPointCloud, *tmp, T.inverse().matrix());
    // 加入新的点云后，对整个点云进行体素滤波
    voxel.setInputCloud(mPointCloud);
    voxel.filter(*tmp);
    mPointCloud->swap(*tmp);
    mPointCloud->is_dense = true; 

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double t = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count(); 
    // std::cout << ", Cost = " << t << std::endl;
}


void PointCloudMapping::getGlobalCloudMap(PointCloud::Ptr &outputMap)
{
    std::unique_lock<std::mutex> locker(mPointCloudMtx);
    outputMap = mPointCloud;
}

    float PointCloudMapping::get_meanDep(cv::Rect2i area, const cv::Mat &imDepth, const cv::Mat &imRGB) { //获得动态对象特征点个数和平均深度
        int num = 0;
        float sumDep = 0;

        for(size_t v = 0; v < imRGB.rows ; v+=3){
            for(size_t u = 0; u < imRGB.cols ; u+=3){
                float d = imDepth.ptr<float>(v)[u];
                if(d <0.01 || d>5){ // 深度值为0 表示测量失败
                    continue;
                }
                cv::Point2i pt(u, v);
                if (area.contains(pt)) {
                    num++;
                    sumDep = sumDep + d;
                    //cout << d << ' ';
                }
            }
        }

        float meanDep = sumDep/num;
        //cout << meanDep <<endl;
        return meanDep;
    }

}




