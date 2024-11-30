/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "Frame.h"

#include "G2oTypes.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "ORBextractor.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include "GeometricCamera.h"
#include <cmath>

#include <random>

#include <thread>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>

namespace ORB_SLAM3
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

cv::Mat imGrayPre;
std::vector<cv::Point2f> prepoint, nextpoint;
std::vector<cv::Point2f> F_prepoint, F_nextpoint;
std::vector<cv::Point2f> F2_prepoint, F2_nextpoint;
std::vector<uchar> state;
std::vector<float> err;
std::vector<std::vector<cv::KeyPoint>> mvKeysPre;

//For stereo fisheye matching
cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);

Frame::Frame(): mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(NULL), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mbHasPose(false), mbHasVelocity(false)
{
#ifdef REGISTER_TIMES
    mTimeStereoMatch = 0;
    mTimeORB_Ext = 0;
#endif
}


//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpcpi(frame.mpcpi),mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mK_(Converter::toMatrix3f(frame.mK)), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mImuCalib(frame.mImuCalib), mnCloseMPs(frame.mnCloseMPs),
     mpImuPreintegrated(frame.mpImuPreintegrated), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame), mImuBias(frame.mImuBias),
     mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mNameFile(frame.mNameFile), mnDataset(frame.mnDataset),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), mpPrevFrame(frame.mpPrevFrame), mpLastKeyFrame(frame.mpLastKeyFrame),
     mbIsSet(frame.mbIsSet), mbImuPreintegrated(frame.mbImuPreintegrated), mpMutexImu(frame.mpMutexImu),
     mpCamera(frame.mpCamera), mpCamera2(frame.mpCamera2), Nleft(frame.Nleft), Nright(frame.Nright),
     monoLeft(frame.monoLeft), monoRight(frame.monoRight), mvLeftToRightMatch(frame.mvLeftToRightMatch),
     mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints),
     mTlr(frame.mTlr), mRlr(frame.mRlr), mtlr(frame.mtlr), mTrl(frame.mTrl),
     mTcw(frame.mTcw), mbHasPose(false), mbHasVelocity(false)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++){
            mGrid[i][j]=frame.mGrid[i][j];
            if(frame.Nleft > 0){
                mGridRight[i][j] = frame.mGridRight[i][j];
            }
        }

    if(frame.mbHasPose)
        SetPose(frame.GetPose());

    if(frame.HasVelocity())
    {
        SetVelocity(frame.GetVelocity());
    }

    mmProjectPoints = frame.mmProjectPoints;
    mmMatchedInImage = frame.mmMatchedInImage;

#ifdef REGISTER_TIMES
    mTimeStereoMatch = frame.mTimeStereoMatch;
    mTimeORB_Ext = frame.mTimeORB_Ext;
#endif
}


Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera) ,mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,0,0);
    thread threadRight(&Frame::ExtractORB,this,1,imRight,0,0);
    threadLeft.join();
    threadRight.join();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    ComputeStereoMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);
    mmProjectPoints.clear();
    mmMatchedInImage.clear();


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);



        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    }
    else
    {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera,Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera),mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();
    mvDynamicArea = mpORBextractorLeft->mvDynamicArea;
    mvStaticArea = mpORBextractorLeft->mvStaticArea;
    mvMovableArea = mpORBextractorLeft->mvMovableArea;

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractORB(0,imGray,0,0);

    // Calculate the dynamic abnormal points and output the T matrix
    // **使用几何法计算动态点**
    cv::Mat  imGrayT = imGray;
    if(imGrayPre.data)
    {
        //（几何法）使用光流进行运动一致性检测，找到异常点并保存

        //std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

        ProcessMovingObject(imGray);

        // std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
        // double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
        // cout << "segment time  =" << ttrack*1000 << endl;

        std::swap(imGrayPre, imGrayT);
    }
    else
    {
        std::swap(imGrayPre, imGrayT);
    }
    // **使用几何法计算动态点**

    //  **根据判断结果移除动态点**
    // for (int k=0; k<mvKeys.size(); ++k){
    //     if (IsInDynamic(k) == true)
    //     //if ((IsInDynamic(k) == true && IsInStatic(k) == false && IsNotMoving(k) == false) )
    //     //if (IsInDynamic(k) == true && IsInStatic(k) == false)
    //    // if (IsInDynamic(k) == true )
    //         { //在动态物体框内,但不在静态物体框内
    //             vbInDynamic_mvKeys.push_back(true);
    //             mvKeys[k] = cv::KeyPoint(-1,-1,-1);
    //         }
    //         else{
    //             vbInDynamic_mvKeys.push_back(false);
    //         }
    // }
    //  **根据判断结果移除动态点**

    //  **动态区域信息获取**
    TMnum.clear();
    KPinfo.clear();
    for (int n = 0; n < mvDynamicArea.size(); n ++){
        int myTM = get_TMnum(mvDynamicArea[n]); //异常点个数
        std::pair<int, float> myKP = get_KPnum(mvDynamicArea[n], imDepth); //特征点个数和平均深度

        TMnum.push_back(myTM);
        KPinfo.push_back(myKP);
    }

    //  **可动区域信息获取**
    TMnum_.clear();
    KPinfo_.clear();
    for (int n = 0; n < mvMovableArea.size(); n ++){
        int myTM = get_TMnum(mvMovableArea[n]); //异常点个数
        std::pair<int, float> myKP = get_KPnum(mvMovableArea[n], imDepth); //特征点个数和平均深度

        TMnum_.push_back(myTM);
        KPinfo_.push_back(myKP);
    }

    //  **更新移动概率**
    movingPro.clear();
    for (int k=0; k<mvKeys.size(); k++){ 

        float movingPro_ = 0;

        //movingPro_ = movingPro_ + do_InDynamic(k, imDepth) +  do_InMoving(k, imDepth) + do_NotMoving(k) +do_InStatic(k);  //分级
        movingPro_ = movingPro_ + do_InDynamic(k, imDepth, TMnum, KPinfo)+ do_InMoving(k, imDepth, TMnum_, KPinfo_);  //深度分割
        //movingPro_ = movingPro_ + do_InDynamic(k, imDepth, TMnum, KPinfo);  //深度分割
        //cout << "before" << movingPro_<< endl;

        if (DynamicFlag == true) {
            movingPro_=  movingPro_ + do_InTM(k);
            //cout << "after" << movingPro_<< endl;
        }

        if (movingPro_ < 0) movingPro_ = 0;
        if (movingPro_ > 1) movingPro_ = 1;
        if (movingPro_ > 0.9)  mvKeys[k] = cv::KeyPoint(-1,-1,-1); //去除高于阈值的点
        movingPro.push_back(movingPro_);

    }
    //  **更新移动概率**

    //mpORBextractorLeft->CheckMovingKeyPoints(imGray,mvKeys, mDescriptors, T_M);

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    if(pPrevF){
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    }
    else{
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();
}


Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, GeometricCamera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(static_cast<Pinhole*>(pCamera)->toK()), mK_(static_cast<Pinhole*>(pCamera)->toK_()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL),mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mpCamera(pCamera),
     mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractORB(0,imGray,0,1000);
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);
    mnCloseMPs = 0;

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,0);
        fy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,1);
        cx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,2);
        cy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }


    mb = mbf/fx;

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
        {
            SetVelocity(pPrevF->GetVelocity());
        }
    }
    else
    {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();
}


void Frame::AssignFeaturesToGrid()
{
    // Fill matrix with points
    const int nCells = FRAME_GRID_COLS*FRAME_GRID_ROWS;

    int nReserve = 0.5f*N/(nCells);

    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++){
            mGrid[i][j].reserve(nReserve);
            if(Nleft != -1){
                mGridRight[i][j].reserve(nReserve);
            }
        }



    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i]
                                                 : (i < Nleft) ? mvKeys[i]
                                                                 : mvKeysRight[i - Nleft];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY)){
            if(Nleft == -1 || i < Nleft)
                mGrid[nGridPosX][nGridPosY].push_back(i);
            else
                mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
        }
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1)
{
    vector<int> vLapping = {x0,x1};
    if(flag==0)
        monoLeft = (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors,vLapping);
    else
        monoRight = (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight,vLapping);
}

bool Frame::isSet() const {
    return mbIsSet;
}

void Frame::SetPose(const Sophus::SE3<float> &Tcw) {
    mTcw = Tcw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

void Frame::SetNewBias(const IMU::Bias &b)
{
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

void Frame::SetVelocity(Eigen::Vector3f Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;
}

Eigen::Vector3f Frame::GetVelocity() const
{
    return mVw;
}

void Frame::SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;

    Sophus::SE3f Twb(Rwb, twb);
    Sophus::SE3f Tbw = Twb.inverse();

    mTcw = mImuCalib.mTcb * Tbw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

void Frame::UpdatePoseMatrices()
{
    Sophus::SE3<float> Twc = mTcw.inverse();
    mRwc = Twc.rotationMatrix();
    mOw = Twc.translation();
    mRcw = mTcw.rotationMatrix();
    mtcw = mTcw.translation();
}

Eigen::Matrix<float,3,1> Frame::GetImuPosition() const {
    return mRwc * mImuCalib.mTcb.translation() + mOw;
}

Eigen::Matrix<float,3,3> Frame::GetImuRotation() {
    return mRwc * mImuCalib.mTcb.rotationMatrix();
}

Sophus::SE3<float> Frame::GetImuPose() {
    return mTcw.inverse() * mImuCalib.mTcb;
}

Sophus::SE3f Frame::GetRelativePoseTrl()
{
    return mTrl;
}

Sophus::SE3f Frame::GetRelativePoseTlr()
{
    return mTlr;
}

Eigen::Matrix3f Frame::GetRelativePoseTlr_rotation(){
    return mTlr.rotationMatrix();
}

Eigen::Vector3f Frame::GetRelativePoseTlr_translation() {
    return mTlr.translation();
}


bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    if(Nleft == -1){
        pMP->mbTrackInView = false;
        pMP->mTrackProjX = -1;
        pMP->mTrackProjY = -1;

        // 3D in absolute coordinates
        Eigen::Matrix<float,3,1> P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const Eigen::Matrix<float,3,1> Pc = mRcw * P + mtcw;
        const float Pc_dist = Pc.norm();

        // Check positive depth
        const float &PcZ = Pc(2);
        const float invz = 1.0f/PcZ;
        if(PcZ<0.0f)
            return false;

        const Eigen::Vector2f uv = mpCamera->project(Pc);

        if(uv(0)<mnMinX || uv(0)>mnMaxX)
            return false;
        if(uv(1)<mnMinY || uv(1)>mnMaxY)
            return false;

        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const Eigen::Vector3f PO = P - mOw;
        const float dist = PO.norm();

        if(dist<minDistance || dist>maxDistance)
            return false;

        // Check viewing angle
        Eigen::Vector3f Pn = pMP->GetNormal();

        const float viewCos = PO.dot(Pn)/dist;

        if(viewCos<viewingCosLimit)
            return false;

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist,this);

        // Data used by the tracking
        pMP->mbTrackInView = true;
        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjXR = uv(0) - mbf*invz;

        pMP->mTrackDepth = Pc_dist;

        pMP->mTrackProjY = uv(1);
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;

        return true;
    }
    else{
        pMP->mbTrackInView = false;
        pMP->mbTrackInViewR = false;
        pMP -> mnTrackScaleLevel = -1;
        pMP -> mnTrackScaleLevelR = -1;

        pMP->mbTrackInView = isInFrustumChecks(pMP,viewingCosLimit);
        pMP->mbTrackInViewR = isInFrustumChecks(pMP,viewingCosLimit,true);

        return pMP->mbTrackInView || pMP->mbTrackInViewR;
    }
}

bool Frame::ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const Eigen::Vector3f Pc = mRcw * P + mtcw;
    const float &PcX = Pc(0);
    const float &PcY= Pc(1);
    const float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    u=fx*PcX*invz+cx;
    v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    float u_distort, v_distort;

    float x = (u - cx) * invfx;
    float y = (v - cy) * invfy;
    float r2 = x * x + y * y;
    float k1 = mDistCoef.at<float>(0);
    float k2 = mDistCoef.at<float>(1);
    float p1 = mDistCoef.at<float>(2);
    float p2 = mDistCoef.at<float>(3);
    float k3 = 0;
    if(mDistCoef.total() == 5)
    {
        k3 = mDistCoef.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    u_distort = x_distort * fx + cx;
    v_distort = y_distort * fy + cy;


    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

Eigen::Vector3f Frame::inRefCoordinates(Eigen::Vector3f pCw)
{
    return mRcw * pCw + mtcw;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel, const bool bRight) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    const int nMinCellX = max(0,(int)floor((x-mnMinX-factorX)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
    {
        return vIndices;
    }

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+factorX)*mfGridElementWidthInv));
    if(nMaxCellX<0)
    {
        return vIndices;
    }

    const int nMinCellY = max(0,(int)floor((y-mnMinY-factorY)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
    {
        return vIndices;
    }

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+factorY)*mfGridElementHeightInv));
    if(nMaxCellY<0)
    {
        return vIndices;
    }

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = (Nleft == -1) ? mvKeysUn[vCell[j]]
                                                         : (!bRight) ? mvKeys[vCell[j]]
                                                                     : mvKeysRight[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<factorX && fabs(disty)<factorY)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);

    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat, static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);


    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }

}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // Undistort corners
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));
    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

bool Frame::UnprojectStereo(const int &i, Eigen::Vector3f &x3D)
{
    const float z = mvDepth[i];
    if(z>0) {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        Eigen::Vector3f x3Dc(x, y, z);
        x3D = mRwc * x3Dc + mOw;
        return true;
    } else
        return false;
}

bool Frame::imuIsPreintegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    return mbImuPreintegrated;
}

void Frame::setIntegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    mbImuPreintegrated = true;
}

Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, GeometricCamera* pCamera2, Sophus::SE3f& Tlr,Frame* pPrevF, const IMU::Calib &ImuCalib)
        :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),  mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
         mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(pCamera2),
         mbHasPose(false), mbHasVelocity(false)

{
    imgLeft = imLeft.clone();
    imgRight = imRight.clone();

    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1]);
    thread threadRight(&Frame::ExtractORB,this,1,imRight,static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1]);
    threadLeft.join();
    threadRight.join();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    Nleft = mvKeys.size();
    Nright = mvKeysRight.size();
    N = Nleft + Nright;

    if(N == 0)
        return;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf / fx;

    // Sophus/Eigen
    mTlr = Tlr;
    mTrl = mTlr.inverse();
    mRlr = mTlr.rotationMatrix();
    mtlr = mTlr.translation();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    ComputeStereoFishEyeMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    //Put all descriptors in the same matrix
    cv::vconcat(mDescriptors,mDescriptorsRight,mDescriptors);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(nullptr));
    mvbOutlier = vector<bool>(N,false);

    AssignFeaturesToGrid();

    mpMutexImu = new std::mutex();

    UndistortKeyPoints();

}

void Frame::ComputeStereoFishEyeMatches() {
    //Speed it up by matching keypoints in the lapping area
    vector<cv::KeyPoint> stereoLeft(mvKeys.begin() + monoLeft, mvKeys.end());
    vector<cv::KeyPoint> stereoRight(mvKeysRight.begin() + monoRight, mvKeysRight.end());

    cv::Mat stereoDescLeft = mDescriptors.rowRange(monoLeft, mDescriptors.rows);
    cv::Mat stereoDescRight = mDescriptorsRight.rowRange(monoRight, mDescriptorsRight.rows);

    mvLeftToRightMatch = vector<int>(Nleft,-1);
    mvRightToLeftMatch = vector<int>(Nright,-1);
    mvDepth = vector<float>(Nleft,-1.0f);
    mvuRight = vector<float>(Nleft,-1);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(Nleft);
    mnCloseMPs = 0;

    //Perform a brute force between Keypoint in the left and right image
    vector<vector<cv::DMatch>> matches;

    BFmatcher.knnMatch(stereoDescLeft,stereoDescRight,matches,2);

    int nMatches = 0;
    int descMatches = 0;

    //Check matches using Lowe's ratio
    for(vector<vector<cv::DMatch>>::iterator it = matches.begin(); it != matches.end(); ++it){
        if((*it).size() >= 2 && (*it)[0].distance < (*it)[1].distance * 0.7){
            //For every good match, check parallax and reprojection error to discard spurious matches
            Eigen::Vector3f p3D;
            descMatches++;
            float sigma1 = mvLevelSigma2[mvKeys[(*it)[0].queryIdx + monoLeft].octave], sigma2 = mvLevelSigma2[mvKeysRight[(*it)[0].trainIdx + monoRight].octave];
            float depth = static_cast<KannalaBrandt8*>(mpCamera)->TriangulateMatches(mpCamera2,mvKeys[(*it)[0].queryIdx + monoLeft],mvKeysRight[(*it)[0].trainIdx + monoRight],mRlr,mtlr,sigma1,sigma2,p3D);
            if(depth > 0.0001f){
                mvLeftToRightMatch[(*it)[0].queryIdx + monoLeft] = (*it)[0].trainIdx + monoRight;
                mvRightToLeftMatch[(*it)[0].trainIdx + monoRight] = (*it)[0].queryIdx + monoLeft;
                mvStereo3Dpoints[(*it)[0].queryIdx + monoLeft] = p3D;
                mvDepth[(*it)[0].queryIdx + monoLeft] = depth;
                nMatches++;
            }
        }
    }
}

bool Frame::isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, bool bRight) {
    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    Eigen::Matrix3f mR;
    Eigen::Vector3f mt, twc;
    if(bRight){
        Eigen::Matrix3f Rrl = mTrl.rotationMatrix();
        Eigen::Vector3f trl = mTrl.translation();
        mR = Rrl * mRcw;
        mt = Rrl * mtcw + trl;
        twc = mRwc * mTlr.translation() + mOw;
    }
    else{
        mR = mRcw;
        mt = mtcw;
        twc = mOw;
    }

    // 3D in camera coordinates
    Eigen::Vector3f Pc = mR * P + mt;
    const float Pc_dist = Pc.norm();
    const float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    Eigen::Vector2f uv;
    if(bRight) uv = mpCamera2->project(Pc);
    else uv = mpCamera->project(Pc);

    if(uv(0)<mnMinX || uv(0)>mnMaxX)
        return false;
    if(uv(1)<mnMinY || uv(1)>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const Eigen::Vector3f PO = P - twc;
    const float dist = PO.norm();

    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    Eigen::Vector3f Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn) / dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    if(bRight){
        pMP->mTrackProjXR = uv(0);
        pMP->mTrackProjYR = uv(1);
        pMP->mnTrackScaleLevelR= nPredictedLevel;
        pMP->mTrackViewCosR = viewCos;
        pMP->mTrackDepthR = Pc_dist;
    }
    else{
        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;
        pMP->mTrackDepth = Pc_dist;
    }

    return true;
}

Eigen::Vector3f Frame::UnprojectStereoFishEye(const int &i){
    return mRwc * mvStereo3Dpoints[i] + mOw;
}

// Epipolar constraints and output the T matrix.
//此函数作用是用几何方法进行运动一致性检测：
// step 1 :计算角点(像素级->亚像素级)
// step 2 :计算光流金字塔(确定角点1,2的匹配关系)
// step 3 :对于光流法得到的角点进行筛选(像素块内像素差的和小于阈值)
// step 4 :计算F矩阵(再对点进行了一次筛选)
// step 5 :根据角点到级线的距离小于0.1筛选最匹配的角点
// step 6:找到需要被删去的异常点
void Frame::ProcessMovingObject(const cv::Mat &imgray)
{
    // Clear the previous data
	F_prepoint.clear();
	F_nextpoint.clear();
	F2_prepoint.clear();
	F2_nextpoint.clear();
	T_M.clear();

	// Detect dynamic target and ultimately optput the T matrix
	//step 1 调用opencv 函数 计算Harris 角点，将结果保存在 prepoint 矩阵当中
    //cv::goodFeaturesToTrack()提取到的角点只能达到像素级别
    //我们则需要使用cv::cornerSubPix()对检测到的角点作进一步的优化计算，可使角点的精度达到亚像素级别。
    // 调用opencv的函数,进行亚像素的角点检测，输出的角点还是放在 prepoint 里面
    cv::goodFeaturesToTrack(imGrayPre, prepoint, 1000, 0.01, 8, cv::Mat(), 3, true, 0.04);
    cv::cornerSubPix(imGrayPre, prepoint, cv::Size(10, 10), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
	
    //step 2 Lucas-Kanade方法计算稀疏特征集的光流。计算光流金字塔，光流金字塔是光流法的一种常见的处理方式，能够避免位移较大时丢失追踪的情况，
    cv::calcOpticalFlowPyrLK(imGrayPre, imgray, prepoint, nextpoint, state, err, cv::Size(22, 22), 5, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.01));

    //step 3 对于光流法得到的 角点进行筛选。筛选的结果放入 F_prepoint F_nextpoint 两个数组当中。光流角点是否跟踪成功保存在status数组当中
	for (int i = 0; i < state.size(); i++)
    {
        if(state[i] != 0)   // 光流跟踪成功的点
        {
            int dx[10] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
            int dy[10] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
            int x1 = prepoint[i].x, y1 = prepoint[i].y;
            int x2 = nextpoint[i].x, y2 = nextpoint[i].y;
            
            // 认为超过规定区域的,太靠近边缘。 跟踪的光流点的status 设置为0 ,一会儿会丢弃这些点
            if ((x1 < limit_edge_corner || x1 >= imgray.cols - limit_edge_corner || x2 < limit_edge_corner || x2 >= imgray.cols - limit_edge_corner
            || y1 < limit_edge_corner || y1 >= imgray.rows - limit_edge_corner || y2 < limit_edge_corner || y2 >= imgray.rows - limit_edge_corner))
            {
                state[i] = 0;
                continue;
            }
            
            // 对于光流跟踪的结果进行验证，匹配对中心3*3的图像块的像素差（sum）太大，那么也舍弃这个匹配点
            double sum_check = 0;
            for (int j = 0; j < 9; j++)
                sum_check += abs(imGrayPre.at<uchar>(y1 + dy[j], x1 + dx[j]) - imgray.at<uchar>(y2 + dy[j], x2 + dx[j]));
            if (sum_check > limit_of_check) state[i] = 0;
            

            //bool flag = 0;
            // 好的光流点存入 F_prepoint F_nextpoint 两个数组当中
            if (state[i])
            {
                // //将处于动态检测框中的点去除
                // for (auto vit_area = mvDynamicArea.begin(); vit_area != mvDynamicArea.end(); vit_area++)
                // {
                //     if (vit_area->contains(cv::Point(x1, y1)) || vit_area->contains(cv::Point(x2, y2))) //rect.contains(Point(x, y));  //返回布尔变量，判断rect是否包含Point(x, y)点 
                //     {
                //         flag = 1;
                //         break;
                //     }
                // }
                // if (!flag) {
                //     F_prepoint.push_back(prepoint[i]);
                //     F_nextpoint.push_back(nextpoint[i]);
                // }
                F_prepoint.push_back(prepoint[i]);
                F_nextpoint.push_back(nextpoint[i]);
            }
        }
    }
    // F-Matrix
    //step 4 筛选之后的光流点计算 F 矩阵
    cv::Mat mask = cv::Mat(cv::Size(1, 300), CV_8UC1);
    cv::Mat F = cv::findFundamentalMat(F_prepoint, F_nextpoint, mask, cv::FM_RANSAC, 0.1, 0.99);

    //step 5 目的是为了得到匹配程度更高的F2_prepoint,F2_nextpoint
    for (int i = 0; i < mask.rows; i++)
    {
        if (mask.at<uchar>(i, 0) == 0);
        else
        {
            // Circle(pre_frame, F_prepoint[i], 6, Scalar(255, 255, 0), 3);
            double A = F.at<double>(0, 0)*F_prepoint[i].x + F.at<double>(0, 1)*F_prepoint[i].y + F.at<double>(0, 2);
            double B = F.at<double>(1, 0)*F_prepoint[i].x + F.at<double>(1, 1)*F_prepoint[i].y + F.at<double>(1, 2);
            double C = F.at<double>(2, 0)*F_prepoint[i].x + F.at<double>(2, 1)*F_prepoint[i].y + F.at<double>(2, 2);
            double dd = fabs(A*F_nextpoint[i].x + B*F_nextpoint[i].y + C) / sqrt(A*A + B*B); //Epipolar constraints
            if (dd <= 2)  //角点2到直线的距离小于0.1(米),则符合要求
            {
                F2_prepoint.push_back(F_prepoint[i]);   // 更加精确的符合要求的角点
                F2_nextpoint.push_back(F_nextpoint[i]);
            }
        }
    }

    //并在最后将它们赋值给F_prepoint,F_nextpoint
    F_prepoint = F2_prepoint;
    F_nextpoint = F2_nextpoint;


    //step6 对第3步LK光流法生成的 nextpoint ，利用极线约束进行验证，并且不满足约束的放入T_M 矩阵，如果不满足约束 那应该就是动态点了
    for (int i = 0; i < prepoint.size(); i++)
    {
        if (state[i] != 0)
        {
            double A = F.at<double>(0, 0)*prepoint[i].x + F.at<double>(0, 1)*prepoint[i].y + F.at<double>(0, 2);
            double B = F.at<double>(1, 0)*prepoint[i].x + F.at<double>(1, 1)*prepoint[i].y + F.at<double>(1, 2);
            double C = F.at<double>(2, 0)*prepoint[i].x + F.at<double>(2, 1)*prepoint[i].y + F.at<double>(2, 2);
            // 点到直线的距离
            double dd = fabs(A*nextpoint[i].x + B*nextpoint[i].y + C) / sqrt(A*A + B*B);

            // Judge outliers
            // 认为大于阈值的点是动态点，存入T_M
            if (dd <= limit_dis_epi) continue;  // 閾值大小是1
            T_M.push_back(nextpoint[i]);
        }
    }

}

    bool Frame::IsInDynamic(const int& i)
     {//判断kp是否在含有异常点的动态对象检测框中 不判断是否动态
        const cv::KeyPoint& kp = mvKeys[i];
        bool in_dynamic = false;
         for (auto vit_area = mvDynamicArea.begin(); vit_area != mvDynamicArea.end(); vit_area++)
                        {
                                    if (vit_area->contains(kp.pt)) //rect.contains(Point(x, y));  //返回布尔变量，判断rect是否包含Point(x, y)点 
                                    {
                                        in_dynamic = true;
                                        break;
                                    }
                        }
        return in_dynamic;
    }

    // bool Frame::IsInDynamic(const int& i)
    //  {//判断kp是否在含有异常点的动态对象检测框中
    //     const cv::KeyPoint& kp = mvKeys[i];
    //     bool in_dynamic = false;
    //      for (auto vit_area = mvDynamicArea.begin(); vit_area != mvDynamicArea.end(); vit_area++)
    //                     {
    //                         for (int i = 0; i < T_M.size(); i ++)
    //                         {
    //                             if (vit_area->contains(T_M[i]) )
    //                             {
    //                                 if (vit_area->contains(kp.pt)) //rect.contains(Point(x, y));  //返回布尔变量，判断rect是否包含Point(x, y)点 
    //                                 {
    //                                     in_dynamic = true;
    //                                     break;
    //                                 }
    //                             }
    //                         }
    //                     }
    //     return in_dynamic;
    // }

    bool Frame::IsNotDynamic(const int& i)
     {//判断kp是否在不含有异常点的动态对象检测框中
        const cv::KeyPoint& kp = mvKeys[i];
                for (auto vit_area = mvDynamicArea.begin(); vit_area != mvDynamicArea.end(); vit_area++)
                                {
                                   if (vit_area->contains(kp.pt)) //rect.contains(Point(x, y));  //返回布尔变量，判断rect是否包含Point(x, y)点 
                                    {
                                        for (int i = 0; i < T_M.size(); i ++)
                                        {
                                            if (vit_area->contains(T_M[i]) )
                                            {
                                                    return false;
                                            } 
                                        }
                                        return true;
                                    }
                                }
        
                return false;
    }

    bool Frame::IsInStatic(const int& i)
     {//判断kp是否在静态对象检测框中
        const cv::KeyPoint& kp = mvKeys[i];
        bool in_static = false;
         for (auto vit_area = mvStaticArea.begin(); vit_area != mvStaticArea.end(); vit_area++)
                        {
                                    if (vit_area->contains(kp.pt)) //rect.contains(Point(x, y));  //返回布尔变量，判断rect是否包含Point(x, y)点 
                                    {
                                        in_static = true;
                                        break;
                                    }
                        }
        return in_static;
    }

    bool Frame::IsNotMoving(const int& i)
     {//判断kp是否在不含有异常点的可动对象检测框中
        const cv::KeyPoint& kp = mvKeys[i];
                for (auto vit_area = mvMovableArea.begin(); vit_area != mvMovableArea.end(); vit_area++)
                                {
                                   if (vit_area->contains(kp.pt)) //rect.contains(Point(x, y));  //返回布尔变量，判断rect是否包含Point(x, y)点 
                                    {
                                        for (int i = 0; i < T_M.size(); i ++)
                                        {
                                            if (vit_area->contains(T_M[i]) )
                                            {
                                                    return false;
                                            } 
                                        }
                                        return true;
                                    }
                                }
        
                return false;
    }

    bool Frame::IsInMoving(const int& i)
     {//判断kp是否在含有异常点的可动对象检测框中
        const cv::KeyPoint& kp = mvKeys[i];
        bool in_moving = false;
         for (auto vit_area = mvMovableArea.begin(); vit_area != mvMovableArea.end(); vit_area++)
                        {
                            for (int i = 0; i < T_M.size(); i ++)
                            {
                                if (vit_area->contains(T_M[i]) )
                                {
                                    if (vit_area->contains(kp.pt)) //rect.contains(Point(x, y));  //返回布尔变量，判断rect是否包含Point(x, y)点 
                                    {
                                        in_moving = true;
                                        break;
                                    }
                                }
                            }
                        }
        return in_moving;
    }
        // float Frame::do_InDynamic(const int& i, const cv::Mat &imDepth)
    // {
    //     DynamicFlag = false;
    //     const cv::KeyPoint& kp = mvKeys[i];
    //     for (int n = 0; n < mvDynamicArea.size(); n ++)
    //                     {
    //                         for (int i = 0; i < T_M.size(); i ++)
    //                         {
    //                             if (mvDynamicArea[n].contains(T_M[i]) )
    //                             {
    //                                 if (mvDynamicArea[n].contains(kp.pt)) //rect.contains(Point(x, y));  //返回布尔变量，判断rect是否包含Point(x, y)点 
    //                                 {
    //                                     DynamicFlag = true;
    //                                     float meanDep = 0;

    //                                     int TMnum = get_TMnum(mvDynamicArea[n]); //异常点个数
    //                                     int KPnum = get_KPnum(mvDynamicArea[n], meanDep, imDepth); //特征点个数
    //                                     //cout << TMnum <<' '<<KPnum<< endl;

    //                                     //特征点到识别框四边的最短距离
    //                                     int dis = min({abs(kp.pt.x - mvDynamicArea[n].x), abs(mvDynamicArea[n].width-(kp.pt.x - mvDynamicArea[n].x)), abs(kp.pt.y - mvDynamicArea[n].y), abs(mvDynamicArea[n].height-(kp.pt.y - mvDynamicArea[n].y))});
    //                                     //特征点到识别框顶点的最短距离
    //                                     float dis2 = get_rectDis (mvDynamicArea[n], kp);

    //                                     //根据识别框大小调整平滑系数
    //                                     float a = -0.001*min({mvDynamicArea[n].width, mvDynamicArea[n].height})+0.4;
    //                                     a = a > 0.1? a : 0.1;
                                        
    //                                     float b = -0.0001*min({mvDynamicArea[n].width, mvDynamicArea[n].height})+0.06;
    //                                     b = b > 0.02? b : 0.02;

    //                                     float M_P1 = 1/(2*exp(-a*dis)+1); //边缘附近递减
    //                                     float M_P2 = -0.3*exp(-b*dis2); //去除角落
    //                                     float DynaFac = 50*(TMnum*1.0/KPnum); //动态因子
    //                                     DynaFac = DynaFac < 1? DynaFac : 1;
    //                                     //DynaFac = DynaFac < 1? b : 1;
    //                                     //cout << DynaFac << ' ';

    //                                     float M_P = DynaFac*(M_P1 + M_P2); //移动概率
    //                                     //cout << M_P << endl;
    //                                     return M_P;
    //                                 }
    //                             }
    //                         }
    //                     }
    //     return 0;    
    // }

    // //基于深度分割
    // float Frame::do_InDynamic(const int& i, const cv::Mat &imDepth,vector<float> TMnum,vector<std::pair<int, float>> KPinfo)
    // {
    //     DynamicFlag = false;
    //     const cv::KeyPoint& kp = mvKeys[i];
    //     for (int n = 0; n < mvDynamicArea.size(); n ++)
    //                     {
    //                         for (int i = 0; i < T_M.size(); i ++)
    //                         {
    //                             if (mvDynamicArea[n].contains(T_M[i]) )
    //                             {
    //                                 if (mvDynamicArea[n].contains(kp.pt)) //rect.contains(Point(x, y));  //返回布尔变量，判断rect是否包含Point(x, y)点 
    //                                 {
    //                                     //DynamicFlag = true;
                                        
    //                                     int TM_num = TMnum[n]; //异常点个数
    //                                     std::pair<int, float> KP_info = KPinfo[n]; //特征点个数和平均深度
    //                                     float v = kp.pt.y;
    //                                     float u = kp.pt.x;
    //                                     //cout << imDepth.at<float>(v,u) << ' ';
    //                                     //if (imDepth.at<float>(v,u) == 0) return 0; //距离太远，深度测量失败
    //                                     float depErr = imDepth.at<float>(v,u)-KP_info.second;
    //                                     if ((depErr < 0.5) && (imDepth.at<float>(v,u) != 0)) DynamicFlag = true;
    //                                     depErr = abs(depErr );
    //                                     //cout << depErr << ' ';

    //                                     //抛物线模型
    //                                     float M_P = - 0.3*(depErr*depErr) + 1; //边缘附近递减

    //                                     //新模型
    //                                     // float c = 1.0;
    //                                     // float M_P =exp(-c*depErr*depErr*depErr*depErr) ; //边缘附近递减

    //                                     //正态分布模型
    //                                     // float sigma = 1; 
    //                                     // float miu = 0; //期望，方差
    //                                     // float M_P =  1.0/(sqrt(2*M_PI)*sigma) * exp(-1*(depErr-miu)*(depErr-miu)/(2*sigma*sigma));
    //                                     //cout << M_P << endl;
                                        
    //                                     float DynaFac = 100*(TM_num*1.0/KP_info.first); //大一些对动态点更敏感
    //                                    // float DynaFac = 50*(TM_num*1.0/KP_info.first); //大一些对动态点更敏感
    //                                     DynaFac = DynaFac < 1? DynaFac : 1;
    //                                    //cout << DynaFac << ' ';

    //                                     //M_P = DynaFac*(M_P); //移动概率
    //                                     //cout << M_P << endl;
    //                                     return M_P;
    //                                 }
    //                             }
    //                         }
    //                     }
    //     return 0;    
    // }

     //基于深度分割，考虑叠加问题
    float Frame::do_InDynamic(const int& i, const cv::Mat &imDepth,vector<float> TMnum,vector<std::pair<int, float>> KPinfo)
    {
        int truei = -1;
        DynamicFlag = false;
        const cv::KeyPoint& kp = mvKeys[i];
        for (int n = 0; n < mvDynamicArea.size(); n ++) {
            for (int i = 0; i < T_M.size(); i ++) {
                if (mvDynamicArea[n].contains(T_M[i]) ) {
                    if (mvDynamicArea[n].contains(kp.pt)) //rect.contains(Point(x, y));  //返回布尔变量，判断rect是否包含Point(x, y)点 
                    {
                        if (truei == -1) truei = n;
                        else if (KPinfo[n].second > KPinfo[truei].second) truei = n;
                    }
                }
            }
        }

        if (truei != -1) {
            int TM_num = TMnum[truei]; //异常点个数
            std::pair<int, float> KP_info = KPinfo[truei]; //特征点个数和平均深度
            float v = kp.pt.y;
            float u = kp.pt.x;
            //cout << imDepth.at<float>(v,u) << ' ';
            //if (imDepth.at<float>(v,u) == 0) return 0; //距离太远，深度测量失败
            float depErr = imDepth.at<float>(v,u)-KP_info.second;
            if ((depErr < 0.5) && (imDepth.at<float>(v,u) != 0)) DynamicFlag = true;
            depErr = abs(depErr );
            //cout << depErr << ' ';

            //抛物线模型
            float M_P = - 0.3*(depErr*depErr) + 1; //边缘附近递减

            //新模型
            // float c = 1.0;
            // float M_P =exp(-c*depErr*depErr*depErr*depErr) ; //边缘附近递减

            //正态分布模型
            // float sigma = 1; 
            // float miu = 0; //期望，方差
            // float M_P =  1.0/(sqrt(2*M_PI)*sigma) * exp(-1*(depErr-miu)*(depErr-miu)/(2*sigma*sigma));
            //cout << M_P << endl;
            
            float DynaFac = 100*(TM_num*1.0/KP_info.first); //大一些对动态点更敏感
            // float DynaFac = 50*(TM_num*1.0/KP_info.first); //大一些对动态点更敏感
            DynaFac = DynaFac < 1? DynaFac : 1;
            //cout << DynaFac << ' ';

            //M_P = DynaFac*(M_P); //移动概率
            //cout << M_P << endl;
            return M_P;
        }

        return 0;    
    }

    float Frame::do_InMoving(const int& i, const cv::Mat &imDepth,vector<float> TMnum,vector<std::pair<int, float>> KPinfo)
    {
        MovingFlag = false;
        const cv::KeyPoint& kp = mvKeys[i];
        for (int n = 0; n < mvMovableArea.size(); n ++)
                        {
                            for (int i = 0; i < T_M.size(); i ++)
                            {
                                if (mvMovableArea[n].contains(T_M[i]) )
                                {
                                    if (mvMovableArea[n].contains(kp.pt)) //rect.contains(Point(x, y));  //返回布尔变量，判断rect是否包含Point(x, y)点 
                                    {
                                        for (auto vit_area1 = mvDynamicArea.begin(); vit_area1 != mvDynamicArea.end(); vit_area1++) {
                                            if (vit_area1->contains(T_M[i]) ) {
                                                return 0;
                                            }
                                        }
                                                                                   
                                        int TM_num = TMnum[n]; //异常点个数
                                        std::pair<int, float> KP_info = KPinfo[n]; //特征点个数和平均深度
                                        float v = kp.pt.y;
                                        float u = kp.pt.x;
                                        //cout << imDepth.at<float>(v,u) << ' ';
                                        //if (imDepth.at<float>(v,u) == 0) return 0; //距离太远，深度测量失败
                                        float depErr = abs(imDepth.at<float>(v,u)-KP_info.second);
                                        //cout << depErr << ' ';

                                        //抛物线模型
                                        float M_P = - 0.3*(depErr*depErr) + 1; //边缘附近递减

                                        //新模型
                                        // float c = 1.0;
                                        // float M_P =exp(-c*depErr*depErr*depErr*depErr) ; //边缘附近递减

                                        //正态分布模型
                                        // float sigma = 1; 
                                        // float miu = 0; //期望，方差
                                        // float M_P =  1.0/(sqrt(2*M_PI)*sigma) * exp(-1*(depErr-miu)*(depErr-miu)/(2*sigma*sigma));
                                        //cout << M_P << endl;
                                        
                                        float DynaFac = 50*(TM_num*1.0/KP_info.first); //大一些对动态点更敏感
                                        DynaFac = DynaFac < 1? DynaFac : 1;
                                       //cout << DynaFac << ' ';

                                        M_P = DynaFac*(M_P); //移动概率
                                        //cout << M_P << endl;
                                        return M_P;
                                        
                                    }
                                }
                            }
                        }
        return 0;    
    }

    // float Frame::do_InMoving(const int& i, const cv::Mat &imDepth)
    // {
    //     MovingFlag = false;
    //     const cv::KeyPoint& kp = mvKeys[i];
    //     for (int n = 0; n < mvMovableArea.size(); n ++)
    //                     {
    //                         for (int i = 0; i < T_M.size(); i ++)
    //                         {
    //                             if (mvMovableArea[n].contains(T_M[i]) )
    //                             {
    //                                 if (mvMovableArea[n].contains(kp.pt)) //rect.contains(Point(x, y));  //返回布尔变量，判断rect是否包含Point(x, y)点 
    //                                 {
    //                                     for (auto vit_area1 = mvDynamicArea.begin(); vit_area1 != mvDynamicArea.end(); vit_area1++) {
    //                                         if (vit_area1->contains(T_M[i]) ) {
    //                                             return 0;
    //                                         }
    //                                     }

    //                                                 MovingFlag = true;
    //                                                 float meanDep = 0;
    //                                                 int TMnum = get_TMnum(mvMovableArea[n]); //异常点个数
    //                                                 int KPnum = get_KPnum(mvMovableArea[n], meanDep, imDepth); //特征点个数

    //                                                 int dis = min({abs(kp.pt.x -mvMovableArea[n].x), abs(mvMovableArea[n].width-(kp.pt.x - mvMovableArea[n].x)), abs(kp.pt.y - mvMovableArea[n].y), abs(mvMovableArea[n].height-(kp.pt.y - mvMovableArea[n].y))});
    //                                                 float dis2 = get_rectDis (mvMovableArea[n], kp);

    //                                                 float a = -0.001*min({mvMovableArea[n].width, mvMovableArea[n].height})+0.4;
    //                                                 a = a > 0.1? a : 0.1;
    //                                                 float b = -0.0001*min({mvMovableArea[n].width, mvMovableArea[n].height})+0.06;
    //                                                 b = b > 0.02? b : 0.02;      

    //                                                 float M_P1 = 1/(2*exp(-a*dis)+1);
    //                                                 float M_P2 = -0.3*exp(-b*dis2);
    //                                                 float DynaFac = 10*(TMnum*1.0/KPnum); //动态因子
    //                                                 DynaFac = DynaFac < 1? DynaFac : 1;
    //                                                // cout << DynaFac << ' ';

    //                                                 float M_P = DynaFac*(M_P1 + M_P2);
    //                                                 //cout << M_P << endl;
    //                                                 return M_P;                                                
                                            
                                        
    //                                 }
    //                             }
    //                         }
    //                     }
    //     return 0;    
    // }


    // float Frame::do_NotMoving(const int& i)
    //  {//判断kp是否在不含有异常点的可动对象检测框中
    //     const cv::KeyPoint& kp = mvKeys[i];
    //             for (int n = 0; n < mvMovableArea.size(); n ++)
    //                             {
    //                                if (mvMovableArea[n].contains(kp.pt)) //rect.contains(Point(x, y));  //返回布尔变量，判断rect是否包含Point(x, y)点 
    //                                 {
    //                                     for (int i = 0; i < T_M.size(); i ++)
    //                                     {
    //                                         if (mvMovableArea[n].contains(T_M[i]) )
    //                                         {
    //                                                 return 0;
    //                                         } 
    //                                     }
    //                                     int dis = min({abs(kp.pt.x -mvMovableArea[n].x), abs(mvMovableArea[n].width-(kp.pt.x - mvMovableArea[n].x)), abs(kp.pt.y - mvMovableArea[n].y), abs(mvMovableArea[n].height-(kp.pt.y - mvMovableArea[n].y))});
    //                                     float dis2 = get_rectDis (mvMovableArea[n], kp);
    //                                     float a = -0.001*min({mvMovableArea[n].width, mvMovableArea[n].height})+0.4;
    //                                     a = a > 0.1? a : 0.1;
    //                                     float b = -0.0001*min({mvMovableArea[n].width, mvMovableArea[n].height})+0.06;
    //                                     b = b > 0.02? b : 0.02;                                        
    //                                     float M_P1 = - 7/(14*exp(-a*dis)+10);
    //                                     float M_P2 = 0.3*exp(-b*dis2);
    //                                     float M_P = M_P1 + M_P2;
    //                                     return M_P;
    //                                 }
    //                             }
        
    //             return 0;
    // }

    // float Frame::do_InStatic(const int& i)
    //  {//判断kp是否在静态对象检测框中
    //     StaticFlag = false;
    //     const cv::KeyPoint& kp = mvKeys[i];
    //      for (int n = 0; n < mvStaticArea.size(); n ++)
    //                     {
    //                                 if (mvStaticArea[n].contains(kp.pt)) //rect.contains(Point(x, y));  //返回布尔变量，判断rect是否包含Point(x, y)点 
    //                                 {
    //                                     StaticFlag = true;
    //                                     int dis = min({abs(kp.pt.x - mvStaticArea[n].x), abs(mvStaticArea[n].width-(kp.pt.x - mvStaticArea[n].x)), abs(kp.pt.y - mvStaticArea[n].y), abs(mvStaticArea[n].height-(kp.pt.y - mvStaticArea[n].y))});
    //                                     //float dis2 = get_rectDis (mvStaticArea[n], kp);
    //                                     float a = -0.001*min({mvStaticArea[n].width, mvStaticArea[n].height})+0.4;
    //                                     a = a > 0.1? a : 0.1;
    //                                     // float b = -0.0001*min({mvStaticArea[n].width, mvStaticArea[n].height})+0.06;
    //                                     // b = b > 0.02? b : 0.02;    
    //                                     float M_P1 = - 8/(16*exp(-a*dis)+10);
    //                                     //float M_P2 = 0.3*exp(-0.05*dis2);
    //                                     float M_P = M_P1; //静态物体规则的偏多，不去除角落
    //                                     return M_P;
    //                                 }
    //                     }
    //     return 0;
    // }

    float Frame::do_InTM(const int& i)
    {
        float dis = 20000;
        for (int n = 0; n < T_M.size(); n ++) 
        {
            float disTemp = get_pointDis(mvKeys[i], T_M[n]);
            dis = dis < disTemp? dis : disTemp;//特征点与异常点的最短距离
        }
        float M_P = 1-exp(0.2*(dis-50));
        //if (StaticFlag == true && M_P > 0) M_P = M_P + 0.8;
        M_P = M_P > 0? M_P : 0;
        //cout << "T_M"<< dis << ' ' << M_P << '0';
        return M_P;    
    }

    float Frame::get_rectDis(cv::Rect2i area, cv::KeyPoint point) { //获得点距离矩形四点的最短距离
        cv::Point tl = area.tl();
        cv::Point tr = area.tl();
        tr.x = tr.x + area.width;
        cv::Point bl = area .br();
        bl.y = bl.y + area.height;
        cv::Point br = area.br();
        float dis = min({get_pointDis(point,tl), get_pointDis(point,tr), get_pointDis(point,bl),get_pointDis(point,br)});
        return dis;
    }

    float Frame::get_pointDis(cv::KeyPoint point, cv::Point poi) { //获得两点之间的距离
        float dis = sqrt((poi.x - point.pt.x)*(poi.x - point.pt.x) + (poi.y - point.pt.y)*(poi.y - point.pt.y));
        return dis;
    }

    int Frame::get_TMnum(cv::Rect2i area) { //获得动态对象异常点个数
        int num = 0;
        for (int i = 0; i < T_M.size(); i ++) {
            if (area.contains(T_M[i])) num++;
        }
        return num;
    }

    // std::pair<int, float> Frame::get_KPnum(cv::Rect2i area, const cv::Mat &imDepth) { //获得动态对象特征点个数和平均深度
    //     int num = 0;
    //     float sumDep = 0;
    //     for (int i = 0; i < mvKeys.size(); i ++) {
    //         if (area.contains(mvKeys[i].pt)) {
    //             float v = mvKeys[i].pt.y;
    //             float u = mvKeys[i].pt.x;
    //             float d = imDepth.at<float>(v,u);
    //             if(d <0.01 || d>10){ // 深度值为0 表示测量失败
    //                 continue;
    //             }
    //             num++;
    //             sumDep = sumDep + d;
    //             //cout << imDepth.at<float>(v,u) << ' ';
    //         }
    //     }
    //     float meanDep = sumDep/num;
    //     //cout << meanDep <<endl;
    //     return std::make_pair(num, meanDep-0.2);
    // }

    std::pair<int, float> Frame::get_KPnum(cv::Rect2i area, const cv::Mat &imDepth) { //获得动态对象特征点个数和平均深度
        int num = 0;
        float sumDep = 0; // 最佳聚类深度和
        float sumDep_ = 0; // 最佳聚类剩余点深度和
        float err = 20000;  //最佳聚类误差
        int result = 0; //最佳聚类点数目
        int result_ = 0;//最佳聚类剩余点数目
        std::vector<cv::Point> points; //用于计算的点集
        std::vector<cv::Point> left_points;//最佳聚类剩余点暂存

        for (int i = 0; i < mvKeys.size(); i ++) {
            if (area.contains(mvKeys[i].pt)) {
                float v = mvKeys[i].pt.y;
                float u = mvKeys[i].pt.x;
                float d = imDepth.at<float>(v,u);
                if(d <0.01 || d>10){ // 深度值为0 表示测量失败
                    continue;
                }
                num++;
                //sumDep = sumDep + d;
                //cout << imDepth.at<float>(v,u) << ' ';
            }
        }
        //float mean = sumDep/num;
        //cout << mean <<' ';

        //std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
        cv::Point tl = area.tl();
        //cout <<tl<<endl;
        int width = area.width;
        int height = area.height;
        // cv::Point point1 = cv::Point(tl.x + width/4, tl.y + height/4);
        // points.push_back(point1);
        // cv::Point point2 = cv::Point(tl.x + width/2, tl.y + height/4);
        // points.push_back(point2);
        // cv::Point point3 = cv::Point(tl.x + 3*(width/4), tl.y + height/4);
        // points.push_back(point3);
        // cv::Point point4 = cv::Point(tl.x + width/4, tl.y + height/2);
        // points.push_back(point4);
        // cv::Point point5 = cv::Point(tl.x + 3*(width/4), tl.y + height/2);
        // points.push_back(point5);
        // cv::Point point6 = cv::Point(tl.x + width/4, tl.y + 3*(height/4));
        // points.push_back(point6);
        // cv::Point point7 = cv::Point(tl.x + width/2, tl.y + 3*(height/4));
        // points.push_back(point7);
        // cv::Point point8 = cv::Point(tl.x + 3*(width/4), tl.y + 3*(height/4));
        // points.push_back(point8);
        // cv::Point point9 = cv::Point(tl.x + width/3, tl.y + height/3);
        // points.push_back(point9);
        // cv::Point point10 = cv::Point(tl.x + 2*(width/3), tl.y + height/3);
        // points.push_back(point10);
        // cv::Point point11 = cv::Point(tl.x + width/3, tl.y + 2*(height/3));
        // points.push_back(point11);
        // cv::Point point12 = cv::Point(tl.x + 2*(width/3), tl.y + 2*(height/3));
        // points.push_back(point12);
        // cv::Point point13 = cv::Point(tl.x + width/2, tl.y + height/2);
        // points.push_back(point13);
        int divi = 10; //获得点集，divi为分割精度
        // for (int n = 1; n < divi; n ++) { //去除边缘部分，共36个点
        //     for (int m = 1; m < divi; m ++) {
        //         cv::Point point = cv::Point(tl.x + n*(width/divi), tl.y + m*(height/divi));
        //         points.push_back(point);
        //     }
        // }

        for (int n = 2; n < (divi-1); n ++) { //去除边缘部分，共25个点
            for (int m = 2; m < (divi-1); m ++) {
                cv::Point point = cv::Point(tl.x + n*(width/divi), tl.y + m*(height/divi));
                points.push_back(point);
            }
        }

        // for (int n = 0; n < points.size(); n ++) {
        //     cout <<points[n]<<' ';
        // }
        // cout <<endl;
        std::vector<cv::Point> tem_left_points; //暂存剩余点

        for (int n = 0; n < points.size(); n ++) {
            float v = points[n].y;
            float u = points[n].x;
            float D = imDepth.at<float>(v,u);
            float temSum = 0;
            float temSum_ = 0;
            float temErr = 0;
            int tem_res = 0;
            int tem_res_ = 0;
            tem_left_points.clear();
            for (int m = 0; m < points.size(); m ++) {
                float v = points[m].y;
                float u = points[m].x;
                float d = imDepth.at<float>(v,u);
                //cout << d <<' ';
                if(d <0.01 || d>10){ // 深度值为0 表示测量失败
                    continue;
                }
                if(abs(D-d) <0.2) { 
                    tem_res++;
                    temSum = temSum +d;
                    //cout << tem_res <<' ';
                }
                else { 
                    tem_res_++;
                    temSum_ = temSum_ +d;
                    tem_left_points.push_back(points[m]);
                    //cout << tem_res <<' ';
                }
            }
            temErr = abs(temSum/(tem_res*1.0)-D); //质心与平均深度的距离作为误差
            //cout << tem_res<< ' '<< result<< ' '<< temErr << ' '<< err << ' ';
            if (tem_res > result) { //优先取数量多的
                result = tem_res;
                sumDep = temSum;
                err = temErr;
                left_points.assign(tem_left_points.begin(), tem_left_points.end());
                result_= tem_res_;
                sumDep_ = temSum_;
                //cout << result << ' '<< sumDep << ' ';
            }
            if (tem_res == result && temErr < err) { //数量相同时，取误差小的
                result = tem_res;
                sumDep = temSum;
                err = temErr;
                left_points.assign(tem_left_points.begin(), tem_left_points.end());
                result_= tem_res_;
                sumDep_ = temSum_;
                //cout << result << ' '<< sumDep << ' ';
            }
            //cout << D <<endl;
        }
        
        float meanDep = sumDep/(result*1.0); //聚类后选择的点的平均深度
        float meanDep_ = sumDep_/(result_*1.0); //剩余点的平均深度
        //cout << meanDep <<endl;

        if (meanDep-meanDep_ > 0.2) { //若平均深度明显大于剩余点，则选择了背景部分，将这部分去除，重新计算
            //cout <<"one more time!"<<endl;
            sumDep = 0;
            err = 20000;
            result = 0;
            points.assign(left_points.begin(), left_points.end());  //将剩余点作为输入点集重新计算

            for (int n = 0; n < points.size(); n ++) {
                float v = points[n].y;
                float u = points[n].x;
                float D = imDepth.at<float>(v,u);
                float temSum = 0;
                float temErr = 0;
                int tem_res = 0;
                for (int m = 0; m < points.size(); m ++) {
                    float v = points[m].y;
                    float u = points[m].x;
                    float d = imDepth.at<float>(v,u);
                    //cout << d <<' ';
                    if(d <0.01 || d>10){ // 深度值为0 表示测量失败
                        continue;
                    }
                    if(abs(D-d) <0.2) { 
                        tem_res++;
                        temSum = temSum +d;
                        //cout << tem_res <<' ';
                    }
                }
                temErr = abs(temSum/(tem_res*1.0)-D);
                //cout << tem_res<< ' '<< result<< ' '<< temErr << ' '<< err << ' ';
                if (tem_res > result) {
                    result = tem_res;
                    sumDep = temSum;
                    err = temErr;
                    //cout << result << ' '<< sumDep << ' ';
                }
                if (tem_res == result && temErr < err) {
                    result = tem_res;
                    sumDep = temSum;
                    err = temErr;
                    //cout << result << ' '<< sumDep << ' ';
                }
                //cout << D <<endl;
            }

            meanDep = sumDep/(result*1.0);  
        }
        // std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
        // double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
        // cout << "segment time  =" << ttrack*1000 << endl;
        //cout << meanDep <<endl;
        return std::make_pair(num, meanDep);
    }



} //namespace ORB_SLAM
