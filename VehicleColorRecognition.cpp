#include <iostream>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>

#include "VehicleColorRecognition.h"

using namespace std;
using namespace cv;

string hsvRecognition(const Mat& h_plane, const Mat& s_plane, const Mat& v_plane);

string recognizeVehicleColor(const Mat& bgrImg)
{
    CV_Assert(bgrImg.channels() == 3);

    /******************************************************/
    /*  计算每个像素的 B/G/R 中的最大值与最小值的差  */
    /******************************************************/
    vector<Mat> bgr_planes;
    cv::split(bgrImg, bgr_planes);    // 分离BGR通道

    // 分两次计算得到每个像素 BGR 三通道中的最大值
    Mat bgrMaxPixels;
    cv::max(bgr_planes[0], bgr_planes[1], bgrMaxPixels);
    cv::max(bgrMaxPixels, bgr_planes[2], bgrMaxPixels);

    // 分两次计算得到每个像素 BGR 三通道中的最小值
    Mat bgrMinPixels;
    cv::min(bgr_planes[0], bgr_planes[1], bgrMinPixels);
    cv::min(bgrMinPixels, bgr_planes[2], bgrMinPixels);

    // 计算 BGR 三通道的最大值与最小值的差值
    Mat bgrDiff;
    cv::subtract(bgrMaxPixels, bgrMinPixels, bgrDiff);

    // 对上面的差值图像进行增强，增强效果对识别准确率有较大影响
    // OpenCV 没有与 Matlab 中的 imadjust 相对应的函数，替代如下
    cv::equalizeHist(bgrDiff, bgrDiff);    // 替代方案一
    //cv::normalize( bgrDiff, bgrDiff, 0, 255 );  // 替代方案二    
    cv::max(bgrDiff, bgrMinPixels, bgrDiff);

    // Otsu 阈值分割
    Mat bgrMask;
    double otsu_thresh = cv::threshold(bgrDiff, bgrMask, 0, 255, THRESH_BINARY | THRESH_OTSU);
    cv::threshold(bgrDiff, bgrMask, otsu_thresh + 25, 255, THRESH_BINARY);

    //namedWindow( "Car Mask", 1 );
    //imshow( "Car Mask", bgrMask );

    // HSV 三通道操作
    Mat hsvImg;
    cv::cvtColor(bgrImg, hsvImg, CV_BGR2HSV);
    vector<Mat> hsv_planes;
    cv::split(hsvImg, hsv_planes);    // 分离HSV通道

    // 利用二值图 bgrMask 将 HSV 三通道中的背景部分设为 0.
    cv::multiply(hsv_planes[0], bgrMask / 255, hsv_planes[0]);
    cv::multiply(hsv_planes[1], bgrMask / 255, hsv_planes[1]);
    cv::multiply(hsv_planes[2], bgrMask / 255, hsv_planes[2]);

    // 利用 HSV 三通道进行颜色识别
    string color = hsvRecognition(hsv_planes[0], hsv_planes[1], hsv_planes[2]);

    return color;
}


string hsvRecognition(const Mat& h_plane, const Mat& s_plane, const Mat& v_plane)
{
    // 分别计算三个通道的颜色直方图
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    Mat h_hist, s_hist, v_hist;
    calcHist(&h_plane, 1, 0, Mat(), h_hist, 1, &histSize, &histRange, true, false);
    calcHist(&s_plane, 1, 0, Mat(), s_hist, 1, &histSize, &histRange, true, false);
    calcHist(&v_plane, 1, 0, Mat(), v_hist, 1, &histSize, &histRange, true, false);

    // 找出除 0 之外出现次数最多的颜色
    int h_maxVal = 0, s_maxVal = 0, v_maxVal = 0;
    float h_maxNum = 0, s_maxNum = 0, v_maxNum = 0;
    for (int r = 1; r < h_hist.rows; r++) {
        float* h_pointer = h_hist.ptr<float>(r);
        float* s_pointer = s_hist.ptr<float>(r);
        float* v_pointer = v_hist.ptr<float>(r);
        for (int c = 0; c < h_hist.cols; c++) {
            if (h_maxNum < h_pointer[c]) {
                h_maxNum = h_pointer[c];
                h_maxVal = cvRound(r / 180.0 * 255.0);
            }
            if (s_maxNum < s_pointer[c]) {
                s_maxNum = s_pointer[c];
                s_maxVal = r;
            }
            if (v_maxNum < v_pointer[c]) {
                v_maxNum = v_pointer[c];
                v_maxVal = r;
            }
        }
    }

    // 调试时输出中间结果
    //cout<<"h = "<<h_maxVal<<endl;
    //cout<<"s = "<<s_maxVal<<endl;
    //cout<<"v = "<<v_maxVal<<endl;

    // 判断颜色
    string color;
    if (v_maxVal < 46) {
        color = "black";
    }
    else if (v_maxVal < 221 && s_maxVal < 43) {
        color = "gray";
    }
    else if (v_maxVal >= 221 && s_maxVal < 43) {
        color = "white";
    }
    else {
        if (h_maxVal <= 15 || h_maxVal > 222) {
            color = "red";
        }
        else if (h_maxVal > 15 && h_maxVal <= 48) {
            color = "yellow";
        }
        else if (h_maxVal > 48 && h_maxVal <= 109) {
            color = "green";
        }
        else if (h_maxVal > 109 && h_maxVal <= 140) {
            color = "cyan";
        }
        else if (h_maxVal > 140 && h_maxVal <= 175) {
            color = "blue";
        }
        else if (h_maxVal > 175 && h_maxVal <= 222) {
            color = "purple";
        }
    }

    return color;
}