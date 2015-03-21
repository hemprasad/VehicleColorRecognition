/*
*  车辆颜色识别
*/

#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/highgui/highgui.hpp>

#include "VehicleColorRecognition.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    int imgNum = 1; // 测试车辆的数量
    char filename[200];

    ofstream fout;
    fout.open("result.txt"); // 保存每辆车的颜色

    double t_start = getTickCount();

    for (int i = 1; i <= imgNum; i++) {
        sprintf(filename, "images/%d.jpg", i); // 车辆图片名

        Mat rgbImg = imread(filename);
        string color = recognizeVehicleColor(rgbImg); // 车辆颜色识别
        fout << i << ".jpg    " << color << endl;
    }

    double t_end = getTickCount();
    double t_cost = (t_end - t_start) / getTickFrequency();
    cout << "Cost time = " << t_cost << endl;

    fout.close();

    return 0;
}