//
// Created by DELL on 2023/11/6.
//
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main() {
  // 读取输入图像（二值图像，其中白色像素是目标）
  Mat binaryImage = imread("../lab7/img_1.png", IMREAD_GRAYSCALE);
  imshow("Binary Transform", binaryImage);
  waitKey(0);
  if (binaryImage.empty()) {
    std::cerr << "Could not open or find the image" << std::endl;
    return -1;
  }

  // 计算距离变换
  Mat distanceTransform;
  distanceTransform = Mat::zeros(binaryImage.size(), CV_32F);

  // 前向扫描
  distanceTransform.setTo(Scalar::all(FLT_MAX));
  for (int y = 0; y < binaryImage.rows; y++) {
    for (int x = 0; x < binaryImage.cols; x++) {
      if (binaryImage.at<uchar>(y, x) == 255) {
        distanceTransform.at<float>(y, x) = 0;
      } else {
        if (y > 0) {
          distanceTransform.at<float>(y, x) =
              std::min(distanceTransform.at<float>(y, x),
                       distanceTransform.at<float>(y - 1, x) + 1);
        }
        if (x > 0) {
          distanceTransform.at<float>(y, x) =
              std::min(distanceTransform.at<float>(y, x),
                       distanceTransform.at<float>(y, x - 1) + 1);
        }
      }
    }
  }

  // 逆向扫描
  for (int y = binaryImage.rows - 1; y >= 0; y--) {
    for (int x = binaryImage.cols - 1; x >= 0; x--) {
      if (y < binaryImage.rows - 1) {
        distanceTransform.at<float>(y, x) =
            std::min(distanceTransform.at<float>(y, x),
                     distanceTransform.at<float>(y + 1, x) + 1);
      }
      if (x < binaryImage.cols - 1) {
        distanceTransform.at<float>(y, x) =
            std::min(distanceTransform.at<float>(y, x),
                     distanceTransform.at<float>(y, x + 1) + 1);
      }
    }
  }

  // 可视化距离场
  normalize(distanceTransform, distanceTransform, 0, 255, NORM_MINMAX);
  distanceTransform.convertTo(distanceTransform, CV_8U);

  // 显示距离场图像
  imshow("Distance Transform", distanceTransform);
  waitKey(0);

  return 0;
}