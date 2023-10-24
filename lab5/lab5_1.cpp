//
// Created by DELL on 2023/10/22.
//
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main() {
  Mat inputImage = imread("../lab5/img.png", IMREAD_GRAYSCALE); // 读取灰度图像

  Mat outputImage = inputImage.clone(); // 创建输出图像并复制输入图像

  // 计算直方图
  int histogram[256] = {0};
  for (int y = 0; y < inputImage.rows; y++) {
    for (int x = 0; x < inputImage.cols; x++) {
      int pixelValue = inputImage.at<uchar>(y, x);
      histogram[pixelValue]++;
    }
  }

  // 计算累积分布函数
  int cumulativeHistogram[256] = {0};
  cumulativeHistogram[0] = histogram[0];
  for (int i = 1; i < 256; i++) {
    cumulativeHistogram[i] = cumulativeHistogram[i - 1] + histogram[i];
  }

  // 直方图均衡化
  int totalPixels = inputImage.rows * inputImage.cols;
  for (int y = 0; y < inputImage.rows; y++) {
    for (int x = 0; x < inputImage.cols; x++) {
      int pixelValue = inputImage.at<uchar>(y, x);
      outputImage.at<uchar>(y, x) = static_cast<uchar>(255 * cumulativeHistogram[pixelValue] / totalPixels);
    }
  }

  // 显示原始图像和均衡化后的图像
  imshow("before", inputImage);
  imshow("after", outputImage);

  waitKey(0);
  return 0;
}
