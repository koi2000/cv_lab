//
// Created by DELL on 2023/11/15.
//
#include <opencv2/opencv.hpp>
using namespace cv;

int main() {
  // 读取输入图像
  Mat inputImage = imread("../lab8/img.png", IMREAD_GRAYSCALE);

  if (inputImage.empty()) {
    std::cerr << "Could not open or find the image" << std::endl;
    return -1;
  }

  // 计算Sobel边缘响应
  Mat sobelX, sobelY;
  Sobel(inputImage, sobelX, CV_32F, 1, 0);
  Sobel(inputImage, sobelY, CV_32F, 0, 1);

  // 计算边缘强度和方向
  Mat edgeMagnitude, edgeDirection;
  cartToPolar(sobelX, sobelY, edgeMagnitude, edgeDirection, true);

  // 非极大值抑制
  Mat nonMaxSuppressed;
  nonMaxSuppressed = edgeMagnitude.clone();

  for (int y = 1; y < inputImage.rows - 1; y++) {
    for (int x = 1; x < inputImage.cols - 1; x++) {
      float angle = edgeDirection.at<float>(y, x);

      // 找到相邻两个像素的坐标
      int x1 = x + static_cast<int>(round(cos(angle)));
      int y1 = y + static_cast<int>(round(sin(angle)));
      int x2 = x - static_cast<int>(round(cos(angle)));
      int y2 = y - static_cast<int>(round(sin(angle)));

      // 进行非极大值抑制
      if (edgeMagnitude.at<float>(y, x) <= edgeMagnitude.at<float>(y1, x1) ||
          edgeMagnitude.at<float>(y, x) <= edgeMagnitude.at<float>(y2, x2)) {
        nonMaxSuppressed.at<float>(y, x) = 0;
      }
    }
  }

  // 使用Canny函数进行边缘检测
  Mat cannyResult;
  Canny(inputImage, cannyResult, 50, 150);

  // 显示结果
  imshow("Original Image", inputImage);
  imshow("Sobel Edge Magnitude", edgeMagnitude / 255.0);
  imshow("Non-Maximum Suppressed", nonMaxSuppressed / 255.0);
  imshow("Canny Result", cannyResult);

  waitKey(0);

  return 0;
}
