//
// Created by DELL on 2023/10/22.
//
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// 自定义直方图均衡化函数
void customHistEqualization(Mat& inputImage, Mat& outputImage) {
  int rows = inputImage.rows;
  int cols = inputImage.cols;
  int totalPixels = rows * cols;

  // 分离通道
  vector<Mat> channels;
  split(inputImage, channels);

  // 为每个通道创建直方图
  vector<Mat> hist(3);
  for (int i = 0; i < 3; i++) {
    hist[i] = Mat(1, 256, CV_32F, Scalar(0));
  }

  // 计算直方图
  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < cols; x++) {
      for (int i = 0; i < 3; i++) {
        uchar pixelValue = channels[i].at<uchar>(y, x);
        hist[i].at<float>(0, pixelValue)++;
      }
    }
  }

  // 计算累积直方图
  for (int i = 0; i < 3; i++) {
    for (int j = 1; j < 256; j++) {
      hist[i].at<float>(0, j) += hist[i].at<float>(0, j - 1);
    }
  }

  // 应用直方图均衡化
  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < cols; x++) {
      for (int i = 0; i < 3; i++) {
        uchar pixelValue = channels[i].at<uchar>(y, x);
        channels[i].at<uchar>(y, x) = static_cast<uchar>(255 * hist[i].at<float>(0, pixelValue) / totalPixels);
      }
    }
  }

  // 合并通道
  merge(channels, outputImage);
}

int main() {
  Mat inputImage = imread("../lab4/img_1.png"); // 读取彩色图像

  Mat outputImage = inputImage.clone(); // 创建输出图像并复制输入图像

  // 应用自定义直方图均衡化
  customHistEqualization(inputImage, outputImage);

  // 显示原始图像和均衡化后的图像
  imshow("Original Image", inputImage);
  imshow("Equalized Image", outputImage);

  waitKey(0);
  return 0;
}
