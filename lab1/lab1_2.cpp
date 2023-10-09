#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// 滑块回调函数
void onSliderChange(int sigmoidParam, void *userdata) {
    sigmoidParam -= 5;
    cv::Mat *inputImage = static_cast<cv::Mat *>(userdata);

    double beta = 1.0;
    beta = beta / (1 + exp(-sigmoidParam)) + 1;
    std::cout << beta << std::endl;
    // cv::Mat adjustedImage = adjustContrast(*inputImage, static_cast<double>(sigmoidParam) / 100.0);
    cv::Mat adjustedImage;
    inputImage->convertTo(adjustedImage, -1, beta, 0);
    cv::imshow("change contrast", adjustedImage);
}

int main() {
    // 读取图像
    cv::Mat inputImage = cv::imread("../b.png");

    if (inputImage.empty()) {
        std::cerr << "can not read image." << std::endl;
        return -1;
    }
    // 创建窗口
    cv::namedWindow("change contrast");

    // 创建滑块控件
    int initialSigmoidParam = 0; // 初始对比度参数
    cv::createTrackbar("contrast", "change contrast",
                       &initialSigmoidParam, 10, onSliderChange, &inputImage);

    // 初始化显示
    onSliderChange(initialSigmoidParam, &inputImage);

    cv::waitKey(0);
    return 0;
}
