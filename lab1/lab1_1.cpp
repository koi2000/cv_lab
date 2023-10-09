#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

int main() {
    cv::Mat img = cv::imread("../a.png", cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cout << "Error" << std::endl;
        return -1;
    }
    // 提取出alpha通道
    cv::Mat alpha;
    cv::extractChannel(img, alpha, 3);
    // 显示alpha通道
     cv::imshow("Alpha channel", alpha);
    // cv::waitKey(0);
    // 读取新的背景图像
    cv::Mat background = cv::imread("../background.png");
    cv::resize(background, background, img.size());

    if (background.empty()) {
        std::cerr << "无法读取背景图像文件." << std::endl;
        return -1;
    }

    cv::Mat res(img.rows, img.cols, CV_8UC4);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            double opaque = 1.0 * img.at<cv::Vec4b>(i, j)[3] / 255;
            for (int k = 0; k < 3; k++) {
                res.at<cv::Vec4b>(i, j)[k] =
                        opaque * img.at<cv::Vec4b>(i, j)[k] + (1 - opaque) * background.at<cv::Vec3b>(i, j)[k];
            }
            res.at<cv::Vec4b>(i, j)[3] = img.at<cv::Vec4b>(i, j)[3];
        }
    }
    imshow("test", res);
    cv::waitKey(0);
    return 0;
}

