//
// Created by DELL on 2023/10/16.
//
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void customBilateralFilter(const Mat& src, Mat& dst, int d, double sigmaColor, double sigmaSpace) {
    Mat temp;
    cvtColor(src, temp, COLOR_BGR2Lab);

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            Vec3b center = temp.at<Vec3b>(y, x);
            double sumWeight = 0.0;
            Vec3d filteredPixel = Vec3d(0, 0, 0);

            for (int i = -d; i <= d; i++) {
                for (int j = -d; j <= d; j++) {
                    int neighborX = x + i;
                    int neighborY = y + j;

                    if (neighborX >= 0 && neighborX < src.cols && neighborY >= 0 && neighborY < src.rows) {
                        Vec3b neighbor = temp.at<Vec3b>(neighborY, neighborX);

                        double colorDiff = norm(neighbor, center, NORM_L2);
                        double spaceDiff = norm(Point(i, j));
                        double weight = exp(-colorDiff * colorDiff / (2.0 * sigmaColor * sigmaColor) - spaceDiff * spaceDiff / (2.0 * sigmaSpace * sigmaSpace));

                        filteredPixel += neighbor * weight;
                        sumWeight += weight;
                    }
                }
            }

            filteredPixel /= sumWeight;
            temp.at<Vec3b>(y, x) = filteredPixel;
        }
    }

    cvtColor(temp, dst, COLOR_Lab2BGR);
}

int main() {
    Mat inputImage = imread("../lab4/img_1.png", IMREAD_COLOR);
    int d = 10;
    double sigmaColor = 10.0;
    double sigmaSpace = 100.0;
    Mat outputImageCustom;
    Mat outputImageOpenCV;
    clock_t start_1 = clock();
    customBilateralFilter(inputImage, outputImageCustom, d, sigmaColor, sigmaSpace);
    clock_t end_1 = clock();
    cout << "my implementation cost: " << end_1 - start_1 << endl;

    clock_t start_2 = clock();
    bilateralFilter(inputImage, outputImageOpenCV, d, sigmaColor, sigmaSpace);
    clock_t end_2 = clock();
    cout << "opencv implementation cost: " << end_2 - start_2 << endl;

    imshow("origin image", inputImage);
    imshow("Custom Bilateral Filter", outputImageCustom);
    imshow("OpenCV Bilateral Filter", outputImageOpenCV);

    waitKey(0);
    return 0;
}
