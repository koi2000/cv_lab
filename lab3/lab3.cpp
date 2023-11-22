//
// Created by DELL on 2023/10/9.
//
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat gaussian_filter(Mat img, double sigma, int kernel_size) {
    int height = img.rows;
    int width = img.cols;
    int channel = img.channels();

    Mat out = Mat::zeros(height, width, CV_8UC3);
    // prepare output
    int pad = kernel_size / 2;
    int _x = 0, _y = 0;
    double kernel_sum = 0;
    float kernel[kernel_size][kernel_size];
    for (int y = 0; y < kernel_size; ++y) {
        for (int x = 0; x < kernel_size; ++x) {
            _y = y - pad;
            _x = x - pad;
            kernel[y][x] = 1 / (2 * M_PI * sigma * sigma)
                           * exp(-(_x * _x + _y * _y) / (2 * sigma * sigma));
            kernel_sum += kernel[y][x];
        }
    }
    // 归一化
    for (int y = 0; y < kernel_size; ++y) {
        for (int x = 0; x < kernel_size; ++x) {
            kernel[y][x] /= kernel_sum;
        }
    }
    // 滤波
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channel; ++c) {
                double v = 0;
                for (int dy = -pad; dy < pad + 1; ++dy) {
                    for (int dx = -pad; dx < pad + 1; ++dx) {
                        int xx = x + dx;
                        int yy = y + dy;
                        // 超过边缘的就不处理了
                        if (0 <= xx && xx < width && 0 <= yy && y < height) {
                            v += (double) img.ptr<Vec3b>(yy)[xx][c] * kernel[dy + pad][dx + pad];
                        }
                    }
                }
                out.ptr<Vec3b>(y)[x][c] = v;
            }
        }
    }
    return out;
}

Mat gaussian_filter_acc(Mat img, double sigma, int kernel_size) {
    int height = img.rows;
    int width = img.cols;
    int channel = img.channels();

    Mat out = Mat::zeros(height, width, CV_8UC3);
    Mat out_y = Mat::zeros(height, width, CV_8UC3);
    int pad = kernel_size / 2;
    int _x = 0, _y = 0;
    float kernel_x[kernel_size];
    float kernel_y[kernel_size];
    double kernel_sum = 0;
    for (int x = 0; x < kernel_size; x++) {
        _x = x - pad;
        kernel_x[x] = 1 / sqrt(2 * M_PI * sigma * sigma) * exp(-(_x * _x) / (2 * sigma * sigma));
        kernel_sum += kernel_x[x];
    }
    // 归一化到1
    for (int x = 0; x < kernel_size; x++) {
        kernel_x[x] /= kernel_sum;
    }

    kernel_sum = 0;
    for (int y = 0; y < kernel_size; y++) {
        _y = y - pad;
        kernel_y[y] = 1 / sqrt(2 * M_PI * sigma * sigma) * exp(-(_y * _y) / (2 * sigma * sigma));
        kernel_sum += kernel_y[y];
    }
    // 归一化到1
    for (int y = 0; y < kernel_size; y++) {
        kernel_y[y] /= kernel_sum;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channel; c++) {
                double v = 0;
                for (int dy = -pad; dy < pad + 1; dy++) {
                    int yy = y + dy;
                    // 超过边缘的就不处理了
                    if (0 <= yy && yy < height) {
                        v += (double) img.ptr<Vec3b>(y + dy)[x][c] * kernel_y[dy + pad];
                    }
                }
                out_y.ptr<Vec3b>(y)[x][c] = v;
            }
        }
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channel; c++) {
                double v = 0;
                for (int dx = -pad; dx < pad + 1; dx++) {
                    int xx = x + dx;
                    if (0 <= xx && xx < width) {
                        v += (double) out_y.ptr<Vec3b>(y)[xx][c] * kernel_x[dx + pad];
                    }
                }
                out.ptr<Vec3b>(y)[x][c] = v;
            }
        }
    }

    return out;
}

int main() {
    Mat img = imread("../lab3/img_1.png", IMREAD_COLOR);
    clock_t start_1 = clock();
    int sigma = 1.3;
    int kernel_size = 9;
    Mat out = gaussian_filter(img, sigma, kernel_size);
    clock_t end_1 = clock();
    cout << "my method consume time: " << end_1 - start_1 << endl;

    clock_t start_2 = clock();
    Mat outputImageOpenCV;
    GaussianBlur(img, outputImageOpenCV, Size(kernel_size, kernel_size), sigma);
    clock_t end_2 = clock();
    cout << "opencv method consume time: " << end_2 - start_2 << endl;
    imshow("out", out);
    waitKey(0);
    return 0;
}