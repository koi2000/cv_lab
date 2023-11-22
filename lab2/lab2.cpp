//
// Created by DELL on 2023/9/21.
//
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat image;

vector<Point2f> quadPoints = {Point2f(0, 0), Point2f(362, 0), Point2f(362, 273), Point2f(0, 273)};
vector<Point2f> targetQuad = {Point2f(0, 0), Point2f(362, 0), Point2f(362, 273), Point2f(0, 273)};
Mat transformedImage;
int selected_vertex = -1;
Point2f previous_point;

void updateImage() {
    // 创建透视变换矩阵
    Mat H = getPerspectiveTransform(quadPoints, targetQuad);
    // 应用透视变换
    warpPerspective(image, transformedImage, H, image.size());
    imshow("Transformed Image", transformedImage);
}


void onMouse(int event, int x, int y, int flags, void *userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        // 检查是否点击了顶点
        for (int i = 0; i < 4; ++i) {
            double distance = norm(Point2f(x, y) - quadPoints[i]);
            if (distance < 10) {
                selected_vertex = i;
                previous_point = Point2f(x, y);
                break;
            }
        }
    } else if (event == EVENT_LBUTTONUP) {
        selected_vertex = -1;
    } else if (event == EVENT_MOUSEMOVE && selected_vertex != -1) {
        // 拖动顶点
        Point2f current_point(x, y);
        quadPoints[selected_vertex] -= current_point - previous_point;
        previous_point = current_point;
        updateImage();
    } else if (event == EVENT_RBUTTONDOWN) {
        // init();
        quadPoints = {Point2f(0, 0), Point2f(362, 0), Point2f(362, 273), Point2f(0, 273)};
        updateImage();
    }
}

int main() {
    image = imread("../lab2/img_1.png");
    if (image.empty()) {
        cout << "读取失败" << endl;
        return 1;
    }
    namedWindow("Transformed Image", WINDOW_NORMAL);
    setMouseCallback("Transformed Image", onMouse);
    updateImage();
    while (true) {
        imshow("Transformed Image", transformedImage);
        char key = waitKey(1);
        if (key == 27) {
            break;
        }
    }
    destroyAllWindows();
    return 0;
}
