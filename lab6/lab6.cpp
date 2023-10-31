#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

// 定义标记状态
enum class MarkingState { NOT_MARKED, MARKING_FOREGROUND, MARKING_BACKGROUND };
// 存储前景和背景像素的颜色信息
vector<Vec3b> foregroundPixels;
vector<Vec3b> backgroundPixels;

Mat inputImage;
// 回调函数，用于处理鼠标事件
void onMouse(int event, int x, int y, int flags, void *userdata) {
  Mat &markedImage = *static_cast<Mat *>(userdata);
  static MarkingState markingState =
      MarkingState::NOT_MARKED; // 静态变量用于保持标记状态

  if (event == EVENT_LBUTTONDOWN) {
    markingState = MarkingState::MARKING_FOREGROUND;
  } else if (event == EVENT_RBUTTONDOWN) {
    markingState = MarkingState::MARKING_BACKGROUND;
  } else if (event == EVENT_MOUSEMOVE) {
    if (markingState == MarkingState::MARKING_FOREGROUND) {
      circle(markedImage, Point(x, y), 3, Scalar(0, 0, 255), -1);
      foregroundPixels.push_back(inputImage.at<Vec3b>(y, x));
    } else if (markingState == MarkingState::MARKING_BACKGROUND) {
      circle(markedImage, Point(x, y), 3, Scalar(0, 255, 0), -1);
      backgroundPixels.push_back(inputImage.at<Vec3b>(y, x));
    }
  } else if (event == EVENT_LBUTTONUP || event == EVENT_RBUTTONUP) {
    markingState = MarkingState::NOT_MARKED;
  }

  imshow("Marked Image", markedImage);
}

int main() {
  inputImage = imread("../lab6/img.png");
  if (inputImage.empty()) {
    cerr << "Could not open or find the image" << endl;
    return -1;
  }

  Mat markedImage = inputImage.clone(); // 创建标记图像

  namedWindow("Marked Image");
  setMouseCallback("Marked Image", onMouse, &markedImage);

  // 显示标记图像并等待用户交互
  imshow("Marked Image", markedImage);

  while (true) {
    char key = waitKey(0);
    if (key == 13) { // 按下回车键
      break;
    }
  }

  // 构建颜色分布
  Mat colorData(foregroundPixels.size() + backgroundPixels.size(), 1,
                CV_8UC1); // 使用单通道图像
  for (size_t i = 0; i < foregroundPixels.size(); i++) {
    Vec3b color = foregroundPixels[i];
    uchar gray_value = static_cast<uchar>(0.299 * color[2] + 0.587 * color[1] +
                                          0.114 * color[0]);
    colorData.at<uchar>(i, 0) = gray_value;
  }
  for (size_t i = 0; i < backgroundPixels.size(); i++) {
    Vec3b color = backgroundPixels[i];
    uchar gray_value = static_cast<uchar>(0.299 * color[2] + 0.587 * color[1] +
                                          0.114 * color[0]);
    colorData.at<uchar>(i + foregroundPixels.size(), 0) = gray_value;
  }

  // 创建EM对象
  Ptr<cv::ml::EM> em = cv::ml::EM::create();
  em->setClustersNumber(2); // 设置GMM组件数量
  em->trainEM(colorData);

  // 获取估计的GMM参数
  Mat means;
  vector<Mat> covs;
  means = em->getMeans();
  em->getCovs(covs);

  // 根据估计的GMM参数对图像进行分割
  Mat probabilityImage(inputImage.size(), CV_32FC1);

  for (int y = 0; y < inputImage.rows; y++) {
    for (int x = 0; x < inputImage.cols; x++) {
      Vec3b pixel = inputImage.at<Vec3b>(y, x);
      Mat sample(1, means.cols, CV_64FC1); // 使用正确的大小
      for (int i = 0; i < means.cols; i++) {
        sample.at<double>(0, i) = static_cast<double>(pixel[i]);
      }
      Mat probs;
      em->predict2(sample, probs);

      // 选择前景概率或背景概率，这里选择前景
      float foregroundProbability = probs.at<float>(0);
      probabilityImage.at<float>(y, x) = foregroundProbability;
    }
  }

  // 根据概率阈值进行分割
  float thresholds = 0.05; // 阈值，根据需要调整
  Mat segmentedImage;
  threshold(probabilityImage, segmentedImage, thresholds, 255, THRESH_BINARY);

  // 显示分割结果
  imshow("Segmented Image", segmentedImage);
  waitKey(0);

  // 保存分割结果
  imwrite("segmented_image.jpg", segmentedImage);

  return 0;
}
