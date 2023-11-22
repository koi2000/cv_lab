#include <opencv2/opencv.hpp>

void fftshift(cv::Mat &mag) {
  int cx = mag.cols / 2;
  int cy = mag.rows / 2;

  cv::Mat q0(mag, cv::Rect(0, 0, cx, cy));
  cv::Mat q1(mag, cv::Rect(cx, 0, cx, cy));
  cv::Mat q2(mag, cv::Rect(0, cy, cx, cy));
  cv::Mat q3(mag, cv::Rect(cx, cy, cx, cy));

  cv::Mat tmp;
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);
}

int main() {
  // 读取图像
  cv::Mat img = cv::imread("../lab9/img_2.png", cv::IMREAD_GRAYSCALE);
  if (img.empty()) {
    std::cerr << "Error: Could not read the image." << std::endl;
    return -1;
  }
  cv::Mat f;
  img.convertTo(f, CV_64F);
  // 获取图像大小
  int M = cv::getOptimalDFTSize(f.rows);
  int N = cv::getOptimalDFTSize(f.cols);

  // 进行傅里叶变换
  cv::Mat F;
  cv::dft(f, F, cv::DFT_COMPLEX_OUTPUT);
  fftshift(F);

  // 构造 n 阶巴特沃兹陷波器
  double D0 = 7;
  int n = 1;
  int v0 = 120;
  int v1 = 138;
  cv::Mat H = cv::Mat::zeros(F.size(), F.type());
  for (int u = 0; u < M; ++u) {
    for (int v = 0; v < N; ++v) {
      double D1 = std::abs(u - M / 2);
      double D2 = std::abs(v - N / 2);
      double D3 = std::abs(v - v0);
      H.at<cv::Vec2d>(u, v)[0] =
          1.0 / (1 + std::pow(D0 / (D1 * D2 * D3), 2 * n));
    }
  }
  cv::Mat G;
  cv::mulSpectrums(F, H, G, cv::DFT_COMPLEX_OUTPUT);
  fftshift(G);

  // 进行逆傅里叶变换
  cv::Mat g;
  cv::idft(G, g, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

  // 显示结果
  cv::imshow("Original Image", img);
  cv::imshow("Filtered Image", g);

  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}