#include <opencv2/highgui.hpp>
#include <opencv2/world.hpp>
#include <opencv2/imgproc.hpp>

#include <map>

std::vector<cv::Rect> roi_grid(const cv::Mat& image, int cols, int rows) {
  std::vector<cv::Rect> rois;
  for (int i = 0; i < cols; i++) {
    for (int j = 0; j < rows; j++) {
      rois.push_back(cv::Rect(i * image.cols / cols, j * image.rows / rows,
                              image.cols / cols, image.rows / rows));
    }
  }
  return rois;
}

cv::Mat mozaic(const cv::Mat& image, const std::vector<cv::Rect> rois,
               std::map<int, int> mapping) {
  cv::Mat mozaic_image = cv::Mat(image.rows, image.cols, image.type());
  for (auto p : mapping) {
    if (p.first < 0 || p.first >= rois.size() || p.second < 0 ||
        p.second >= rois.size()) {
      throw(std::invalid_argument(
          "Mapping indices exceeds roi buffer size; inner exception "));
    }
    image(rois[p.first]).copyTo(mozaic_image(rois[p.second]));
  }

  return mozaic_image;
}

int main() {
  std::string assets_path =
      "D:/Workspace/Zajêcia/CPO/2019-2020/WM_CPO_Classes/assets/";

  // Map is object which contains pairs (ob1,ob2); (key,value)
  std::map<std::string, cv::Mat> asset_images;
  asset_images["lena"] = cv::imread(assets_path + "lena.jpg", cv::IMREAD_COLOR);

  // Task #1
  // CV_EXPORTS_W Mat imread( const String& filename, int flags = IMREAD_COLOR
  // );
  /*cv::Mat lena_image = asset_images.at("lena");
  std::string task_1_winname = "task_1_display";
  cv::namedWindow(task_1_winname, cv::WINDOW_NORMAL);
  cv::imshow(task_1_winname, lena_image);
  cv::waitKey(0);*/

  // Task #2
  // cv::Mat lena_image = asset_images.at("lena");
  // std::string task_1_winname = "task_1_display";
  // cv::namedWindow(task_1_winname, cv::WINDOW_AUTOSIZE);
  // cv::imshow(task_1_winname, lena_image);
  // cv::waitKey(0);
  // cv::destroyAllWindows();
  // lena_image.release(); //delete image from memory

  // TAsk #3,4
  /*cv::Mat lena_image = asset_images.at("lena").clone();
  cv::Rect roi = cv::Rect(300, 300, 500, 500);
  cv::Mat lena_roi = lena_image(roi).clone();
  cv::Mat lena_roi_cloned = lena_image(roi).clone();
  lena_roi = 0.5 * lena_roi;
  lena_roi_cloned = 0.5 * lena_roi_cloned;*/

  // Task 5
  /*cv::Mat lena_image = asset_images.at("lena").clone();
  std::vector<cv::Mat> mozaic_rois;

  auto rois = roi_grid(lena_image, 2, 2);
  cv::Mat mozaic_image = mozaic(
      lena_image, rois, std::map<int, int>{{-2, 0}, {0, 1}, {3, 2}, {1, 3}});*/

  // Task #6

  cv::Mat lena_image = asset_images.at("lena").clone();

  // roi
  /* cv::Rect roi = cv::Rect(lena_image.rows/2 - 50, lena_image.cols/2 - 50,
   100, 100); lena_image(roi) = cv::Scalar(0, 0, 255);*/

  // cv::Mat::at
  //for (int row = lena_image.rows / 2 - 50; row < lena_image.rows / 2 + 50;
  //     row++) {
  //  for (int col = lena_image.cols / 2 - 50; col < lena_image.cols / 2 + 50;
  //       col++) {
  //    // 8UC3 -> cv::Vec3b; vector (b,g,r)
  //    /*lena_image.at<cv::Vec3b>(row, col)[0] = 0;
  //    lena_image.at<cv::Vec3b>(row, col)[1] = 0;
  //    lena_image.at<cv::Vec3b>(row, col)[2] = 255;*/
  //    //lena_image.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 255);
	 // // 8U -> unsigned char
  //    //lena_image.at<unsigned char>(row, col) = 255;
	 // // 32F -> float 64F -> double
	 // // 32FC3 -> cv::Vec3f 64FC3 -> cv::Vec3d
  //    //lena_image.at<cv::Vec3f>(row, col) = cv::Vec3f(0, 0, 1);
  //  }
  //}
  /*for (int row = lena_image.rows / 2 - 50; row < lena_image.rows / 2 + 50;
       row++) {
    cv::Vec3b* image_row = lena_image.ptr<cv::Vec3b>(row);
    for (int col = lena_image.cols / 2 - 50; col < lena_image.cols / 2 + 50;
         col++) {
      image_row[col] = cv::Vec3b(0, 0, 255);
    }
  }*/
  //lena_image.convertTo(lena_image, CV_32FC3, 1 / 255.0);
  unsigned char* data = lena_image.data;
  for (int row = lena_image.rows / 2 - 50; row < lena_image.rows / 2 + 50;
       row++) {
    int row_offset = row * lena_image.step;
    for (int col = lena_image.cols / 2 - 50; col < lena_image.cols / 2 + 50;
         col++) {
      int column_offset = col * lena_image.elemSize();
      int pixel_offset = row_offset + column_offset;

	  //flor floats
	  /*float red = 1;
      float rest = 0;
      std::memcpy(data + pixel_offset, &rest, 4);
      std::memcpy(data + pixel_offset+4, &rest, 4);
      std::memcpy(data + pixel_offset + 8, &red, 4);*/
	  // for uchars
	  /*data[pixel_offset] = 0;
      data[pixel_offset+1] = 0;
      data[pixel_offset+2] = 255;*/
    }
  }
  return 0;
}