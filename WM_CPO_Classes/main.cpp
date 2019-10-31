#include <opencv2/highgui.hpp>
#include <opencv2/world.hpp>
#include <opencv2/imgproc.hpp>

#include <map>
#include <iostream>

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
      "D:/Workspace/Zajêcia/CPO/2019-2020/CPO_WM_19_20/assets/";

  // Map is object which contains pairs (ob1,ob2); (key,value)
  std::map<std::string, cv::Mat> asset_images;
  asset_images["lena"] = cv::imread(assets_path + "lena.jpg", cv::IMREAD_COLOR);
  asset_images["dancer"] = cv::imread(assets_path + "dancer.jpg", cv::IMREAD_COLOR);

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
  /*unsigned char* data = lena_image.data;
  for (int row = lena_image.rows / 2 - 50; row < lena_image.rows / 2 + 50;
       row++) {
    int row_offset = row * lena_image.step;
    for (int col = lena_image.cols / 2 - 50; col < lena_image.cols / 2 + 50;
         col++) {
      int column_offset = col * lena_image.elemSize();
      int pixel_offset = row_offset + column_offset;*/

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
 //   }
 // }

  // Task #10
  //Load images and prepare outputs
  cv::Mat lena_image = asset_images.at("lena").clone();
  cv::Mat dancer_image = asset_images.at("dancer").clone();

  //Adjust lena to dancer

  //lena_image.convertTo(lena_image, dancer_image.type());
  //cv::resize(lena_image, lena_image, dancer_image.size());
   /*cv::Mat image_sum = lena_image + dancer_image;
   cv::Mat image_diff = lena_image - dancer_image;*/
  // Task #11
  // Load images and prepare outputs
  // adjust image types in torder to have better results
  // cv::Mat image_sum, image_normalized, image_diff, image_mul, image_div,
  // image_absdiff; 
  // lena_image.convertTo(lena_image, CV_32FC3, 1.0 / 255.0, 0);
  // dancer_image.convertTo(dancer_image, CV_32FC3, 1.0 / 255.0, 0);
  // cv::add(lena_image, dancer_image, image_sum); //add images
  // cv::normalize(image_sum, image_normalized, 0, 1, cv::NORM_MINMAX);
  // //normalize image 
  // cv::subtract(lena_image, dancer_image, image_diff);
  // //subtract images 
  // cv::multiply(lena_image, dancer_image, image_mul);
  // //multiply images 
  // cv::divide(lena_image, dancer_image, image_div); //divide
  // cv::absdiff(dancer_image,lena_image , image_absdiff); // absolute
  //// difference |x1-x2|

  // Task #12
  // Load images and prepare outputs
  // cv::Mat sum, sum_masked;
  //cv::Mat mask = cv::Mat::zeros(dancer_image.size(), CV_8U);  // create mask
  // cv::circle(mask, cv::Point(1000, 1000), 300, cv::Scalar(255), -1); //add
  // white circle 
  // cv::add(lena_image, dancer_image, sum); //add without mask
  // cv::add(lena_image, dancer_image, sum_masked, mask); //add with mask - what
  // is the difference? cv::bitwise_not(mask,mask); lena_image.copyTo(sum_masked,
  // mask); lena_image.setTo(cv::Scalar(0, 0, 255), mask); //bonus: setTo with
  // mask
  // lena_image.setTo(cv::Scalar(0, 0, 255), mask);
   /*cv::bitwise_not(mask, mask);
   lena_image.copyTo(sum_masked,mask);*/
  // //Task #13

   //lena_image.setTo(cv::Scalar(50, 100, 200));
   //lena_image = cv::Scalar(50, 150, 250);
   //lena_image = 255; //what happend?
   //lena_image = cv::Scalar(255); //any difference?
   //lena_image = cv::Scalar::all(255); // is it white already?

  // Task #14
  // cv::Mat img1 = cv::imread("lena.bmp");
  // rectangular mask
   //cv::Mat mask = cv::Mat::zeros(lena_image.size(), CV_8U);
   //cv::rectangle(mask, cv::Point(210, 210), cv::Point(290, 290),
   //cv::Scalar(255), -1); 
   //lena_image.setTo(cv::Scalar(0, 0, 255), mask); //setTo
   ////with mask 
   //lena_image(cv::Rect(210, 210, 80, 80)).setTo(cv::Scalar(0, 255,255)); //ROI approach

  // Task #15
  // cv::Mat img1 = cv::imread("lena.bmp", 0); //load grayscale
  // Four different ways to solve a problem - possible only with grayscale
   //lena_image = lena_image - cv::Scalar::all(40); //subtract
   //lena_image = lena_image + cv::Scalar::all(40); //add
   //cv::subtract(lena_image, cv::Scalar::all(40), lena_image); //subtract
   //cv::add(lena_image, cv::Scalar::all(40), lena_image); //add

  // Task #16
  /* cv::Mat img1 = cv::imread("lena.bmp");
   cv::multiply(lena_image, cv::Scalar(1, 1, 2), lena_image);*/

  //// Task 17
  /* cv::Mat img1 = cv::imread("lena.bmp");
   cv::Mat img2 = cv::imread("dancer.jpg");*/
  // cv::Mat res1, res2;
  // cv::Mat mask;
  // if (lena_image.type() != dancer_image.type())
  //{
  //	dancer_image.convertTo(dancer_image, lena_image.type(), 1, 0); //adjust image types
  //}
  // cv::Size size = (lena_image.size() + dancer_image.size()) / 2; //new size  as mean value 
  // cv::resize(lena_image, lena_image, size/4); // resize 1st image 
  // cv::resize(dancer_image, dancer_image, size/4); //resize 2nd image 
  // cv::addWeighted(lena_image, 0.4, dancer_image, 0.6, -30, res1); // res =
  // //0.4*img1+0.6*img2-30

  //// BONUS: Crossfade between images
  // for (int i = 0; i < 100; i++)
  //{
  //	cv::addWeighted(lena_image, 0.01*i, dancer_image, 1 - 0.01*i, 0, res1);
  //	cv::imshow("test", res1);
  //	cv::waitKey(20); //20fps
  //}

  ////Task #18
  // cv::Mat img1 = cv::imread("lena.bmp");
  // cv::bitwise_not(lena_image, lena_image);

  //////Task #19
   //cv::Mat img1 = cv::Mat::zeros(500, 500, CV_8U);
   //cv::Mat img2 = cv::Mat::zeros(500, 500, CV_8U);
   //cv::circle(img1, cv::Point(200, 250), 100, cv::Scalar::all(255), -1);
   //cv::circle(img2, cv::Point(300, 250), 100, cv::Scalar::all(255), -1);
  ////Prepare outputs res1,res2,res3,res
   //cv::Mat and_res, or_res, xor_res, not_res;
   //cv::bitwise_and(img1, img2, and_res);
   //cv::bitwise_or(img1, img2, or_res);
   //cv::bitwise_xor(img1, img2, xor_res);
   //cv::bitwise_and(img1, img2, and_res);
   //cv::bitwise_not(and_res, not_res);

  ////Task #20
  /* cv::Mat M0 = cv::Mat::zeros(10, 10, CV_8U);
   cv::Mat M1 = cv::Mat::zeros(10, 10, CV_8U);
   M1 = cv::Scalar(160);
   cv::Mat mask = cv::Mat::zeros(10, 10, CV_8U);
   mask(cv::Rect(3, 3, 1, 1)) = 255;
   M1.copyTo(M0, mask);*/

   
   cv::cvtColor(lena_image, lena_image, cv::COLOR_BGR2GRAY);
   std::cout<< cv::threshold(lena_image, lena_image, 100, 255, cv::THRESH_OTSU);
  return 0;
}