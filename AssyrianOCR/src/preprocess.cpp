#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>
#include "../include/preprocess.h"

int detect_skew(const char* fname) {

  //Load file in grayscale (0 arg), get size.
  cv::Mat src = cv::imread(fname, 0);
  cv::Size size = src.size();

  cv::imwrite("Gray_Image.jpg", src);

  //Invert Colors (text has to be white)
  cv::bitwise_not(src, src);

  cv::imwrite("Gray_Image_Inverted.jpg", src);

  //Store lines in vector
  std::vector<cv::Vec4i> lines;

  //Hough Lines to convert image to lines.
  //Arguments: (File, array, step size, theta, votes required, minimum line length, max line gap)
  cv::HoughLinesP(src, lines, 1, CV_PI/180, 100, size.width / 2.f, 1000);

  //Calculates each line's angle made with the horizontal plane and takes the
  //mean of the lines, the mean is the resultant skew estimation.
  cv::Mat disp_lines(size, CV_8UC1, cv::Scalar(0, 0, 0));
  double angle = 0.;
  unsigned nb_lines = lines.size();
  for (unsigned i = 0; i < nb_lines; ++i)
  {
    cv::line(disp_lines, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255, 0 ,0));
    angle += atan2((double)lines[i][3] - lines[i][1], (double)lines[i][2] - lines[i][0]);
  }
  angle /= nb_lines; // mean angle, in radians.

  cv::imwrite("Hough.jpg", disp_lines);
  double skewAngle = angle * 180 / CV_PI;
  if (skewAngle < 0) {skewAngle -= 0.5;}
  else {skewAngle += 0.5;}
  skewAngle = (int)skewAngle;
  return skewAngle;
}

void fix_skew(const char* fname) {
	int skewAngle = detect_skew(fname);
	//Print Skew Angle
	std::cout << "File " << fname << ": " << skewAngle << std::endl;
	cv::Mat src = cv::imread(fname, -1);

	cv::Point2f center((src.cols-1)/2.0, (src.rows-1)/2.0);
    cv::Mat rot = cv::getRotationMatrix2D(center, skewAngle, 1.0);
    // determine bounding rectangle, center not relevant
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), skewAngle).boundingRect2f();
    // adjust transformation matrix
    rot.at<double>(0,2) += bbox.width/2.0 - src.cols/2.0;
    rot.at<double>(1,2) += bbox.height/2.0 - src.rows/2.0;

    cv::Mat dst;
    cv::warpAffine(src, dst, rot, bbox.size());
    cv::imwrite("rotated_im.jpeg", dst);
}

