#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>


using namespace cv;
using namespace cv::dnn;
using namespace std;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanVal(104.0, 177.0, 123.0);

int main()
{
  cv::dnn::Net nnet = cv::dnn::readNet("deploy.prototxt.txt","res10_300x300_ssd_iter_140000.caffemodel");
  cv::VideoCapture cap("C:\\path");
  if (!cap.isOpened())
  {
    return EXIT_FAILURE;
  }
  cv::Mat blob, frame, detections;
  static const std::string kWinName = "Deep learning image classification in OpenCV";
  for (;;)
  {
    Mat frame;
    cap >> frame; // get a new frame from camera/video or read image

    if (frame.empty())
    {
      waitKey();
      break;
    }

    if (frame.channels() == 4)
      cvtColor(frame, frame, COLOR_BGRA2BGR);

    
    Mat inputBlob = blobFromImage(frame, inScaleFactor,
    Size(inWidth, inHeight), meanVal, false, false); 
    nnet.setInput(inputBlob, "data"); 
    
    Mat detection = nnet.forward("detection_out"); 

    vector<double> layersTimings;
    double freq = getTickFrequency() / 1000;
    double time = nnet.getPerfProfile(layersTimings) / freq;

    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    ostringstream ss;
    ss << "FPS: " << 1000 / time << " ; time: " << time << " ms";
    putText(frame, ss.str(), Point(20, 20), 0, 0.5, Scalar(0, 0, 255));

    
    for (int i = 0; i < detectionMat.rows; i++)
    {
      float confidence = detectionMat.at<float>(i, 2);

      if (confidence > 0.9)
      {
        int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
        int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
        int xRightTop   = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
        int yRightTop   = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

        Rect object((int)xLeftBottom, (int)yLeftBottom,
          (int)(xRightTop - xLeftBottom),
          (int)(yRightTop - yLeftBottom));

        rectangle(frame, object, Scalar(0, 255, 0));

        ss.str("");
        ss << confidence;
        String conf(ss.str());
        String label = "Face: " + conf;
        int baseLine = 0;
        cout << object.x << " " << object.y << " " << object.width << " " << object.height << "\n";
      }
    }

    imshow("detections", frame);
    if (waitKey(1) >= 0) break;
  }



  return EXIT_SUCCESS;
}
 