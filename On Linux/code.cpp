#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv/cv.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(int argc, const char** argv)
{
	//create cascade classifier object used for face detection
	CascadeClassifier face_cascade;
	CascadeClassifier righteye_cascade;
	CascadeClassifier lefteye_cascade;
	CascadeClassifier mouth_cascade;

	//use haarcascade library
	face_cascade.load("haarcascade_frontalface_alt.xml");
	righteye_cascade.load("haarcascade_righteye_2splits.xml");
	lefteye_cascade.load("haarcascade_lefteye_2splits.xml");
	mouth_cascade.load("haarcascade_mcs_mouth.xml");

	//setup Video Capture device and link it to the first capture device
	//VideoCapture captureDevice;
	//captureDevice.open(0);

	//setup	image files used in the capture process
	Mat captureFrame;
	Mat grayscaleFrame;
	Mat binaryFrame;

	captureFrame = imread("img.jpeg");

	//create a window to present the results
	namedWindow("Mood Detection", 1);

	//create a loop to capture and find faces
	//while(true)
	//{
		//capture a new image frame
		//captureDevice>>captureFrame;

		//convert captured image to grayscale and  equalize
		cvtColor(captureFrame, grayscaleFrame, CV_BGR2GRAY);
		equalizeHist(grayscaleFrame, grayscaleFrame);

		//create a vector array to store the face and eyes found
		std::vector<Rect> faces;

		//find faces and store them in the vector array
		face_cascade.detectMultiScale(grayscaleFrame, faces, 1.1, 3 , CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_SCALE_IMAGE, Size(30,30));

		//draw a rectangle for all found faces in the vector array on original image
		for (int i=0;i<faces.size();i++)
		{
			Point pt1(faces[i].x +  faces[i].width, faces[i].y + faces[i].height);
			Point pt2(faces[i].x, faces[i].y);

			rectangle(captureFrame, pt1, pt2, cvScalar(0, 255, 0, 0), 1, 8, 0);
			
			Mat faceROI = grayscaleFrame( faces[i] );
			//imshow("face", faceROI);
		}

		std::vector<Rect> reye;
		
		//find eyes and store them in the vector array
		righteye_cascade.detectMultiScale(grayscaleFrame, reye, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_SCALE_IMAGE, Size(30,30));

		//draw a rectangle for all found eyes in the vector array on original image
		for (int j=0;j<reye.size();j++)
		{	
			Point pt3(reye[j].x +  reye[j].width, reye[j].y + reye[j].height);
			Point pt4(reye[j].x, reye[j].y);

			rectangle(captureFrame, pt3, pt4, cvScalar(0, 255, 0, 0), 1, 8, 0);

			//convert captured image to binary
			threshold(grayscaleFrame, binaryFrame, 50, 255, cv::THRESH_BINARY);

			Mat reyeROI = binaryFrame(reye[j]);
			//resize...
			Mat dst, dst2, dst3, dst5;
			flip(reyeROI, dst, 0);
			//imshow("flip", dst);
			Mat crop = dst(Rect(3 ,3 , 40, 21));
			flip(crop, dst2, 0);
			Mat element5(5,5,CV_8U,cv::Scalar(1));
			erode(dst2, dst5, Mat(), Point(-1, -1), 2, 1, 1);
			//dilate(dst5, dst3, Mat(), Point(-1, -1), 2, 1, 1);
			//morphologyEx(dst5,dst3,cv::MORPH_OPEN,element5);

			//invert
			Mat inv =  cv::Scalar::all(255) - dst5;
			//imshow("left eye invert", inv);			

			//Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
			std::vector<std::vector<cv::Point> > contours;
			vector<Vec4i> hier;
			//cv::Mat contourOutput = image.clone();
			cv::findContours( inv, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE );

			//Draw the contours
	  		cv::Mat contourImage(inv.size(), CV_8UC3, cv::Scalar(255,255,255));
			for (size_t idx = 0; idx < contours.size(); idx++) {
				cv::drawContours(contourImage, contours, idx, Scalar(0, 0, 0), 1, 8, hier);
			}

			imshow("right eye contour", contourImage);
						
			imshow("right eye erode", dst5);
			//imshow("right eye open", dst3);
			imshow("right eye crop", dst2);
			//imshow("right eye", reyeROI);
		}

		std::vector<Rect> mouth;
		mouth_cascade.detectMultiScale(grayscaleFrame, mouth, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_SCALE_IMAGE, Size(30,30));
		for (int k=0;k<mouth.size();k++)
		{	
			Point pt5(mouth[k].x +  mouth[k].width, mouth[k].y + mouth[k].height);
			Point pt6(mouth[k].x, mouth[k].y);

			rectangle(captureFrame, pt5, pt6, cvScalar(0, 255, 0, 0), 1, 8, 0);
			
			//convert captured image to binary
			threshold(grayscaleFrame, binaryFrame, 90, 255, cv::THRESH_BINARY);

			Mat mouthROI = binaryFrame(mouth[k]);
			//resize...
			Mat dst, dst5;
			Mat crop = mouthROI(Rect(0.5, 0.5, 63, 31));
			Mat element5(5,5,CV_8U,cv::Scalar(1));
			//dilate(crop, dst, Mat(), Point(-1, -1), 2, 1, 1);
			erode(crop, dst5, Mat(), Point(-1, -1), 2, 1, 1);
			//dilate(dst5, dst, Mat(), Point(-1, -1), 2, 1, 1);
			//morphologyEx(crop, dst5, cv::MORPH_OPEN, element5);

			//invert
			Mat inv =  cv::Scalar::all(255) - dst5;

			//Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
			std::vector<std::vector<cv::Point> > contours;
			vector<Vec4i> hier;
			//cv::Mat contourOutput = image.clone();
			cv::findContours( inv, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE );

			//Draw the contours
	  		cv::Mat contourImage(inv.size(), CV_8UC3, cv::Scalar(255,255,255));
			for (size_t idx = 0; idx < contours.size(); idx++) {
				cv::drawContours(contourImage, contours, idx, Scalar(0, 0, 0), 1, 8, hier);
			}

			imshow("mouth contour", contourImage);

			imshow("mouth erosion", dst5);
			//imshow("mouth after dilate", dst);
			imshow("mouth crop", crop);
			//imshow("mouth", mouthROI);
		}

		std::vector<Rect> leye;
		
		//find eyes and store them in the vector array
		lefteye_cascade.detectMultiScale(grayscaleFrame, leye, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_SCALE_IMAGE, Size(30,30));

		//draw a rectangle for all found eyes in the vector array on original image
		for (int l=0;l<leye.size();l++)
		{	
			Point pt7(leye[l].x +  leye[l].width, leye[l].y + leye[l].height);
			Point pt8(leye[l].x, leye[l].y);

			rectangle(captureFrame, pt7, pt8, cvScalar(0, 255, 0, 0), 1, 8, 0);

			//convert captured image to binary
			threshold(grayscaleFrame, binaryFrame, 65, 255, cv::THRESH_BINARY);

			Mat leyeROI = binaryFrame(leye[l]);
			//resize...
			Mat dst, dst2, dst3, dst4, dst5;
			flip(leyeROI, dst, 0);
			//imshow("flip", dst);
			Mat crop = dst(Rect(3 ,3 , 40, 22));
			flip(crop, dst2, 0);
			//erode(dst2, dst3, Mat(), Point(-1, -1), 2, 1, 1);
			Mat element5(5,5,CV_8U,cv::Scalar(1));
			morphologyEx(dst2,dst5,cv::MORPH_OPEN,element5);
			//dilate(dst5, dst3, Mat(), Point(-1, -1), 2, 1, 1);
			//erode(dst5, dst3, Mat(), Point(-1, -1), 2, 1, 1);
			//morphologyEx(dst4,dst5,cv::MORPH_CLOSE,element5);

			//invert
			Mat inv =  cv::Scalar::all(255) - dst5;
			//imshow("left eye invert", inv);			

			//Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
			std::vector<std::vector<cv::Point> > contours;
			vector<Vec4i> hier;
			//cv::Mat contourOutput = image.clone();
			cv::findContours( inv, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE );

			//Draw the contours
	  		cv::Mat contourImage(inv.size(), CV_8UC3, cv::Scalar(255,255,255));
			for (size_t idx = 0; idx < contours.size(); idx++) {
				cv::drawContours(contourImage, contours, idx, Scalar(0, 0, 0), 1, 8, hier);
			}

			imshow("left eye contour", contourImage);
			imshow("left eye crop", dst2);
			imshow("left eye open", dst5);
			//imshow("left eye", leyeROI);
		}			

		//print output
		imshow("Mood Detection", captureFrame);

		//pause for 33ms
		waitKey(0);
	//}

	return 0;
}
