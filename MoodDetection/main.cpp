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
	CascadeClassifier eyes_cascade;
	CascadeClassifier mouth_cascade;

	//use haarcascade library
	face_cascade.load("haarcascade_frontalface_alt.xml");
	eyes_cascade.load("haarcascade_eye_tree_eyeglasses.xml");
	mouth_cascade.load("haarcascade_mcs_mouth.xml");

	//setup Video Capture device and link it to the first capture device
	VideoCapture captureDevice;
	captureDevice.open(0);

	//setup	image files used in the capture process
	Mat captureFrame;
	Mat grayscaleFrame;

	//create a window to present the results
	namedWindow("Mood Detection", 1);

	//create a loop to capture and find faces
	while(true)
	{
		//capture a new image frame
		captureDevice>>captureFrame;

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
			std::vector<Rect> eyes;
		
			//find eyes and store them in the vector array
			eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_SCALE_IMAGE, Size(30,30));

			//draw a rectangle for all found eyes in the vector array on original image
			for (int j=0;j<eyes.size();j++)
			{	
				Point center(faces[i].x +  eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
				int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
				circle( captureFrame, center, radius, Scalar( 255, 0, 0 ), 1, 8, 0 );
			}

			Mat eyesROI = grayscaleFrame(eyes[j]);
			imshow("eyes", eyesROI);
			
			std::vector<Rect> mouth;
			mouth_cascade.detectMultiScale(faceROI, mouth, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_SCALE_IMAGE, Size(30,30));
			for (int k=0;k<mouth.size();k++)
			{	
				Point center(faces[i].x +  mouth[k].x + mouth[k].width*0.5, faces[i].y + mouth[k].y + mouth[k].height*0.5);
				int radius = cvRound( (mouth[k].width + mouth[k].height)*0.25 );
				circle( captureFrame, center, radius, Scalar( 255, 0, 0 ), 1, 8, 0 );
			}

		}


		//print output
		imshow("Mood Detection", captureFrame);

		//pause for 33ms
		waitKey(33);
	}

	return 0;
}
