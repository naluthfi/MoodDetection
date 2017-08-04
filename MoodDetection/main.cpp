#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(int argc, const char** argv)
{
	//create cascade classifier object used for face detection
	CascadeClassifier face_cascade;

	//create cascade classifier object used for eyes detection
	CascadeClassifier eyes_cascade;

	//use haarcascade_frontalface_alt.xml library
	face_cascade.load("haarcascade_frontalface_alt.xml");

	//use haarcascade_eye_tree_eyeglasses.xml library
	eyes_cascade.load("haarcascade_eye_tree_eyeglasses.xml");

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
		std::vector<Rect> eyes;

		//find faces and store them in the vector array
		face_cascade.detectMultiScale(grayscaleFrame, faces, 1.1, 3 , CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_SCALE_IMAGE, Size(30,30));

		//find eyes and store them in the vector array
		eyes_cascade.detectMultiScale(grayscaleFrame, eyes, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_SCALE_IMAGE, Size(15,15));

		//draw a rectangle for all found faces in the vector array on original image
		for (int i=0;i<faces.size();i++)
		{
			Point pt1(faces[i].x +  faces[i].width, faces[i].y + faces[i].height);
			Point pt2(faces[i].x, faces[i].y);

			rectangle(captureFrame, pt1, pt2, cvScalar(0, 255, 0, 0), 1, 8, 0);
		}

		//draw a rectangle for all found eyes in the vector array on original image
		for (int i=0;i<eyes.size();i++)
		{
			Point pt1(eyes[i].x +  eyes[i].width, eyes[i].y + eyes[i].height);
			Point pt2(eyes[i].x, eyes[i].y);

			rectangle(captureFrame, pt1, pt2, cvScalar(0, 255, 0, 0), 1, 8, 0);
		}

		//print output
		imshow("outputCapture", captureFrame);

		//pause for 33ms
		waitKey(33);
	}

	return 0;
}
