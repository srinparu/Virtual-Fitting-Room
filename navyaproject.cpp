/* Virtual Fitting Room*/

#include "stdafx.h"
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "xlmParser/rapidxml.hpp"
#include "xlmParser/rapidxml_utils.hpp"
#include <thread>
#include <iostream>
#include <Windows.h>
#include <exception>
#include <string.h>

using namespace cv;
using namespace rapidxml;

void detectUpperBody(Mat*);
void detectFist(Mat*);
void insertImage(Mat, CvRect);
CvSeq *faces = NULL, *fist = NULL;
Mat frame, image;
CvHaarClassifierCascade *upperBodyCascade;
CvMemStorage *upperBodyStorage;
CvHaarClassifierCascade *fistCascade;
CvMemStorage *fistStorage;
char cloth[100] = { '\0' };
xml_node<> *node = NULL;
int changecloth = 1, gestureflag = 0, snapshot = 0;


int main(int argc, char** argv)
{

	VideoCapture capture;		// Capture the video

	Mat left = imread("resources/leftarrow.png", CV_LOAD_IMAGE_UNCHANGED);		// Get the left arrow image for choosing different clothes
	Mat right = imread("resources/rightarrow.png", CV_LOAD_IMAGE_UNCHANGED);	// Get the right arrow image for choosing different clothes
	Mat cam = imread("resources/camera.png", CV_LOAD_IMAGE_UNCHANGED);		    // Get the camera image for Snapshot
	int key = 0;
	char *UpperBodyfile = "cascades/haarcascade_mcs_upperbody.xml";		// Get haarcascade xml file for upperbody detection 
	char *Fistfile = "cascades/fist.xml";							// Get haarcascade xml file for fist detection

	upperBodyCascade = (CvHaarClassifierCascade*)cvLoad(UpperBodyfile, 0, 0, 0);		// Loading upperbody detection haarcascades 
	fistCascade = (CvHaarClassifierCascade*)cvLoad(Fistfile, 0, 0, 0);			// Loading fist detection haarcascades 

	upperBodyStorage = cvCreateMemStorage(0);		// Dynamic data structure storage for upper body
	fistStorage = cvCreateMemStorage(0);			// Dynamic data structure storage for fist

	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);		// Size(width) of the window 
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);		// Size(height) of the window
	capture.open(0);								// Start video feed
	if (!capture.isOpened())
		return -1;									// If video feed fails, exit

	assert(upperBodyCascade && upperBodyStorage);	// Assert turns out the program if its argument turns out to be false
	assert(fistCascade && fistStorage);

	std::thread *t;
	int j = 24;

	file<> xmlFile("resources/clothes.xml");	// Get the collection of clothes
	xml_document<> doc;		
	doc.parse<0>(xmlFile.data());				// Parsing the file data
	node = (((doc.first_node("clothes"))->first_node("shirts"))->first_node());		// Get the shirts one by one from the xml file of clothes

	strcpy_s(cloth, node->value());		// Copying the shirt to cloth

	char a[10];							// Name of snapshot file
	strcpy_s(a, "a");

	while (key != 'q')				// Press q to close the window
	{	
		/* Get the frame */
		capture >> frame;			// Read frame from video feed
		flip(frame, frame, 1);		// Flip around y-axis to get mirror effect for the video

		++j;						// Counter to start detection at specific intervals
		if (j % 25 == 0)			// Which frame to be processed
		{
			t = new std::thread(detectUpperBody, &frame);	// Create thread
			t->join();										// Initializing thread 
			t = new std::thread(detectFist, &frame);		// Create thread
			t->join();										// Initializing thread
			j = 0;
		}

		if (changecloth == 1)	// If palm is detected on the arrows then change the cloth
		{
			image = imread(cloth, CV_LOAD_IMAGE_UNCHANGED);		// Get the cloth to be superimposed on the body
			changecloth = 0;
		}

		for (int i = 0; i < (faces ? faces->total : 0); i++)	
		{

			CvRect *rect = (CvRect*)cvGetSeqElem(faces, i);		// Pointer to upper body

			/* Display Cloths on the body*/
			CvRect rect1(*rect);	
			rect1.y += rect1.height - 100;			// Get the body part by subtracting the face
			insertImage(image, rect1);

			/* Display icons like arrows and camera*/
			CvRect rect2;
			rect2.x = 150; rect2.y = 25; rect2.height = 70; rect2.width = 70;	
			insertImage(left, rect2);

			rect2.x = 400; rect2.y = 25; rect2.height = 70; rect2.width = 70;	
			insertImage(right, rect2);

			rect2.x = 535; rect2.y = 80; rect2.height = 70; rect2.width = 70;	
			insertImage(cam, rect2);

		}

		if (snapshot == 1)		// If fist is detected and is over snapshot icon
		{
			char s[100] = { "snapshots/snapshot" };
			strcat_s(s, a);
			a[0] += 1;
			strcat_s(s, ".jpg");	// Save the snapshot with .jpg extension
			strcat_s(s, "\0");
			std::cout << "snap";
			cv::imwrite(s, frame);
			snapshot = 0;
		}
		cv::imshow("Virtual Fitting Room", frame);		// Name for the window
		key = cvWaitKey(10);
	}

	cvReleaseHaarClassifierCascade(&upperBodyCascade);	// Release the upper-body cascade classifier 
	cvReleaseMemStorage(&upperBodyStorage);				// Release the upper-body storage 
	cvReleaseHaarClassifierCascade(&fistCascade);		// Release the fist cascade classifier 
	cvReleaseMemStorage(&fistStorage);					// Release the fist storage 
	return 0;

}

void insertImage(Mat image, CvRect rect1)		// Function to insert images 
{
	try
	{
		resize(image, image, Size(rect1.width, rect1.height));		// Resize the cloth according to the body(rect1) size

		if (!image.empty())							// If there is cloth image
		{
			for (int y = 0;; y++)
			{
				if (y == image.rows - 1)			// If the image length reaches the body length 
				{
					break;
				}
				if (y + rect1.y == frame.rows - 1)	// If the maximum length of the body or frame is reached
				{
					break;
				}
				cv::Vec3b* src_pixel = frame.ptr<Vec3b>(y + rect1.y);	// y displacement - 3 channel input frame
				cv::Vec4b* ovl_pixel = image.ptr<cv::Vec4b>(y);			// 4 channel cloth image
				src_pixel += rect1.x;									// x displacement

				for (int x = 0; x < image.cols; x++, ++src_pixel, ++ovl_pixel)
				{
					if (x == image.cols - 1)		// If the image width reaches the body width
					{
						break;
					}
					if (x == frame.cols - 1)		// If the maximum width of the body or frame is reached 
					{
						break;
					}
					double alpha = (double)(*ovl_pixel).val[3] / 255.0;	// Get the alpha value between 0 and 1 from 0 to 255 values
					for (int c = 0; c < 3; c++)
					{
						(*src_pixel).val[c] = (double)((*src_pixel).val[c] * (1 - alpha)) + ((*ovl_pixel).val[c] * alpha); // For transparent cloth images
					}
				}
			}
		}
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << '\n';
	}
}

void detectUpperBody(Mat* img)		// Function to detect upper-body
{
	IplImage *image2 = cvCreateImage(cvSize(img->cols, img->rows), 8, 3);	// Create temporary image
	IplImage temp = *img;
	cvCopy(&temp, image2);
	faces = cvHaarDetectObjects(image2, upperBodyCascade, upperBodyStorage, 1.1, 3, 0, cvSize(200, 200)); // Get the detected region
}

void detectFist(Mat* img)			// Function to detcet fist
{
	IplImage *image2 = cvCreateImage(cvSize(img->cols, img->rows), 8, 3);	// Create temporary image
	IplImage temp = *img;
	cvCopy(&temp, image2);
	fist = cvHaarDetectObjects(image2, fistCascade, fistStorage, 1.1, 3, 0, cvSize(20, 20));	// Get the detected region

	for (int i = 0; i < (fist ? fist->total : 0); i++)
	{
		CvRect *rect = (CvRect*)cvGetSeqElem(fist, i);
		if (gestureflag == 0)
		{
			if ((rect->x > 120 && rect->x < 250) && (rect->y > 10 && rect->y < 100)) // Coordinates where to detect fist on left arrow
			{
				if (node->previous_sibling())
					node = node->previous_sibling();	// Switch to previous cloth
				else
					node = node->parent()->last_node();	// Remain on the same cloth

				strcpy_s(cloth, node->value());
				changecloth = 1;
				gestureflag = 1;
			}
			else
				if ((rect->x > 370 && rect->x < 500) && (rect->y > 10 && rect->y < 100)) // Coordinates where to detect fist on right arrow
				{
					if (node->next_sibling())
						node = node->next_sibling();			// Switch to next cloth
					else
						node = node->parent()->first_node();	// Remain on the same cloth

					strcpy_s(cloth, node->value());
					changecloth = 1;
					gestureflag = 1;
				}
				else
					if ((rect->x > 450 && rect->x < 640) && (rect->y > 60 && rect->y < 210)) // Coordinates where to detect fist for snapshot
					{
						snapshot = 1;
						gestureflag = 1;
					}
					else
					{
						gestureflag = 0;
					}
		}
		else
		{
			gestureflag = 0;
		}
	}
}
