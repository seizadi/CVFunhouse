//
//  CVFFarneback.m
//  CVFunhouse
//
//  Created by John Brewer on 7/22/12.
//  Copyright (c) 2012 Jera Design LLC. All rights reserved.
//

// Based on the OpenCV example: <opencv>/samples/c/fback_c.c

#import "CVFFarneback.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc_c.h"

using namespace cv;

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, const Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point2i(x,y), Point2i(cvRound(x+fxy.x), cvRound(y+fxy.y)), color);
            circle(cflowmap, Point2i(x,y), 2, color, -1);
        }
}

@interface CVFFarneback () {
    Mat *prevgray;
    Mat *gray;
    Mat *flow;
    Mat *cflow;
}

@end

@implementation CVFFarneback

/*
 *  processIplImage
 *
 *  Inputs:
 *      iplImage: an IplImage in BGRA format, 8 bits per pixel.
 *          YOU ARE RESPONSIBLE FOR CALLING cvReleaseImage on this image.
 *
 *  Outputs:
 *      When you are done, call imageReady: with an RGB, RGBA, or grayscale
 *      IplImage with 8-bits per pixel.
 *
 *      You can call imageReady: from any thread and it will do the right thing.
 *      You can fork as many threads to process the image as you like; just call
 *      imageReady when you are done.
 *
 *      imageReady: will dispose of the IplImage you pass it once the system is
 *      done with it.
 */
-(void)processIplImage:(IplImage*)frame
{
    int firstFrame = (gray == 0);
    if(!gray)
    {
        (gray = new Mat())->create(frame->height, frame->width, CV_8UC1);
        (prevgray = new Mat())->create(gray->rows, gray->cols, CV_8UC1);
        (flow = new Mat())->create(gray->rows, gray->cols, CV_32FC2);
        (cflow = new Mat())->create(gray->rows, gray->cols, CV_8UC3);
    }
    cvCvtColor(frame, gray, CV_BGR2GRAY);
    cvReleaseImage(&frame);
    
    cvCvtColor(gray, cflow, CV_GRAY2BGR);

    if( !firstFrame )
    {
        calcOpticalFlowFarneback(*prevgray, *gray, *flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        drawOptFlowMap(*flow, *cflow, 16, CV_RGB(0, 255, 0));
    }

    cvCopy(gray, prevgray, nil);
    
    // Call imageReady with your new image.
    IplImage *tempImage = (IplImage *) cvAlloc(sizeof(IplImage));
    IplImage *outImage = cvGetImage(cflow, tempImage);
    [self imageReady: outImage];
}

@end
