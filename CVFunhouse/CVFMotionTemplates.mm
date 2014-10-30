//
//  CVFMotionTemplates.m
//  CVFunhouse
//
//  Created by John Brewer on 7/25/12.
//  Copyright (c) 2012 Jera Design LLC. All rights reserved.
//

// Based on the OpenCV example: <opencv>/samples/c/motempl.c

#import "CVFMotionTemplates.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc_c.h"

using namespace cv;

// various tracking parameters (in seconds)
const double MHI_DURATION = 1;
const double MAX_TIME_DELTA = 0.5;
const double MIN_TIME_DELTA = 0.05;
// number of cyclic frame buffer used for motion detection
// (should, probably, depend on FPS)
const int N = 4;

// ring image buffer
IplImage **buf = 0;
int last = 0;

// temporary images
IplImage *mhi = 0; // MHI
IplImage *orient = 0; // orientation
IplImage *mask = 0; // valid orientation mask
IplImage *segmask = 0; // motion segmentation map
CvMemStorage* storage = 0; // temporary storage

void cvUpdateMotionHistory( const void* silhouette, void* mhimg,
                      double timestamp, double mhi_duration )
{
    CvMat  silhstub, *silh = cvGetMat(silhouette, &silhstub);
    CvMat  mhistub, *mhi = cvGetMat(mhimg, &mhistub);
    
    if( !CV_IS_MASK_ARR( silh ))
        CV_Error( CV_StsBadMask, "" );
    
    if( CV_MAT_TYPE( mhi->type ) != CV_32FC1 )
        CV_Error( CV_StsUnsupportedFormat, "" );
    
    if( !CV_ARE_SIZES_EQ( mhi, silh ))
        CV_Error( CV_StsUnmatchedSizes, "" );
    
    CvSize size = cvGetSize( mhi );
    
    if( CV_IS_MAT_CONT( mhi->type & silh->type ))
    {
        size.width *= size.height;
        size.height = 1;
    }
    
    float ts = (float)timestamp;
    float delbound = (float)(timestamp - mhi_duration);
    int x, y;
    
    for( y = 0; y < size.height; y++ )
    {
        const uchar* silhData = silh->data.ptr + silh->step*y;
        float* mhiData = (float*)(mhi->data.ptr + mhi->step*y);
        
        for(x = 0; x < size.width; x++ )
        {
            float val = mhiData[x];
            val = silhData[x] ? ts : val < delbound ? 0 : val;
            mhiData[x] = val;
        }
    }
}

void cvCalcMotionGradient( const CvArr* mhiimg, CvArr* maskimg,
                     CvArr* orientation,
                     double delta1, double delta2,
                     int aperture_size )
{
    CvMat  mhistub, *mhi = cvGetMat(mhiimg, &mhistub);
    CvMat  maskstub, *mask = cvGetMat(maskimg, &maskstub);
    CvMat  orientstub, *orient = cvGetMat(orientation, &orientstub);
    CvMat  dX_min_row, dY_max_row, orient_row, mask_row;
    CvSize size;
    int x, y;
    
    float  gradient_epsilon = 1e-4f * aperture_size * aperture_size;
    float  min_delta, max_delta;
    
    if( !CV_IS_MASK_ARR( mask ))
        CV_Error( CV_StsBadMask, "" );
    
    if( aperture_size < 3 || aperture_size > 7 || (aperture_size & 1) == 0 )
        CV_Error( CV_StsOutOfRange, "aperture_size must be 3, 5 or 7" );
    
    if( delta1 <= 0 || delta2 <= 0 )
        CV_Error( CV_StsOutOfRange, "both delta's must be positive" );
    
    if( CV_MAT_TYPE( mhi->type ) != CV_32FC1 || CV_MAT_TYPE( orient->type ) != CV_32FC1 )
        CV_Error( CV_StsUnsupportedFormat,
                 "MHI and orientation must be single-channel floating-point images" );
    
    if( !CV_ARE_SIZES_EQ( mhi, mask ) || !CV_ARE_SIZES_EQ( orient, mhi ))
        CV_Error( CV_StsUnmatchedSizes, "" );
    
    if( orient->data.ptr == mhi->data.ptr )
        CV_Error( CV_StsInplaceNotSupported, "orientation image must be different from MHI" );
    
    if( delta1 > delta2 )
    {
        double t;
        CV_SWAP( delta1, delta2, t );
    }
    
    size = CvSize(mhi->rows, mhi->cols);
    min_delta = (float)delta1;
    max_delta = (float)delta2;
    CvMat* dX_min = cvCreateMat( mhi->rows, mhi->cols, CV_32F );
    CvMat* dY_max = cvCreateMat( mhi->rows, mhi->cols, CV_32F );
    
    
    // calc Dx and Dy
    cvSobel( mhi, dX_min, 1, 0, aperture_size );
    cvSobel( mhi, dY_max, 0, 1, aperture_size );
    cvGetRow( dX_min, &dX_min_row, 0 );
    cvGetRow( dY_max, &dY_max_row, 0 );
    cvGetRow( orient, &orient_row, 0 );
    cvGetRow( mask, &mask_row, 0 );
    
    // calc gradient
    for( y = 0; y < size.height; y++ )
    {
        dX_min_row.data.ptr = dX_min->data.ptr + y*dX_min->step;
        dY_max_row.data.ptr = dY_max->data.ptr + y*dY_max->step;
        orient_row.data.ptr = orient->data.ptr + y*orient->step;
        mask_row.data.ptr = mask->data.ptr + y*mask->step;
        cvCartToPolar( &dX_min_row, &dY_max_row, 0, &orient_row, 1 );
        
        // make orientation zero where the gradient is very small
        for( x = 0; x < size.width; x++ )
        {
            float dY = dY_max_row.data.fl[x];
            float dX = dX_min_row.data.fl[x];
            
            if( fabs(dX) < gradient_epsilon && fabs(dY) < gradient_epsilon )
            {
                mask_row.data.ptr[x] = 0;
                orient_row.data.i[x] = 0;
            }
            else
                mask_row.data.ptr[x] = 1;
        }
    }
    
    
    cvErode( mhi, dX_min, 0, (aperture_size-1)/2);
    cvDilate( mhi, dY_max, 0, (aperture_size-1)/2);
    
    // mask off pixels which have little motion difference in their neighborhood
    for( y = 0; y < size.height; y++ )
    {
        dX_min_row.data.ptr = dX_min->data.ptr + y*dX_min->step;
        dY_max_row.data.ptr = dY_max->data.ptr + y*dY_max->step;
        mask_row.data.ptr = mask->data.ptr + y*mask->step;
        orient_row.data.ptr = orient->data.ptr + y*orient->step;
        
        for( x = 0; x < size.width; x++ )
        {
            float d0 = dY_max_row.data.fl[x] - dX_min_row.data.fl[x];
            
            if( mask_row.data.ptr[x] == 0 || d0 < min_delta || max_delta < d0 )
            {
                mask_row.data.ptr[x] = 0;
                orient_row.data.i[x] = 0;
            }
        }
    }
}

CvSeq* cvSegmentMotion( const CvArr* mhiimg, CvArr* segmask, CvMemStorage* storage,
                double timestamp, double seg_thresh )
{
    CvSeq* components = 0;
    
    CvMat  mhistub, *mhi = cvGetMat(mhiimg, &mhistub);
    CvMat  maskstub, *mask = cvGetMat(segmask, &maskstub);
    Cv32suf v, comp_idx;
    int stub_val, ts;
    int x, y;
    
    if( !storage )
        CV_Error( CV_StsNullPtr, "NULL memory storage" );
    
    mhi = cvGetMat( mhi, &mhistub );
    mask = cvGetMat( mask, &maskstub );
    
    if( CV_MAT_TYPE( mhi->type ) != CV_32FC1 || CV_MAT_TYPE( mask->type ) != CV_32FC1 )
        CV_Error( CV_BadDepth, "Both MHI and the destination mask" );
    
    if( !CV_ARE_SIZES_EQ( mhi, mask ))
        CV_Error( CV_StsUnmatchedSizes, "" );
    
    CvMat* mask8u = cvCreateMat( mhi->rows + 2, mhi->cols + 2, CV_8UC1 );
    cvZero( mask8u );
    cvZero( mask );
    components = cvCreateSeq( CV_SEQ_KIND_GENERIC, sizeof(CvSeq),
                             sizeof(CvConnectedComp), storage );
    
    v.f = (float)timestamp; ts = v.i;
    v.f = FLT_MAX*0.1f; stub_val = v.i;
    comp_idx.f = 1;
    
    for( y = 0; y < mhi->rows; y++ )
    {
        int* mhi_row = (int*)(mhi->data.ptr + y*mhi->step);
        for( x = 0; x < mhi->cols; x++ )
        {
            if( mhi_row[x] == 0 )
                mhi_row[x] = stub_val;
        }
    }
    
    
    for( y = 0; y < mhi->rows; y++ )
    {
        int* mhi_row = (int*)(mhi->data.ptr + y*mhi->step);
        uchar* mask8u_row = mask8u->data.ptr + (y+1)*mask8u->step + 1;
        
        for( x = 0; x < mhi->cols; x++ )
        {
            if( mhi_row[x] == ts && mask8u_row[x] == 0 )
            {
                CvConnectedComp comp;
                int x1, y1;
                CvScalar _seg_thresh = cvRealScalar(seg_thresh);
                CvPoint seed = cvPoint(x,y);
                
                cvFloodFill( mhi, seed, cvRealScalar(0), _seg_thresh, _seg_thresh,
                            &comp, CV_FLOODFILL_MASK_ONLY + 2*256 + 4, mask8u );
                
                for( y1 = 0; y1 < comp.rect.height; y1++ )
                {
                    int* mask_row1 = (int*)(mask->data.ptr +
                                            (comp.rect.y + y1)*mask->step) + comp.rect.x;
                    uchar* mask8u_row1 = mask8u->data.ptr +
                    (comp.rect.y + y1+1)*mask8u->step + comp.rect.x+1;
                    
                    for( x1 = 0; x1 < comp.rect.width; x1++ )
                    {
                        if( mask8u_row1[x1] > 1 )
                        {
                            mask8u_row1[x1] = 1;
                            mask_row1[x1] = comp_idx.i;
                        }
                    }
                }
                comp_idx.f++;
                cvSeqPush( components, &comp );
            }
        }
    }
    
    for( y = 0; y < mhi->rows; y++ )
    {
        int* mhi_row = (int*)(mhi->data.ptr + y*mhi->step);
        for( x = 0; x < mhi->cols; x++ )
        {
            if( mhi_row[x] == stub_val )
                mhi_row[x] = 0;
        }
    }
    
    return components;
}

CV_IMPL double
cvCalcGlobalOrientation( const void* orientation, const void* maskimg, const void* mhiimg,
                        double curr_mhi_timestamp, double mhi_duration )
{
    int hist_size = 12;
    
    CvMat  mhistub, *mhi = cvGetMat(mhiimg, &mhistub);
    CvMat  maskstub, *mask = cvGetMat(maskimg, &maskstub);
    CvMat  orientstub, *orient = cvGetMat(orientation, &orientstub);
    void*  _orient;
    float _ranges[] = { 0, 360 };
    float* ranges = _ranges;
    int base_orient;
    float shift_orient = 0, shift_weight = 0;
    float a, b, fbase_orient;
    float delbound;
    CvMat mhi_row, mask_row, orient_row;
    int x, y, mhi_rows, mhi_cols;
    
    if( !CV_IS_MASK_ARR( mask ))
        CV_Error( CV_StsBadMask, "" );
    
    if( CV_MAT_TYPE( mhi->type ) != CV_32FC1 || CV_MAT_TYPE( orient->type ) != CV_32FC1 )
        CV_Error( CV_StsUnsupportedFormat,
                 "MHI and orientation must be single-channel floating-point images" );
    
    if( !CV_ARE_SIZES_EQ( mhi, mask ) || !CV_ARE_SIZES_EQ( orient, mhi ))
        CV_Error( CV_StsUnmatchedSizes, "" );
    
    if( mhi_duration <= 0 )
        CV_Error( CV_StsOutOfRange, "MHI duration must be positive" );
    
    if( orient->data.ptr == mhi->data.ptr )
        CV_Error( CV_StsInplaceNotSupported, "orientation image must be different from MHI" );
    
    // calculate histogram of different orientation values
    CvHistogram* hist = cvCreateHist( 1, &hist_size, CV_HIST_ARRAY, &ranges );
    _orient = orient;
    cvCalcArrHist( &_orient, hist, 0, mask );
    
    // find the maximum index (the dominant orientation)
    cvGetMinMaxHistValue( hist, 0, 0, 0, &base_orient );
    fbase_orient = base_orient*360.f/hist_size;
    
    
    // override timestamp with the maximum value in MHI
    cvMinMaxLoc( mhi, 0, &curr_mhi_timestamp, 0, 0, mask );
    
    // find the shift relative to the dominant orientation as weighted sum of relative angles
    a = (float)(254. / 255. / mhi_duration);
    b = (float)(1. - curr_mhi_timestamp * a);
    delbound = (float)(curr_mhi_timestamp - mhi_duration);
    mhi_rows = mhi->rows;
    mhi_cols = mhi->cols;
    
    if( CV_IS_MAT_CONT( mhi->type & mask->type & orient->type ))
    {
        mhi_cols *= mhi_rows;
        mhi_rows = 1;
    }
    
    cvGetRow( mhi, &mhi_row, 0 );
    cvGetRow( mask, &mask_row, 0 );
    cvGetRow( orient, &orient_row, 0 );
    
    /*
     a = 254/(255*dt)
     b = 1 - t*a = 1 - 254*t/(255*dur) =
     (255*dt - 254*t)/(255*dt) =
     (dt - (t - dt)*254)/(255*dt);
     --------------------------------------------------------
     ax + b = 254*x/(255*dt) + (dt - (t - dt)*254)/(255*dt) =
     (254*x + dt - (t - dt)*254)/(255*dt) =
     ((x - (t - dt))*254 + dt)/(255*dt) =
     (((x - (t - dt))/dt)*254 + 1)/255 = (((x - low_time)/dt)*254 + 1)/255
     */
    for( y = 0; y < mhi_rows; y++ )
    {
        mhi_row.data.ptr = mhi->data.ptr + mhi->step*y;
        mask_row.data.ptr = mask->data.ptr + mask->step*y;
        orient_row.data.ptr = orient->data.ptr + orient->step*y;
        
        for( x = 0; x < mhi_cols; x++ )
            if( mask_row.data.ptr[x] != 0 && mhi_row.data.fl[x] > delbound )
            {
                /*
                 orient in 0..360, base_orient in 0..360
                 -> (rel_angle = orient - base_orient) in -360..360.
                 rel_angle is translated to -180..180
                 */
                float weight = mhi_row.data.fl[x] * a + b;
                float rel_angle = orient_row.data.fl[x] - fbase_orient;
                
                rel_angle += (rel_angle < -180 ? 360 : 0);
                rel_angle += (rel_angle > 180 ? -360 : 0);
                
                if( fabs(rel_angle) < 45 )
                {
                    shift_orient += weight * rel_angle;
                    shift_weight += weight;
                }
            }
    }
    
    // add the dominant orientation and the relative shift
    if( shift_weight == 0 )
        shift_weight = 0.01f;
    
    fbase_orient += shift_orient / shift_weight;
    fbase_orient -= (fbase_orient < 360 ? 0 : 360);
    fbase_orient += (fbase_orient >= 0 ? 0 : 360);
    
    return fbase_orient;
}

// parameters:
//  img - input video frame
//  dst - resultant motion picture
//  args - optional parameters
static void  update_mhi( IplImage* img, IplImage* dst, int diff_threshold )
{
    double timestamp = (double)clock()/CLOCKS_PER_SEC; // get current time in seconds
    CvSize size = cvSize(img->width,img->height); // get current frame size
    int i, idx1 = last, idx2;
    IplImage* silh;
    CvSeq* seq;
    CvRect comp_rect;
    double count;
    double angle;
    CvPoint center;
    double magnitude;
    CvScalar color;
    
    // allocate images at the beginning or
    // reallocate them if the frame size is changed
    if( !mhi || mhi->width != size.width || mhi->height != size.height ) {
        if( buf == 0 ) {
            buf = (IplImage**)malloc(N*sizeof(buf[0]));
            memset( buf, 0, N*sizeof(buf[0]));
        }
        
        for( i = 0; i < N; i++ ) {
            cvReleaseImage( &buf[i] );
            buf[i] = cvCreateImage( size, IPL_DEPTH_8U, 1 );
            cvZero( buf[i] );
        }
        cvReleaseImage( &mhi );
        cvReleaseImage( &orient );
        cvReleaseImage( &segmask );
        cvReleaseImage( &mask );
        
        mhi = cvCreateImage( size, IPL_DEPTH_32F, 1 );
        cvZero( mhi ); // clear MHI at the beginning
        orient = cvCreateImage( size, IPL_DEPTH_32F, 1 );
        segmask = cvCreateImage( size, IPL_DEPTH_32F, 1 );
        mask = cvCreateImage( size, IPL_DEPTH_8U, 1 );
    }
    
    cvCvtColor( img, buf[last], CV_BGR2GRAY ); // convert frame to grayscale
    
    idx2 = (last + 1) % N; // index of (last - (N-1))th frame
    last = idx2;
    
    silh = buf[idx2];
    cvAbsDiff( buf[idx1], buf[idx2], silh ); // get difference between frames
    
    cvThreshold( silh, silh, diff_threshold, 1, CV_THRESH_BINARY ); // and threshold it
    cvUpdateMotionHistory( silh, mhi, timestamp, MHI_DURATION ); // update MHI
    
    // convert MHI to blue 8u image
    cvCvtScale( mhi, mask, 255./MHI_DURATION,
               (MHI_DURATION - timestamp)*255./MHI_DURATION );
    cvZero( dst );
    cvMerge( mask, 0, 0, 0, dst );
    
    // calculate motion gradient orientation and valid orientation mask
    cvCalcMotionGradient( mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3 );
    
    if( !storage )
        storage = cvCreateMemStorage(0);
    else
        cvClearMemStorage(storage);
    
    // segment motion: get sequence of motion components
    // segmask is marked motion components map. It is not used further
    seq = cvSegmentMotion( mhi, segmask, storage, timestamp, MAX_TIME_DELTA );
    
    // iterate through the motion components,
    // One more iteration (i == -1) corresponds to the whole image (global motion)
    for( i = 0; i < seq->total; i++ ) {
        
        comp_rect = ((CvConnectedComp*)cvGetSeqElem( seq, i ))->rect;
        if( comp_rect.width + comp_rect.height < 200 ) // reject very small components
            continue;
        color = CV_RGB(255,0,0);
        magnitude = 30;
        
        // select component ROI
        cvSetImageROI( silh, comp_rect );
        cvSetImageROI( mhi, comp_rect );
        cvSetImageROI( orient, comp_rect );
        cvSetImageROI( mask, comp_rect );
        
        // calculate orientation
        angle = cvCalcGlobalOrientation( orient, mask, mhi, timestamp, MHI_DURATION);
        angle = 360.0 - angle;  // adjust for images with top-left origin
        
        count = cvNorm( silh, 0, CV_L1, 0 ); // calculate number of points within silhouette ROI
        
        cvResetImageROI( mhi );
        cvResetImageROI( orient );
        cvResetImageROI( mask );
        cvResetImageROI( silh );
        
        // check for the case of little motion
        if( count < comp_rect.width*comp_rect.height * 0.05 )
            continue;
        
        // draw a clock with arrow indicating the direction
        center = cvPoint( (comp_rect.x + comp_rect.width/2),
                         (comp_rect.y + comp_rect.height/2) );
        
        cvCircle( dst, center, cvRound(magnitude*1.2), color, 3, CV_AA, 0 );
        cvLine( dst, center, cvPoint( cvRound( center.x + magnitude*cos(angle*CV_PI/180)),
                                     cvRound( center.y - magnitude*sin(angle*CV_PI/180))), color, 3, CV_AA, 0 );
    }
}

@interface CVFMotionTemplates () {
    IplImage* motion;
}

@end

@implementation CVFMotionTemplates

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
-(void)processIplImage:(IplImage*)image
{
    if( !motion )
    {
        motion = cvCreateImage( cvSize(image->width,image->height), 8, 3 );
        cvZero( motion );
        motion->origin = image->origin;
    }
    
    update_mhi( image, motion, 30 );
    
    cvReleaseImage(&image);
    
    // Call imageReady with your new image.
    IplImage *outImage = cvCloneImage(motion);
    [self imageReady:outImage];
}

@end
