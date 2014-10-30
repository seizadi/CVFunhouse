This project is based on CVFunhouse https://github.com/jeradesign/CVFunhouse.git

The changes are to support OpenCV 3.0 distribution, also the a build from OpenCV trunk: https://github.com/Itseez/opencv.git @Oct 24 16:29

The built XCode project is under opencv2.framework in XCode project. 

This project should contain everything you need to build and run CVFunhouse
under Xcode 6.1, IOS 8.1. If you run into any problems building or running, please file a bug.

To get started writing your own OpenCV code, try modifying the CVFPassThru
example. It contains thorough comments explaining exactly what you need to do.
Plus it starts out working, so you can easily tell if you break anything as you
hack.

NOTE: CVFunhouse includes a copy of the OpenCV library built as an iOS
framework, opencv2.framework, based on OpenCV 3.0

It was built from OpenCV trunk: https://github.com/Itseez/opencv.git @Oct 24 16:29

OpenCV is licensed separately under similar terms. See the file
"OpenCV license.txt" for details. For more information on OpenCV (including full
source code to the library), see the [OpenCV website](http://opencv.org/).

TODO:
- Lucas Kanade Demo is broken will fix
