#include <iostream>
#include <stdio.h>
#include "openCV.h"
#include "PerformanceGraph.h"

void sample(){
}

int main(int argc, char **argv)
{
	cv::VideoCapture cam(0);
	cv::Mat img(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat kp_img;
	std::vector<cv::KeyPoint> kp;
	cv::Mat dscr;

	if (!cam.isOpened()) {
		std::cout << "camera not found." << std::endl;
		return 0;
	}

	PerformanceGraph pg = PerformanceGraph();

	pg.setCalcFrame(30); // set number of execution per each task
	pg.setGraphSize(1000, 500); // set graph canvas size(width, height)
	pg.setIgnoreFirstFrame(true); // set true if first execution needs long execution time then others

	pg.setOnStart([&]() { // execute before start
		kp.clear();
	});

	pg.setOnBeforeTask([&]() { // execute before every one task
		cam >> img;
	});

	pg.setOnAfterTask([&]() { // execute after every one task
		cv::imshow(pg.getCurrentLabel(), kp_img);
		cv::waitKey(30);
	});

	pg.setOnStartTask([&]() { // execute when eash task start

	});

	pg.setOnFinishTask([&]() { // execute when each task is all finish
		cv::destroyWindow(pg.getCurrentLabel());
		std::cout << pg.getCurrentLabel() << " -> mean : " << pg.getCurrentMean<float>() << " stddev : " << pg.getCurrentStddev<float>() << std::endl;
	});

	pg.setOnFinish([&]() { // execute when all tasks are ends
		cv::destroyAllWindows();
	});

	//// addTask(label, function) ////
	// pg.addTask("test", sample);
	// or
	// pg.addTask("test", [&](){ ... });
	pg.addTask("akaze", [&](){
		cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
		akaze->setNOctaveLayers(1);
		akaze->setNOctaves(1);
		akaze->detectAndCompute(img, cv::noArray(), kp, dscr);
		cv::drawKeypoints(img, kp, kp_img);
	});

	pg.addTask("surf", [&]() {
		cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();
		surf->setNOctaveLayers(1);
		surf->setNOctaves(1);
		surf->detectAndCompute(img, cv::noArray(), kp, dscr);
		cv::drawKeypoints(img, kp, kp_img);
	});

	pg.addTask("sift", [&]() {
		cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create(0, 1);
		sift->detectAndCompute(img, cv::noArray(), kp, dscr);
		cv::drawKeypoints(img, kp, kp_img); 
	});

	pg.run();
	pg.showGraph();
	
	cv::waitKey();
}