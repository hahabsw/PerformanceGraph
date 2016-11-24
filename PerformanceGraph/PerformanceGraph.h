#pragma once
#include <iostream>
#include <stdio.h>
#include <vector>
#include <functional>
#include <numeric>
#include "openCV.h"

class PerformanceGraph {
public:
	PerformanceGraph(int calcFrame = 30, bool ignoreFirstFrame = true, int width = 1000, int height = 500){
		this->ignoreFirstFrame = ignoreFirstFrame;
		this->graphSize = cv::Size(width, height);
		this->calcFrame = calcFrame;
		this->maxVal = 0;
		this->currentTaskIdx = 0;
		this->isFirst = true;
	};
private:
	std::vector<std::function<void()>> tasks;
	cv::Size graphSize;
	bool ignoreFirstFrame;
	bool isFirst;
	int maxVal;
	int calcFrame;
	int currentTaskIdx;
	std::vector<cv::Scalar> colors;
	std::vector<std::vector<int64>> cts;
	std::vector<std::string> labels;
	std::function<void()> startCallback = nullptr;
	std::function<void()> finishCallback = nullptr;
	std::function<void()> beforeTaskCallback = nullptr;
	std::function<void()> afterTaskCallback = nullptr;
	std::function<void()> startTaskCallback = nullptr;
	std::function<void()> finishTaskCallback = nullptr;
	void setColors() {
		cv::Scalar colors[6] = { // preset
			cv::Scalar(255, 0, 0),
			cv::Scalar(0, 255, 0),
			cv::Scalar(0, 0, 255),
			cv::Scalar(255, 255, 0),
			cv::Scalar(0, 255, 255),
			cv::Scalar(255, 0, 255)
		};
		cv::RNG rng(0xFFFFFFFF);

		this->colors = std::vector<cv::Scalar>(colors, colors + sizeof(colors) / sizeof(colors[0]));

		for (int i = this->colors.size(); i < tasks.size(); i++) {
			int icolor = (unsigned)rng;
			this->colors.push_back(cv::Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255));
		}
	}
public:
	void setOnStart(std::function<void()> cb) { startCallback = cb; }
	void setOnFinish(std::function<void()> cb) { finishCallback = cb; }
	void setOnBeforeTask(std::function<void()> cb) { beforeTaskCallback = cb; }
	void setOnAfterTask(std::function<void()> cb) { afterTaskCallback = cb; }
	void setOnFinishTask(std::function<void()> cb) { finishTaskCallback = cb; };
	void setOnStartTask(std::function<void()> cb) { startTaskCallback = cb; };
	void setCalcFrame(int calcFrame) { this->calcFrame = calcFrame; }
	void setIgnoreFirstFrame(bool ignoreFirstFrame) { this->ignoreFirstFrame = ignoreFirstFrame; }
	void setGraphSize(int width, int height) { this->graphSize = cv::Size(width, height); }
	std::vector<std::string> getLables() { return this->labels; }
	int getCurrentTaskIdx() { return this->currentTaskIdx; }
	std::string getCurrentLabel() { return labels[currentTaskIdx];  }
	template <typename T>
	T getMean(int idx) {
		std::vector<int64> v = cts[idx];
		return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
	}
	template <typename T>
	T getCurrentMean() {
		return this->getMean<T>(this->currentTaskIdx);
	}
	template <typename T>
	T getStddev(int idx) {
		std::vector<int64> v = cts[idx];
		T diff_sum;
		T m = this->getMean<T>(idx);
		for (std::vector<int64>::iterator it = v.begin(); it != v.end(); ++it)
			diff_sum += ((*it - m)*(*it - m));
		return diff_sum / v.size();
	}
	template <typename T>
	T getCurrentStddev() {
		return this->getStddev<T>(this->currentTaskIdx);
	}
	void addTask(std::string label, std::function<void()> task) {
		labels.push_back(label);
		tasks.push_back(task);
		cts.push_back(std::vector<int64>());
	}
	void run() {
		setColors();
		if(startCallback != nullptr)
			startCallback();

		while (1)
		{
			if (isFirst && startTaskCallback != nullptr)
				startTaskCallback();

			if(beforeTaskCallback != nullptr)
				beforeTaskCallback();
			int64 st = cv::getTickCount();
			std::function<void()> task = tasks[currentTaskIdx];
			task();
			int64 ctime = (int64)((cv::getTickCount() - st) * 1000 / cv::getTickFrequency());
			if(afterTaskCallback != nullptr)
				afterTaskCallback();

			
			if (!ignoreFirstFrame || (!isFirst && ignoreFirstFrame)) {
				cts[currentTaskIdx].push_back(ctime);
				if (maxVal < ctime)
					maxVal = ctime;
			}
			
			isFirst = false;

			// putText(kp_img, cv::format("Calculation Time : %d ms", (int)((cv::getTickCount() - st) * 1000 / cv::getTickFrequency())), cv::Point(10, 25), 1, 1, cv::Scalar::all(255));
			if (cts[currentTaskIdx].size() == this->calcFrame) {
				if (finishTaskCallback != nullptr)
					finishTaskCallback();
				currentTaskIdx++;
				isFirst = true;
				if (currentTaskIdx == tasks.size()) {
					break;
				}
			}
		}
		if(finishCallback != nullptr)
			finishCallback();
	}

	void showGraph() {
		cv::Mat graph = cv::Mat(graphSize, CV_8UC3, cv::Scalar(0, 0, 0));

		maxVal += 20;

		int scale = graphSize.width / cts[0].size();
		double tscale = (double)graphSize.height / (double)maxVal;

		for (int i = 0; i < tasks.size(); i++) {
			for (int j = 1; j < cts[i].size(); j++) {
				cv::line(graph, cv::Point((j - 1) * scale + 5, graphSize.height - ((double)cts[i][j - 1] * tscale)), cv::Point(j * scale + 5, graphSize.height - ((double)cts[i][j] * tscale)), colors[i]);
				cv::putText(graph, std::to_string(cts[i][j]), cv::Point(j * scale + 5, graphSize.height - ((double)cts[i][j] * tscale) - 5), 0, 0.3, cv::Scalar(255, 255, 255));
			}
			cv::putText(graph, labels[i], cv::Point(0, graphSize.height - ((double)cts[i][0] * tscale) + 5), 0, 0.5, cv::Scalar(255, 255, 255));
			
		}

		cv::imshow("PerformanceGraph", graph);
	}
};
