#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <Windows.h>
#include "cnnNet.h"
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <random>
#include <chrono>

cv::Mat roundMat(const cv::Mat& src);

int main()
{
	torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
	size_t num = torch::cuda::device_count();
	std::cout << "device number: " << num << std::endl;

	int tsince = 40;
	std::string folder = "./sourceImgDir";
	std::string outfolder = folder + "_N2F";
	std::filesystem::create_directories(outfolder);

	std::vector<std::string> file_list;
	for (const auto& entry : std::filesystem::directory_iterator(folder))
	{
		file_list.push_back(entry.path().filename().string());
	}


	for (size_t v = 0; v < file_list.size(); v++)
	{
		// 记录起始时间
		std::chrono::steady_clock::time_point time_start = std::chrono::high_resolution_clock::now();

		std::string file_name = file_list[v];
		std::cout << file_name << std::endl;
		std::string file_name_noSuffix = file_name.substr(0, file_name.find_last_of("."));
		int nfile_indx = 0;
		if (file_name == ".")
		{
			continue;
		}

		bool notdone = true;
		double learning_rate = 0.001;
		cv::Mat img = cv::imread(folder + "/" + file_name, cv::IMREAD_GRAYSCALE);
		int nH= img.rows;
		int nW = img.cols;
		cv::Mat sumMat = cv::Mat::zeros(cv::Size(nW, nH), CV_32FC1);
		cv::Mat avgMat = cv::Mat::zeros(cv::Size(nW, nH), CV_32FC1);

		while (notdone)
		{
			// 读取图像判断
			if (img.empty())
			{
				std::cerr << "ERROR reading image: " << file_name << std::endl;
				continue;
			}
			else {
				std::cout << "img shape:(" << img.rows << "," << img.cols << "," << img.channels() << ")" << std::endl;
			}


			double minner, maxer_tmp, minner_tmp, maxer;
			cv::minMaxLoc(img, &minner, &maxer_tmp); // 计算图像中的最小值minner
			img = img - minner; // 图像减去最小值
			cv::minMaxLoc(img, &minner_tmp, &maxer); // 计算减去最小值之后的图像中的最大值
			std::cout << "minner:" << minner << std::endl;

			// 精度类型转换:  CV_8UC1 -> CV_32FC1
			img.convertTo(img, CV_32FC1);
			// 图像归一化
			img = img / maxer;

			// 宽高 偶数化 
			if (img.rows % 2 == 1)
			{
				nH = img.rows - 1;
			}
			if (img.cols % 2 == 1)
			{
				nW = img.cols - 1;
			}

			// 经过修正后的图像（宽、高都为偶数）
			cv::Mat imgZ = img(cv::Rect(0, 0, nW, nH)); // imgZ : CV_32FC1
			cv::imwrite("out/img.tiff", img);
			printf_s("%s %d imgZ shape:[nW=%d,nH=%d]", __FUNCTION__, __LINE__, nW, nH);

			// 分别抽取奇数行、偶数行形成两幅图
			std::vector<torch::Tensor> listImgH; 
			cv::Mat imgIn1 = cv::Mat::zeros(cv::Size(nW, nH / 2), CV_32FC1);
			cv::Mat imgIn2 = cv::Mat::zeros(cv::Size(nW, nH / 2), CV_32FC1);

			// 取图像奇数列作为左图，偶数列作为右图。
			for (int i = 0; i < imgIn1.rows; i++)
			{
				for (int j = 0; j < imgIn1.cols; j++)
				{
					if (j % 2 == 0) // 原数列
					{
						imgIn1.at<float>(i, j) = imgZ.at<float>(2 * i + 1, j);
						imgIn2.at<float>(i, j) = imgZ.at<float>(2 * i, j);
					}
					if (j % 2 == 1)
					{
						imgIn1.at<float>(i, j) = imgZ.at<float>(2 * i, j);
						imgIn2.at<float>(i, j) = imgZ.at<float>(2 * i + 1, j);
					}
				}
			}
			cv::imwrite("out/imgIn1.tiff", imgIn1);
			cv::imwrite("out/imgIn2.tiff", imgIn2);

			// 将cv::Mat 转为 tensor.   （Height,Width,Channels）-> (1,Channels,Height,Width)
			torch::Tensor imgIn1_tensor = torch::from_blob(imgIn1.data, { 1,imgIn1.rows,imgIn1.cols,imgIn1.channels() }, torch::kF32).to(device).clone();
			imgIn1_tensor = imgIn1_tensor.permute({ 0,3,1,2 }); // 交换通道顺序，

			torch::Tensor imgIn2_tensor = torch::from_blob(imgIn2.data, { 1,imgIn2.rows,imgIn2.cols,imgIn2.channels() }, torch::kF32).to(device).clone();
			imgIn2_tensor = imgIn2_tensor.permute({ 0,3,1,2 });

			listImgH.push_back(imgIn1_tensor);
			listImgH.push_back(imgIn2_tensor);

			std::vector<at::Tensor> listImgV;
			cv::Mat imgIn3 = cv::Mat::zeros(cv::Size(nW / 2, nH), CV_32FC1);
			cv::Mat imgIn4 = cv::Mat::zeros(cv::Size(nW / 2, nH), CV_32FC1);

			for (int i = 0; i < imgIn3.rows; i++)
			{
				for (int j = 0; j < imgIn3.cols; j++)
				{
					if (j % 2 == 0)
					{
						imgIn3.at<float>(i, j) = imgZ.at<float>(i, 2 * j + 1);
						imgIn4.at<float>(i, j) = imgZ.at<float>(i, 2 * j);
					}
					if (j % 2 == 1)
					{
						imgIn3.at<float>(i, j) = imgZ.at<float>(i, 2 * j);
						imgIn4.at<float>(i, j) = imgZ.at<float>(i, 2 * j + 1);
					}
				}
			}

			cv::imwrite("out/imgIn3.tiff", imgIn3);
			cv::imwrite("out/imgIn4.tiff", imgIn4);

			torch::Tensor ImgIn3_tensor = torch::from_blob(imgIn3.data, { 1, imgIn3.rows, imgIn3.cols,imgIn3.channels() }, torch::kF32).to(device).clone();
			ImgIn3_tensor = ImgIn3_tensor.permute({ 0, 3, 1, 2 });

			torch::Tensor ImgIn4_tensor = torch::from_blob(imgIn4.data, { 1, imgIn4.rows, imgIn4.cols, imgIn4.channels() }, torch::kF32).to(device).clone();
			ImgIn4_tensor = ImgIn4_tensor.permute({ 0, 3, 1, 2 });

			listImgV.push_back(ImgIn3_tensor);
			listImgV.push_back(ImgIn4_tensor);


			torch::Tensor Img_tensor = torch::from_blob(img.data, { 1, img.rows, img.cols, img.channels() }, torch::kF32).to(device).clone();
			Img_tensor = Img_tensor.permute({ 0, 3, 1, 2 });

#if 1
			//@test 将img转为tensor，然后再转回img，查看数据是否正确转换
			Img_tensor = Img_tensor.to(torch::kCPU);
			at::Tensor Img_tensor_noise = Img_tensor.permute({ 0,2,3,1 }).squeeze().detach();
			cv::Mat imgFromTensor(img.rows, img.cols, CV_32FC1, Img_tensor_noise.data_ptr());
			printf_s("%s %d:: transform Img_tensor_noise to imgFromTensor[mat] data.\n", __FUNCTION__, __LINE__);
			cv::imwrite("out/imgFromTensor.tiff", imgFromTensor);

#endif
			

			std::vector<at::Tensor> listimgV1, listimgV2, listimgH1, listimgH2;
			std::vector<std::vector<at::Tensor>> listimg;

			listimgV1.push_back(listImgV[0]);
			listimgV1.push_back(listImgV[1]);
			listimgV2.push_back(listImgV[1]);
			listimgV2.push_back(listImgV[0]);

			listimgH1.push_back(listImgH[1]);
			listimgH1.push_back(listImgH[0]);
			listimgH2.push_back(listImgH[0]);
			listimgH2.push_back(listImgH[1]);



			listimg.push_back(listimgH1);
			listimg.push_back(listimgH2);
			listimg.push_back(listimgV1);
			listimg.push_back(listimgV2);

			Net net;
			net.to(device);
			printf_s("%s %d Net has been initialized.\n", __FUNCTION__, __LINE__);

			torch::nn::BCELoss criterion = torch::nn::BCELoss();
			torch::optim::Adam optimizer = torch::optim::Adam(net.parameters(), learning_rate);

			double running_loss1 = 0.0;
			double running_loss2 = 0.0;

			double maxpsnr = -DBL_MAX;

			size_t timesince = 0;

			std::vector<cv::Mat> last10;
			std::vector<double> last10psnr;
			cv::Mat cleaned = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
			cv::Mat outlean = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
			while (timesince <= tsince)
			{
				printf_s("%s %d:: while loop begining. timesince = %d - %d. \n", __FUNCTION__, __LINE__, timesince, tsince);
				std::random_device rd;
				std::mt19937 gen(rd());
				std::uniform_int_distribution<> distrib(0, listimg.size() - 1);
				int indx = distrib(gen);
				printf_s("%s %d:: listimg->indx = %d \n", __FUNCTION__, __LINE__, indx);
				std::vector<torch::Tensor> data;
				data = listimg[indx];
				at::Tensor inputs = data[0];
				at::Tensor labello = data[1];
				//梯度清零
				optimizer.zero_grad();
				// 前向传播
				//std::cout << "tensor inputs:" << inputs << std::endl;
				torch::Tensor outputs;
				try
				{
					torch::Tensor inputs_gpu = inputs.to(device);
					printf_s("%s %d:: inputs.to(device).\n", __FUNCTION__, __LINE__);
					//std::cout << __FUNCTION__ << " " << __LINE__ << "inputs_gpu:"<<inputs_gpu << std::endl;
					outputs = net.forward(inputs_gpu);
					//std::cout << __FUNCTION__ << " " << __LINE__ << "outputs:" << outputs << std::endl;

					printf_s("%s %d:: outputs = net.forward(inputs_gpu).\n", __FUNCTION__, __LINE__);
				}
				catch (const std::exception& e)
				{
					printf_s("%s %d:: exception:%s.\n", __FUNCTION__, __LINE__, e.what());
					//std::cout << "exception: " << e.what() << std::endl;
				}

				// 计算交叉熵
				torch::Tensor labello_device = labello.to(device);
				printf_s("%s %d:: labello_device' dim = %d.\n", __FUNCTION__, __LINE__, labello_device.dim());
				printf_s("%s %d:: outputs' dim = %d.\n", __FUNCTION__, __LINE__, outputs.dim());
				auto loss1 = criterion(outputs, labello_device);

				auto loss = loss1;
				running_loss1 += loss1.item().toDouble();
				// 误差反向传播
				loss.backward();
				printf_s("%s %d:: loss.backward().\n", __FUNCTION__, __LINE__);
				// 更新参数
				optimizer.step();
				printf_s("%s %d:: optimizer.step().\n", __FUNCTION__, __LINE__);
				running_loss1 = 0.0;

				{
					torch::NoGradGuard no_grad;

					//cv::Mat out = cleaned * maxer + minner;
					//cv::Mat out = cleaned.clone();
					//cleaned.release();
					//out.convertTo(out, CV_8UC1, maxer, minner);
					last10.push_back(outlean);
					//cv::imwrite("out/out.tiff", out);
					Img_tensor = Img_tensor.to(device);
					at::Tensor outputstest = net.forward(Img_tensor);

					// 提取结果图像
					outputstest = outputstest.to(torch::kCPU);
					at::Tensor outTemp = outputstest.permute({ 0,2,3,1 }).squeeze().detach();
					//outTemp = outTemp.to(torch::kCPU);
					cleaned = cv::Mat(img.rows, img.cols, CV_32FC1, outTemp.data_ptr());
					printf_s("%s %d:: get cleaned data.\n", __FUNCTION__, __LINE__);
					cv::imwrite("out/cleaned.tiff", cleaned);
#if 0
					for (int i = 0; i < cleaned.rows; i++)
					{
						const float* ptr = cleaned.ptr<float>(i);
						for (int j = 0; j < cleaned.cols; j++)
						{
							float pix = ptr[j];
							printf_s("%s %d :: cleaned[%d][%d] = %.10f \n", __FUNCTION__, __LINE__, i, j, pix);
						}
					}
#endif
					Img_tensor = Img_tensor.to(torch::kCPU);
					auto Img_tensor_noise = Img_tensor.permute({ 0,2,3,1 }).squeeze().detach();
					cv::Mat noiseMat(img.rows, img.cols, CV_32FC1, Img_tensor.data_ptr());
					printf_s("%s %d:: get noiseMat data.\n", __FUNCTION__, __LINE__);
					//cv::imwrite("out/noiseMat.tiff", noiseMat);


					cv::Mat diffMat = noiseMat - cleaned;
					//cv::imwrite("out/diffMat.tiff", diffMat);
			
					cv::pow(diffMat, 2, diffMat);
					cv::Scalar mean, stdDev;
					cv::meanStdDev(diffMat, mean, stdDev);
					double ps = -mean[0];
					printf_s("%s %d:: get ps = %.10f \n", __FUNCTION__, __LINE__, ps);
					last10psnr.push_back(ps);
					if (ps > maxpsnr)
					{
						maxpsnr = ps;
						outlean = cleaned * maxer + minner;
						cv::imwrite("out/outlean.tiff", outlean);
						timesince = 0;
						//nfile_indx++;
					}
					else
					{
						timesince += 1.0;
					}
					printf_s("%s %d:: update last10psnr value.\n", __FUNCTION__, __LINE__);
				}
			}

			int nSize = last10.size();
			for (int i = 0; i < nSize; i++)
			{
				cv::Mat tmp = last10.at(i);
				//cv::Mat tmpWrite = tmp * 255;
				//tmpWrite.convertTo(tmpWrite, CV_8UC1);
				//cv::imwrite(outfolder + "/" + file_name_noSuffix + "_last10_" + std::to_string(i) + ".tiff", tmpWrite);
				cv::add(tmp, sumMat, sumMat);
			}
			avgMat = sumMat / last10.size();
			cv::Mat innerMat = avgMat(cv::Rect(1, 1, avgMat.cols - 1, avgMat.rows - 1));
			cv::Mat innerMatOut = innerMat.clone();
			innerMatOut.convertTo(innerMatOut, CV_8UC1);
			cv::imwrite(outfolder + "/" + file_name_noSuffix + "_innerMat.tiff", innerMatOut);
			cv::Scalar meanInnerMat, stdDevInnerMat;
			cv::meanStdDev(innerMat, meanInnerMat, stdDevInnerMat);
			cv::Mat rdMat = roundMat(innerMat - meanInnerMat[0]);
			int nNunZeroNum = cv::countNonZero(rdMat);
			if (nNunZeroNum < 25 && learning_rate != 0.000005)
			{
				learning_rate = 0.000005;
			}
			else
			{
				notdone = false;
				std::chrono::steady_clock::time_point time_end = std::chrono::high_resolution_clock::now();
				float total_inf = std::chrono::duration_cast<std::chrono::microseconds> (time_end - time_start).count() / 1000000.0;
				printf("--- %.f seconds --- \n", total_inf);
			}
		}
		//avgMat = avgMat * 255;
		avgMat.convertTo(avgMat, CV_8UC1);
		cv::imwrite(outfolder + "/" + file_name, avgMat);

	}

	return 0;
}

cv::Mat roundMat(const cv::Mat& src)
{
	cv::Mat dst = src.clone();
	switch (src.depth())
	{
	case CV_32F: {
		for (int row = 0; row < src.rows; row++)
		{
			for (int col = 0; col < src.cols; col++)
			{
				dst.at<float>(row, col) = std::round(src.at<float>(row, col));
			}
		}
		break;
	}
	case CV_64F: {
		for (int row = 0; row < src.rows; row++)
		{
			for (int col = 0; col < src.cols; col++)
			{
				dst.at<double>(row, col) = std::round(src.at<double>(row, col));
			}
		}
		break;
	}
	default:

		return src;
	}
	return dst;
}