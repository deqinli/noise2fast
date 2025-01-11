#pragma once
#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/nn/functional/activation.h>
#include <torch/nn/functional/loss.h>

struct Net : torch::nn::Module
{
    // 构造函数
    Net() :
        conv1(torch::nn::Conv2dOptions(1, 64, 3).stride(1).padding(1)),
        conv2(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
        conv3(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
        conv4(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
        conv5(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
        conv6(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
        conv7(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
        conv8(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
        conv9(torch::nn::Conv2dOptions(64, 1, 1))

    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("conv5", conv5);
        register_module("conv6", conv6);
        register_module("conv7", conv7);
        register_module("conv8", conv8);
        register_module("conv9", conv9);


    }
    // 成员函数：前向传播
    torch::Tensor forward(torch::Tensor x) {
        x = conv1(x);
        x = torch::relu(x);
        x = conv2(x);
        x = torch::relu(x);
        x = conv3(x);
        x = torch::relu(x);
        x = conv4(x);
        x = torch::relu(x);
        x = conv5(x);
        x = torch::relu(x);
        x = conv6(x);
        x = torch::relu(x);
        x = conv7(x);
        x = torch::relu(x);
        x = conv8(x);
        x = torch::relu(x);
        x = conv9(x);
        x = torch::sigmoid(x);

        return x;
    }


    // 模块成员
    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Conv2d conv3;
    torch::nn::Conv2d conv4;
    torch::nn::Conv2d conv5;
    torch::nn::Conv2d conv6;
    torch::nn::Conv2d conv7;
    torch::nn::Conv2d conv8;
    torch::nn::Conv2d conv9;
};