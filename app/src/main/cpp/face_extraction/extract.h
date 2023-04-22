//
// Created by SonNN27.NXT on 13/04/2023.
//

#ifndef FACE_RECOGNITION_MNN_EXTRACT_H
#define FACE_RECOGNITION_MNN_EXTRACT_H


#include <memory>
#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>
#include <math.h>
#include <opencv2/core.hpp>

#define FACE_VECTORIZATION_OUTPUT_NODE "683"
#define FACE_VECTORIZATION_MODEL_NAME "glint360k_cosface.mnn"
#define INPUT_W 112
#define INPUT_H 112
#define FEAT_DIM 512

class FaceExtraction
{
private:
    std::shared_ptr<MNN::Interpreter> m_interpreter;
    MNN::Session *m_session = nullptr;
    MNN::Tensor *m_tensor = nullptr;
    cv::Size_<int> inputSize_ = cv::Size(INPUT_W, INPUT_H);
    const char* outputNode;
    const float mean[3] = { 0.5f * 255.f, 0.5f * 255.f, 0.5f * 255.f };
    const float norm[3] = { 1 / 0.5f / 255.f, 1 / 0.5f / 255.f, 1 / 0.5f / 255.f };
    int feat_dim = FEAT_DIM;

public:
    FaceExtraction(const std::string& modelPath, const char* outputNode, int numThreads);
    ~FaceExtraction();
    bool extract(const cv::Mat &img_face, std::vector<float>* feat);
};


#endif //FACE_RECOGNITION_MNN_EXTRACT_H
