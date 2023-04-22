//
// Created by SonNN27.NXT on 13/04/2023.
//

#include "extract.h"

FaceExtraction::FaceExtraction(const std::string& modelPath,
                               const char *outputNode,
                               int numThreads=1) {
    this->outputNode = outputNode;
    m_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(modelPath.c_str()));
    MNN::ScheduleConfig config;
    config.numThread = numThreads;
    config.type = MNN_FORWARD_CPU;
    MNN::BackendConfig backend_config;
    backend_config.memory    = MNN::BackendConfig::Memory_Normal;
    backend_config.power     = MNN::BackendConfig::Power_Normal;
    backend_config.precision = MNN::BackendConfig::Precision_Normal;
    config.backendConfig = &backend_config;

    m_session = m_interpreter->createSession(config);
    m_tensor = m_interpreter->getSessionInput(m_session, nullptr);
}

FaceExtraction::~FaceExtraction() {
    m_interpreter->releaseModel();
    m_interpreter->releaseSession(m_session);
}

bool FaceExtraction::extract(const cv::Mat &img_face, std::vector<float> *feat) {
    if (img_face.empty())
    {
        std::cout << "image is empty ,please check!" << std::endl;
        return false;
    }
    cv::Mat resized_image;
    cv::resize(img_face, resized_image, inputSize_);

    std::shared_ptr<MNN::CV::ImageProcess> pretreat(
    MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, mean, 3,
                                  norm, 3));
    pretreat->convert(resized_image.data, inputSize_.width,
                      inputSize_.height, resized_image.step[0], m_tensor);

    m_interpreter->resizeTensor(m_tensor, {1, 3, inputSize_.height, inputSize_.width});
//        std::cout << m_tensor->dimensions();
    m_interpreter->resizeSession(m_session);
    m_interpreter->runSession(m_session);
    const MNN::Tensor *outputTensor = m_interpreter->getSessionOutput(m_session, outputNode);

    MNN::Tensor outputTensorHost(outputTensor, outputTensor->getDimensionType());
    outputTensor->copyToHostTensor(&outputTensorHost);
    for(int i=0; i<feat_dim; i++){
        feat->push_back(outputTensor->host<float>()[i]);
    }
    return true;
}
