//
// Created by SonNN27.NXT on 13/04/2023.
//

#ifndef FACE_RECOGNITION_MNN_UTILS_H
#define FACE_RECOGNITION_MNN_UTILS_H
#include <iostream>
#include <cmath>
#include <vector>

#define kFaceFeatureDim 512
float CalculateSimilarity(const std::vector<float>&feat1, const std::vector<float>& feat2);
#endif //FACE_RECOGNITION_MNN_UTILS_H
