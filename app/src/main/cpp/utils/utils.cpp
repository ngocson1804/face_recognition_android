//
// Created by SonNN27.NXT on 13/04/2023.
//
#include <iostream>
#include <cmath>
#include <vector>

#define kFaceFeatureDim 512

float CalculateSimilarity(const std::vector<float>&feat1, const std::vector<float>& feat2) {
    if (feat1.size() != feat2.size()) {
        std::cout << "feature size not match." << std::endl;
        return 10003;
    }
    float inner_product = 0.0f;
    float feat_norm1 = 0.0f;
    float feat_norm2 = 0.0f;

    for(int i = 0; i < kFaceFeatureDim; ++i) {
        inner_product += feat1[i] * feat2[i];
        feat_norm1 += feat1[i] * feat1[i];
        feat_norm2 += feat2[i] * feat2[i];
    }
    return inner_product / sqrt(feat_norm1) / sqrt(feat_norm2);
}
