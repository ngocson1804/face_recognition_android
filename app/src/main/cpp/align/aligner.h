//
// Created by SonNN27.NXT on 26/03/2023.
//

#ifndef FACE_RECOGNITION_ALIGNER_H
#define FACE_RECOGNITION_ALIGNER_H
#include "opencv2/core.hpp"

namespace mirror {
    class Aligner {
    public:
        Aligner();
        ~Aligner();

        int AlignFace(const cv::Mat & img_src,
                      cv::Point2f* keypoints, cv::Mat * face_aligned);

    private:
        class Impl;
        Impl* impl_;
    };

}
#endif //FACE_RECOGNITION_ALIGNER_H
