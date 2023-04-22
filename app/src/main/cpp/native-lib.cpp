#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "face_detection/SCRFD.h"
#include "face_extraction/extract.h"
#include "align/aligner.h"
#include <android/log.h>
#include "utils/utils.h"


#define  LOG_TAG    "native-lib"
#define  ALOG(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)

bool faceDetectionInitOk = false;
bool faceVectorizationInitOk = false;
bool faceAlignmentInitOk = false;

static SCRFD *faceDetector;
static FaceExtraction *faceExtractor;
static mirror::Aligner *faceAligner;
static std::string img_path;

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */,
        jstring model_path) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_faceDetectionInit(
        JNIEnv* env,
        jobject /* this */,
        jstring modelPath) {
    if (faceDetectionInitOk) {
        ALOG("face detector already initialized");
        return true;
    }
    jboolean tRet = false;
    if (nullptr == modelPath) {
        ALOG("model dir is empty");
        return tRet;
    }

    std::string rootModelPath = env->GetStringUTFChars(modelPath, nullptr);
    if (rootModelPath.empty()) {
        ALOG("model dir is empty");
        return tRet;
    }
    const std::string&  tModelDir = rootModelPath;
    std::string faceDetectionModelPath = tModelDir + FACE_DETECTION_MODEL;
    faceDetector = new SCRFD();
    faceDetector->load_heads(scrfd_2_5g_bnkps_head_info);
    faceDetector->reload(faceDetectionModelPath, true,
                         FACE_DETECTION_INPUT_SIZE,
                         2, 1);
    faceDetectionInitOk = true;
    tRet = true;

    return tRet;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_myapplication_MainActivity_faceDetect(
        JNIEnv* env,
        jobject /* this */,
        jbyteArray imageData,
        jint width,
        jint height) {

    if (!faceDetectionInitOk) {
        ALOG("faceDetector is not initialized!");
        return nullptr;
    }
    jbyte *imageData_ = env->GetByteArrayElements(imageData, nullptr);
    if (nullptr == imageData) {
        ALOG("img data is null");
        return nullptr;
    }
    std::vector<FaceInfo> results;
    cv::Mat inputImage;
    inputImage.create(height, width, CV_8UC3);
    for (int i = 0; i < width * height * 3; i++) {
        imageData_[i] = static_cast<unsigned char>(imageData_[i] & 0xFF);
    }
    memcpy(inputImage.data, (unsigned char *) imageData_, width * height * 3);
    faceDetector->detect(inputImage, results);
    auto num_face = static_cast<int32_t>(results.size());
    int out_size = 1 + num_face * 14; // [xmin, ymin, xmax, ymax, x1, y1, x2, y2, ...., x5, y5]
    auto *allFaceInfo = new float[out_size];
    allFaceInfo[0] = num_face;
    for (int i = 0; i < num_face; i++) {
        allFaceInfo[14 * i + 1] = results[i].x1;//left
        allFaceInfo[14 * i + 2] = results[i].y1;//top
        allFaceInfo[14 * i + 3] = results[i].x2;//right
        allFaceInfo[14 * i + 4] = results[i].y2;//bottom
        for (int j=0; j<10; j++){
            allFaceInfo[14 * i + 5 +j] = results[i].lmk[j];
        }
    }
    jfloatArray tFaceInfo = env->NewFloatArray(out_size);
    env->SetFloatArrayRegion(tFaceInfo, 0, out_size, allFaceInfo);
    env->ReleaseByteArrayElements(imageData, imageData_, 0);
    delete[] allFaceInfo;
    inputImage.release();
    return tFaceInfo;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_faceVectorizationInit(
        JNIEnv* env,
        jobject /* this */,
        jstring modelPath){
    if (faceVectorizationInitOk) {
        ALOG("Face vectorization model already initialized");
        return true;
    }
    jboolean tRet = false;
    if (nullptr == modelPath) {
        ALOG("Face vectorization model dir is empty");
        return tRet;
    }

    std::string rootModelPath = env->GetStringUTFChars(modelPath, nullptr);
    if (rootModelPath.empty()) {
        ALOG("Face vectorization model dir is empty");
        return tRet;
    }
    const std::string&  tModelDir = rootModelPath;
    std::string faceVectorizationModelPath = tModelDir + FACE_VECTORIZATION_MODEL_NAME;
    faceExtractor = new FaceExtraction(faceVectorizationModelPath, FACE_VECTORIZATION_OUTPUT_NODE, 1);
    faceVectorizationInitOk = true;
    tRet = true;

    return tRet;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_myapplication_MainActivity_faceVectorize(
        JNIEnv* env,
        jobject /* this */,
        jbyteArray imageData) {

    if (!faceDetectionInitOk) {
        ALOG("faceDetector is not initialized!");
        return nullptr;
    }
    jbyte *imageData_ = env->GetByteArrayElements(imageData, nullptr);
    if (nullptr == imageData) {
        ALOG("img data is null");
        return nullptr;
    }
    cv::Mat inputImage;
    inputImage.create(INPUT_H, INPUT_W, CV_8UC3);

    memcpy(inputImage.data, (unsigned char *) imageData_, INPUT_W * INPUT_H * 3);
    std::vector<float> out_feat;
    faceExtractor->extract(inputImage, &out_feat);

    auto num_vec = static_cast<int32_t>(out_feat.size());

    auto *output = new float[num_vec];
//    int count = 0;
//    for (int i = 0; i < num_vec; i++) {
//        output[i] = out_feat[i];
//        ALOG("feat %f, %d", out_feat[i], count);
//        count += 1;
//    }

    jfloatArray tVecInfo = env->NewFloatArray(num_vec);
    env->SetFloatArrayRegion(tVecInfo, 0, num_vec, output);
    env->ReleaseByteArrayElements(imageData, imageData_, 0);

    delete[] output;
    inputImage.release();

    return tVecInfo;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_faceAlignmentInit(
        JNIEnv* env,
        jobject /* this */
        ){
    if (faceAlignmentInitOk) {
        ALOG("face detector already initialized");
    }
    faceAligner = new mirror::Aligner();
    faceAlignmentInitOk = true;
    return true;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_myapplication_MainActivity_extractFeaturePipeline(
        JNIEnv* env,
        jobject /* this */,
        jbyteArray imageData,
        jint width,
        jint height){
    if (!faceDetectionInitOk) {
        ALOG("faceDetector is not initialized!");
        return nullptr;
    }
    if (!faceVectorizationInitOk){
        ALOG("faceVectorization is not initialized!");
        return nullptr;
    }
    if (!faceAlignmentInitOk){
        ALOG("faceAlignment is not initialized!");
        return nullptr;
    }
    jbyte *imageData_ = env->GetByteArrayElements(imageData, nullptr);
    if (nullptr == imageData) {
        ALOG("img data is null");
        return nullptr;
    }
    ALOG("Start to extract feature");

    std::vector<FaceInfo> detResults;
    cv::Mat inputImage;
    inputImage.create(height, width, CV_8UC3);
    for (int i = 0; i < width * height * 3; i++) {
        imageData_[i] = static_cast<unsigned char>(imageData_[i] & 0xFF);
    }
    memcpy(inputImage.data, (unsigned char *) imageData_, width * height * 3);
    ALOG("Detect face");
    faceDetector->detect(inputImage, detResults);

    auto numFaces = static_cast<int32_t>(detResults.size());
    int outSize = 1 + (14 + FEAT_DIM) * numFaces;
    auto *allFaceInfo = new float[outSize];
    ALOG("**********");
    ALOG("NUM FACE %d", numFaces);

    allFaceInfo[0] = numFaces;
    cv::Point2f landmark[5];
    cv::Mat alignedFace;

    for(int faceIdx=0; faceIdx < numFaces; faceIdx++){
        std::vector<float> outFeat;
        for(int lmk_idx=0; lmk_idx<5; lmk_idx++){
            landmark[lmk_idx].x = detResults[faceIdx].lmk[2*lmk_idx];
            landmark[lmk_idx].y = detResults[faceIdx].lmk[2*lmk_idx+1];
        }

        faceAligner->AlignFace(inputImage, landmark, &alignedFace);
        faceExtractor->extract(alignedFace, &outFeat);
        allFaceInfo[(14 + FEAT_DIM) * faceIdx + 1] = detResults[faceIdx].x1;
        allFaceInfo[(14 + FEAT_DIM) * faceIdx + 2] = detResults[faceIdx].y1;
        allFaceInfo[(14 + FEAT_DIM) * faceIdx + 3] = detResults[faceIdx].x2;
        allFaceInfo[(14 + FEAT_DIM) * faceIdx + 4] = detResults[faceIdx].y2;
        // add landmark to res array
        for (int j=0; j<10; j++){
            allFaceInfo[(14 + FEAT_DIM) * faceIdx + 5 +j] = detResults[faceIdx].lmk[j];
        }
        // add feat to res array
        for(int featIdx=0; featIdx<FEAT_DIM; featIdx++){
            allFaceInfo[(14 + FEAT_DIM) * faceIdx + 15 + featIdx] = outFeat[featIdx];
        }
//        time_copy = ((double) cv::getTickCount() - time_copy) / cv::getTickFrequency();
//        ALOG("Copy tensor time：%f", time_copy);
        std::vector<float>().swap(outFeat);
    }
    inputImage.release();
    alignedFace.release();
    jfloatArray tFaceInfo = env->NewFloatArray(outSize);
    env->SetFloatArrayRegion(tFaceInfo, 0, outSize, allFaceInfo);
    env->ReleaseByteArrayElements(imageData, imageData_, 0);
    delete[] allFaceInfo;

    return tFaceInfo;
}

extern "C" JNIEXPORT jfloat JNICALL
Java_com_example_myapplication_MainActivity_calculateSimilarity(
        JNIEnv* env,
        jobject /* this */,
        jfloatArray feat1,
        jfloatArray feat2){
    jfloat *feat1_ = env->GetFloatArrayElements(feat1, nullptr);
    jfloat *feat2_ = env->GetFloatArrayElements(feat2, nullptr);
    float inner_product = 0.0f;
    float feat_norm1 = 0.0f;
    float feat_norm2 = 0.0f;
    for(int i=0; i <FEAT_DIM; i++){
        inner_product += feat1_[i] * feat2_[i];
        feat_norm1 += feat1_[i] * feat1_[i];
        feat_norm2 += feat2_[i] * feat2_[i];
    }
    return inner_product / sqrt(feat_norm1) / sqrt(feat_norm2);
}
