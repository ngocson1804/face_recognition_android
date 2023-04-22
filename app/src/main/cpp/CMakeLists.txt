# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.22.1)

project("face_recognition_mnn")

#include_directories(/Users/sonnn27.nxt/MNN/include/)
#include_directories(/Users/sonnn27.nxt/Downloads/opencv-4.1.0-android/OpenCV-android-sdk/sdk/native/jni/include)
include_directories(include)
add_library(lib_opencv SHARED IMPORTED)
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libopencv_java4.so)
#set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION /Users/sonnn27.nxt/Downloads/opencv-4.1.0-android/OpenCV-android-sdk/sdk/native/libs/armeabi-v7a/libopencv_java4.so)

add_library(MNN SHARED IMPORTED)
add_library(MNN_CL SHARED IMPORTED)

#set_target_properties(MNN
#        PROPERTIES IMPORTED_LOCATION
#        /Users/sonnn27.nxt/MNN/project/android/build_64/libMNN.so)
#set_target_properties(MNN_CL
#        PROPERTIES IMPORTED_LOCATION
#        /Users/sonnn27.nxt/MNN/project/android/build_64/libMNN_CL.so)
set_target_properties(MNN
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libMNN.so)
set_target_properties(MNN_CL
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libMNN_CL.so)

add_library( # Sets the name of the library.
        face_recognition_mnn
        SHARED
        native-lib.cpp
        face_detection/SCRFD.cpp
        face_extraction/extract.cpp
        align/aligner.cpp
        utils/utils.cpp
        )

find_library(
        log-lib
        log)

target_link_libraries( # Specifies the target library.
        face_recognition_mnn
        MNN
        MNN_CL
        ${log-lib}
        lib_opencv
        )