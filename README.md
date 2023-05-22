# face_recognition_mnn
## 1. Download models
1.1 Face detection model from [here](https://drive.google.com/file/d/1AdXZRfjU8nwDmK7ly7eQAFjSDwo4lh4B/view?usp=sharing) and copy it to app/src/main/assets <br />
1.2 Face vectorization model from [here](https://drive.google.com/file/d/1KaAXd-QCiEJs3bVycZwqvV6e9Fs5SxeK/view?usp=sharing) and copy it to app/src/main/assets <br />
1.3 Threshold: for r50-cosface model: if score >=  0.42199, it is the same person
## 2. Build app

## 3. if you need to build lib for MNN
3.1 Change directory to MNN folder <br />
3.2 Edit ciscripts/Android/64.sh file <br />
3.3 Build so file <br />
Download android-ndk-r16 from [here](https://github.com/android/ndk/wiki/Unsupported-Downloads)
```bash
export ANDROID_NDK="/Users/sonnn27.nxt/WorkSpace/personal/android/android-ndk-r16b"
./ciscripts/Android/64.sh
```

## 4. if you need to get different version of opencv-android-sdk
Download android-ndk-r16 from [here](https://github.com/opencv/opencv/releases).Then copy  .so file in folder OpenCV-android-sdk/sdk/native/libs to jniLibs