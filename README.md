# face_recognition_mnn
## 1. Download models
### 1.1 Face detection model from [here](https://drive.google.com/file/d/1AdXZRfjU8nwDmK7ly7eQAFjSDwo4lh4B/view?usp=sharing) and copy it to app/src/main/assets
### 1.2 Face vectorization model from [here](https://drive.google.com/file/d/1KaAXd-QCiEJs3bVycZwqvV6e9Fs5SxeK/view?usp=sharing) and copy it to app/src/main/assets
## 2. Build app

# 3. if you need to build lib for MNN
## 3.1 change directory to MNN folder
## 3.2 edit ciscripts/Android/64.sh file
## 3.3 build so file
#### Download android-ndk-r16 from [here](https://github.com/android/ndk/wiki/Unsupported-Downloads)
```bash
export ANDROID_NDK="/Users/sonnn27.nxt/WorkSpace/personal/android/android-ndk-r16b"
./ciscripts/Android/64.sh
```