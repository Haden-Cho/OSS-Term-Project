## 프로젝트 개요
opencv, dilb,face-recognition을 활용하여 영상,사진,웹캠에서 얼굴인식이 가능하게 만들었다.


얼굴인식을 영화 수리남에 활용하여 극중 쥐새끼인 배우 유인석을 다른 배우들과 구분하여 Mouse로 찾게 만들었다.

## face-recognition in image
![image output.jpeg](https://github.com/standardstone/standardstone/blob/main/image%20output.jpeg)


## face-recognition in video
![video output.gif](https://github.com/standardstone/standardstone/blob/main/video%20output.gif)


## face-recognition in webcam
![screenshot output.png](https://github.com/standardstone/standardstone/blob/main/screenshot%20output.png)


## Download Models
- [shape_predictor_68_face_landmarks.dat.bz2](https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2)
- [dlib_face_recognition_resnet_model_v1.dat](https://github.com/kairess/simple_face_recognition/raw/master/models/dlib_face_recognition_resnet_model_v1.dat)

## 필요 패키지
- Python 3+
- dlib
- OpenCV
- face_reconition
- numpy
- matplotlib (for visualization)


## 실행방법

### image,video
-dlib과opencv,패키지를 pip를 사용하여 설치해준다.


-shape_predictor_68_face_landmarks.dat.bz2,dlib_face_recognition_resnet_model_v1.dat을 다운로드 받아 압축해제 후 파이썬 코드가 있는 위치로 복사해준다.


-실행시켜준다.

### webcam
-face_recognition과 opencv 패키지를 pip 사용하여 설치해준다.


-실행시켜준다.

## 참고자료 
[kairess/simple_face_recognition](https://github.com/kairess/simple_face_recognition)


[ageitgey/face_recognition](https://github.com/ageitgey/face_recognition)
