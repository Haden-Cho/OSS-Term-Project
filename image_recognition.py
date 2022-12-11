import dlib # face detection + face recognition 작업
import cv2 # 이미지 작업
# 이하 기본 라이브러리들
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

# 얼굴 찾는 함수
def find_faces(img):
    dets = detector(img, 1) # 얼굴 찾은 결과물 넣는 변수

    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    
    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int) # 얼굴에서 68개의 점 구하기
    for k, d in enumerate(dets): # 얼굴의 개수 만큼 loop
        rect = ((d.left(), d.top()), (d.right(), d.bottom())) # 얼굴 상하 좌우 좌표 넣기
        rects.append(rect) # 얼굴별로 append

        # 랜드마크 찾기 (눈, 코, 입 어디있는지 찾기)
        shape = sp(img, d)
        
        # convert dlib shape to numpy array
        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)
        
    return rects, shapes, shapes_np

# 얼굴을 인코드 하는 함수
def encode_faces(img, shapes):
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape) # 얼굴 인코딩
        face_descriptors.append(np.array(face_descriptor))

    return np.array(face_descriptors)

# 이미지 경로 지정
img_paths = {
    'Mouse': 'img/yoo.jpg'
}

# 연산 결과 저장할 변수
descs = {
    'Mouse': None
}

# 이미지 수 만큼 loop
for name, img_path in img_paths.items():
    img_bgr = cv2.imread(img_path) # openCV 이용해 이미지 불러오기
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # bgr 이미지를 rgb 형태로 변환

    _, img_shapes, _ = find_faces(img_rgb) # 미리 선언한 함수로 얼굴 찾아서 landmark들 받아오기
    descs[name] = encode_faces(img_rgb, img_shapes)[0] # 인코드 결과 각 사람의 이름에 맞게 저장

np.save('img/descs.npy', descs) # 이미지 저장해주기

# 이미지 인코딩 작업
img_bgr = cv2.imread('img/surinam.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

rects, shapes, _ = find_faces(img_rgb)
descriptors = encode_faces(img_rgb, shapes)

fig, ax = plt.subplots(1, figsize=(20, 20))
ax.imshow(img_rgb)

for i, desc in enumerate(descriptors):
    
    found = False
    for name, saved_desc in descs.items():
        # parameter A와 B 벡터 사이의 유클리드 거리 측정 함수
        dist = np.linalg.norm([desc] - saved_desc, axis=1) 

        # 유클리드 거리가 0.4보다 작다면 쥐새끼 찾은 것으로 인식
        if dist < 0.4:
            found = True

            # 이름 쓰는 함수
            text = ax.text(rects[i][0][0], rects[i][0][1], name,
                    color='r', fontsize=40, fontweight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=10, foreground='white'), path_effects.Normal()])
            
            # 얼굴 부분에 사각형 그리는 함수
            rect = patches.Rectangle(rects[i][0],
                                 rects[i][1][1] - rects[i][0][1],
                                 rects[i][1][0] - rects[i][0][0],
                                 linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            break
    
    # 쥐새끼 아니면 시민
    if not found:
        ax.text(rects[i][0][0], rects[i][0][1], 'human',
                color='w', fontsize=20, fontweight='bold')
        rect = patches.Rectangle(rects[i][0],
                             rects[i][1][1] - rects[i][0][1],
                             rects[i][1][0] - rects[i][0][0],
                             linewidth=2, edgecolor='w', facecolor='none')
        ax.add_patch(rect)

plt.axis('off')
plt.savefig('result/output.png')
plt.show()