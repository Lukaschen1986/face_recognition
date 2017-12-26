# webcam
# Get a reference to webcam #0 (the default one)
img = fr.load_image_file("IMG_2926.jpg")
wjd_img = fr.load_image_file("吴荆东.jpg") # 载入需要识别的人脸图片
face_known = fr.face_encodings(wjd_img)[0] # 编码

face_locs = fr.face_locations(img, model="hog") # 检测视频中的人脸
face_objs = fr.face_encodings(img, known_face_locations=face_locs) # 编码

face_names = [] # 储存识别到的人脸，未识别的命名为"Unknown"
for face_obj in face_objs:
    # face_obj = face_objs[0]
    match = fr.compare_faces([face_known], face_obj, tolerance=0.3)
    if match[0]:
        name = "wjd"
    else:
        name = "Unknown"
    face_names.append(name)
    
for (row_up, col_right, row_down, col_left), name in zip(face_locs, face_names):
    cv2.rectangle(img, (col_left, row_up), (col_right, row_down), (0, 0, 255), 2) # Draw a box around the face
    cv2.rectangle(img, (col_left, row_down-15), (col_right, row_down), (0, 0, 255), cv2.FILLED) # Draw a label with a name below the face
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, name, (col_left+6, row_down-6), font, 0.5, (255, 255, 255), 1)

Image.fromarray(img)

cv2.imshow('Video', img)    
