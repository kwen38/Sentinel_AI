import cv2
from batch_face import (
    RetinaFace,
    drawLandmark_multiple,
    LandmarkPredictor,
    SixDRep
)
import os
import numpy as np

def draw_landmarks(face, landmark, pose, img):
    head_pose = SixDRep(gpu_id=-1) # change from 0 to gpu_id=-1 for Mac users
    print(face[1])
    img = drawLandmark_multiple(img, face[0], landmark)
    head_pose.plot_pose_cube(img, face[0], pose['pitch'], pose['yaw'], pose['roll'])
    return img

def crop_eyes(face, img):
    img = cv2.UMat(img).get()
    features = face[1]
    eyes = features[0], features[1]
    # List to store cropped eyes
    cropped_eyes = []
    for eye in eyes:
        # Round the coordinates to integers, as pixel indices must be integers
        print(eye)
        x = int(eye[0])
        y = int(eye[1])

        # Define the width and height of the rectangle to crop
        width = 50  # Example width
        height = 30  # Example height

        # Compute the top-left corner of the rectangle
        x_min = max(x - width // 2, 0)
        y_min = max(y - height // 2, 0)

        # Compute the bottom-right corner of the rectangle
        x_max = min(x + width // 2, img.shape[1])
        y_max = min(y + height // 2, img.shape[0])

        # Crop the image using array slicing (OpenCV images are NumPy arrays)
        cropped_eye = img[y_min:y_max, x_min:x_max]
        
        cropped_eyes.append(cropped_eye)
    # Concatenate cropped eye images horizontally
    if len(cropped_eyes) == 2:
        concatenated_eyes = cv2.hconcat([cropped_eyes[0], cropped_eyes[1]])
    else:
        # If there are not exactly two eyes detected returm Npne
        concatenated_eyes = None

    return concatenated_eyes

def parse_roi_box_from_bbox(bbox, img_shape):
    h, w = img_shape
    left, top, right, bottom = bbox
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)

    roi_box = np.zeros((4))
    roi_box[[0, 2]] = clip(center_x, size, w)
    roi_box[[1, 3]] = clip(center_y, size, h)
    return roi_box

def crop_img(img, roi_box):
    h, w = img.shape[:2]
    print(roi_box)
    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]

    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res

def clip(center, size, max_size):
    end = center + size / 2
    if end > max_size:
        end = max_size
    start = end - size
    if start < 0:
        start = 0
        end = start + size
    return start, end

def get_input_data(image):
    result_list = []

    predictor = LandmarkPredictor(gpu_id=-1)
    detector = RetinaFace(gpu_id=-1)
    head_pose = SixDRep(gpu_id=-1)

    # Process a single image
    img = image
    all_faces = detector.pseudo_batch_detect([img], cv=True, threshold=0.9)
    all_landmarks = predictor(all_faces, [img], from_fd=True)
    all_poses = head_pose(all_faces, [img])

    for faces, landmarks, pose in zip(all_faces, all_landmarks, all_poses):
        for face, landmark, pose_data in zip(faces, landmarks, pose):
            bbox = parse_roi_box_from_bbox(face[0], img.shape[:2])
            concatenated_eyes = crop_eyes(face, img)
            pitch, yaw, roll = pose_data['pitch'], pose_data['yaw'], pose_data['roll']
            img_with_landmarks = draw_landmarks(face, landmark, pose_data, img.copy())

            result = {
                'p_pred_deg': [pitch],
                'y_pred_deg': [yaw],
                'r_pred_deg': [roll],
                'image': concatenated_eyes,
                'box': bbox.tolist(),
                'landmarks': [l.tolist() for l in landmark]
            }

            result_list.append(result)

    return result_list

