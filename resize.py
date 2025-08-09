import cv2

def resize_image(path, max_size=512):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(path, img)
