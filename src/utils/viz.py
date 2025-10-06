import cv2


def draw_box(img, box, color=(0,255,0), text=None):
    x1,y1,x2,y2 = map(int, [box.x1, box.y1, box.x2, box.y2])
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    if text:
        cv2.putText(img, text, (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img