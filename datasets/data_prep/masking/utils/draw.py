import numpy as np
import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, possibilities, identities=None, offset=(0,0)):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.circle(img, (int((x1 + x2)/ 2), int((y1 + y2) / 2)), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,str(round(possibilities[i], 2)),(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)

    return img


def draw_maskes(img, masks, possibilities, identities, dynamic_points=None, prepoint=None, currpoint=None):
    for i in range(len(identities)):
        id = int(identities[i]) if identities is not None else 0    
        color = compute_color_for_labels(id)

        if possibilities[i] < 0.3:
            img[masks[i] == 1] = color

    if prepoint is not None and currpoint is not None:
        for prev_pts, pts in zip(prepoint, currpoint):
            x1, y1 = prev_pts.ravel()
            x2, y2 = pts.ravel()
            img = cv2.line(img, (x2, y2), (x1, y1), color=(0, 255, 0), thickness=1)
            img = cv2.circle(img, (x2, y2), radius=2, color=(0, 0, 255), thickness=-1)

    if dynamic_points is not None:
        for pts in dynamic_points:
            x, y = pts.ravel()
            img = cv2.circle(img, (x, y), radius=2, color=(255, 0, 0), thickness=-1)

    return img


if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
