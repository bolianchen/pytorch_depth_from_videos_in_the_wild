import cv2
import numpy as np

class DynamicObjectDetector(object):
    limit_edge_corner = 5
    limit_of_check = 2120
    limit_dis_epi = 1
    dx = (-1, 0, 1, -1, 0, 1, -1, 0, 1)
    dy = (-1, -1, -1, 0, 0, 0, 1, 1, 1)

    def __call__(self, img_gray_pre, img_gray):
        prepoint = cv2.goodFeaturesToTrack(
            img_gray_pre, 1000, 0.01, 8, blockSize=3, useHarrisDetector=True, k=0.04
        )
        prepoint = cv2.cornerSubPix(
            img_gray_pre, prepoint, (10, 10), (-1, -1), 
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )
        nextpoint, state, err = cv2.calcOpticalFlowPyrLK(
            img_gray_pre, img_gray, prepoint, None, winSize=(22, 22), maxLevel=5, 
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
        )
        state = state.squeeze(-1)

        F_prepoint = []
        F_nextpoint = []
        for i in range(len(state)):
            if state[i] != 0:
                x1, y1 = prepoint[i, 0]
                x2, y2 = nextpoint[i, 0]
                if (x1 < self.limit_edge_corner or x1 >= img_gray.shape[1] - self.limit_edge_corner or
                    x2 < self.limit_edge_corner or x2 >= img_gray.shape[1] - self.limit_edge_corner or
                    y1 < self.limit_edge_corner or y1 >= img_gray.shape[0] - self.limit_edge_corner or
                    x2 < self.limit_edge_corner or y2 >= img_gray.shape[0] - self.limit_edge_corner):
                    state[i] = 0
                    continue
                sum_check = 0
                for j in range(9):
                    sum_check += abs(
                        int(img_gray_pre[int(y1+self.dy[j]), int(x1+self.dx[j])]) - \
                        int(img_gray[int(y2+self.dy[j]), int(x2+self.dx[j])])
                    )
                    if sum_check > self.limit_of_check:
                        state[i] = 0
                    if state[i]:
                        F_prepoint.append(prepoint[i])
                        F_nextpoint.append(nextpoint[i])
        F_prepoint = np.stack(F_prepoint)
        F_nextpoint = np.stack(F_nextpoint)

        F, mask = cv2.findFundamentalMat(F_prepoint, F_nextpoint, ransacReprojThreshold=0.1, confidence=0.99)
        F_prepoint_extend = np.concatenate([F_prepoint.squeeze(1), 
                                            np.ones((F_prepoint.shape[0], 1))], axis=1) # (N, 3)
        F_nextpoint_extend = np.concatenate([F_nextpoint.squeeze(1), 
                                             np.ones((F_nextpoint.shape[0], 1))], axis=1) # (N, 3)
        coeff = F @ F_prepoint_extend.T # (3, N)
        dd = np.sum(coeff * F_nextpoint_extend.T, axis=0) / np.sqrt(coeff[0]**2 + coeff[1]**2) # (N,)
        F_prepoint = F_prepoint[(dd <= 0.1) * (mask[:, 0] != 0)]
        F_nextpoint = F_nextpoint[(dd <= 0.1) * (mask[:, 0] != 0)]
        
        F, mask = cv2.findFundamentalMat(F_prepoint, F_nextpoint, ransacReprojThreshold=0.1, confidence=0.99)

        prepoint_extend = np.concatenate([prepoint.squeeze(1), 
                                          np.ones((prepoint.shape[0], 1))], axis=1) # (N, 3)
        nextpoint_extend = np.concatenate([nextpoint.squeeze(1), 
                                           np.ones((nextpoint.shape[0], 1))], axis=1) # (N, 3)
        coeff = F @ prepoint_extend.T # (3, N)
        dd = np.sum(coeff * nextpoint_extend.T, axis=0) / np.sqrt(coeff[0]**2 + coeff[1]**2) # (N,)
        
        outliers = (state != 0) * (dd > self.limit_dis_epi)
        outlier_ratio = outliers.sum() / nextpoint.shape[0]
        dynamic_points = nextpoint[outliers, 0]

        return dynamic_points, outlier_ratio

class SimpleQueue:
    def __init__(self, size):
        self.container = [0] * size
        self.size = size
        self.head = 0
        self.weight = 0
    
    def add(self, value):
        if self.full():
            assert False, "Container is full"
        self.container[self.head] = value
        self.head += 1
        if (self.head >= self.size):
            self.head -= self.size
        self.weight += 1

    def pop(self):
        if self.empty():
            assert False, "Container is empty"
        idx = self.head - self.weight
        idx = idx if idx >= 0 else idx + self.size
        self.weight -= 1
        item = self.container[idx]
        self.container[idx] = None
        return item
    
    def get(self, idx):
        if (idx <= 0):
            assert False, "Index should be a positive integer"
        if self.empty():
            assert False, "Container is empty"
        if (idx > self.weight):
            assert False, "Can't get value"
        idx = self.head - idx
        idx = idx if idx >= 0 else idx + self.size
        return self.container[idx]

    def length(self):
        return self.weight

    def empty(self):
        return self.weight == 0

    def full(self):
        return self.weight == self.size

    def average(self):
        """
        This function is specifically used for calculating average of ECR ratio
        """
        if self.empty():
            assert False, "Container is empty"
        value = 0.0
        for i in range(self.head - self.weight, self.head):
            if i < 0:
                i += self.size
            value += self.container[i]
        return value / float(self.weight)


def dilation(img, dilation_elem, dilation_size):
    if dilation_elem == 0:
        dilation_shape = cv2.MORPH_RECT
    elif dilation_elem == 1:
        dilation_shape = cv2.MORPH_CROSS
    elif dilation_elem == 2:
        dilation_shape = cv2.MORPH_ELLIPSE
    else:
        assert False, "incorrect input for dilation_elem"

    element = cv2.getStructuringElement(dilation_shape, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                       (dilation_size, dilation_size))
    dilatation_dst = cv2.dilate(img, element)
    return dilatation_dst

def canny(img):
    ratio = 5
    low_threshold = 20
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray, (7, 7), 3, 0, borderType=cv2.BORDER_DEFAULT)
    detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, 3)
    return detected_edges

def edge_change_ratio(prev_src, curr_src):
    prev_edge = canny(prev_src)
    prev_dilate = dilation(prev_edge, 2, 1)

    curr_edge = canny(curr_src)
    curr_dilate = dilation(curr_edge, 2, 1)

    curr_inverted = (255 - curr_dilate)
    prev_inverted = (255 - prev_dilate)

    #cv2.imshow('VIA Preprocess', prev_edge)
    #cv2.waitKey(30)
    #cv2.imshow('VIA Preprocess', prev_dilate)
    #cv2.waitKey(30)

    curr_s = np.sum(curr_edge)
    prev_s = np.sum(prev_edge)
    x_in = np.sum((curr_edge & prev_inverted))
    x_out = np.sum((prev_edge & curr_inverted))

    return max(x_in / (curr_s + 1e-13), x_out / (prev_s + 1e-13))

def postprocess(static_scene, num_of_frames=30):
    """
    This function cut off "num_of_frames" frames before and after each group of continuous static scenes
    """
    pulling_off_scene = []
    starting_off_scene = []
    queue = SimpleQueue(3)

    # 頭尾補零
    queue.add(0)
    static_scene.append(0)

    for idx in static_scene:
        queue.add(idx)
        if queue.length() < 3:
            continue

        first = queue.get(3)
        second = queue.get(2)
        third = queue.get(1)

        if second - 1 != first:
            for i in range(1, num_of_frames + 6):
                pulling_off_scene.append(second - i)

        if second + 1 != third:
            for i in range(1, num_of_frames + 1):
                starting_off_scene.append(second + i)

        queue.pop()

    static_scene.extend(pulling_off_scene)
    static_scene.extend(starting_off_scene)
    return static_scene


def test(filepath):
    fps = 10

    thresh_hold = [0.24, 0.26, 0.28, 0.3, 0.32, 0.34]
    for file in filepath:
        results = []
        for thresh in thresh_hold:
            print("Now processing file: {}, threshold: {}".format(file, thresh))

            vidcap = cv2.VideoCapture("good_video/" + file + ".mp4")
            
            # get the approximate number of frames
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            # get the approximate frame rate 
            raw_fps = vidcap.get(cv2.CAP_PROP_FPS)

            if fps:
                # original fps
                assert raw_fps >= fps, "the specified fps is higher than the raw video"
                # the period of saving images from the video
                period = round(raw_fps/fps)
            else:
                # save every frame
                period = 1

            if frame_count==0:
                print('frame count cannot be read')
            else:
                print(f'there are {frame_count} frames in the video')

            # ==========================
            # ======= Preprocess =======
            # ==========================
            print("Start preprocessing...")
            static_scene = []
            img_queue = SimpleQueue(3)
            ecr_queue = SimpleQueue(5)
            count = 0
            while True:
                success, image = vidcap.read()

                # repeat if video reading has not started
                if vidcap.get(cv2.CAP_PROP_POS_MSEC) == 0.0:
                    success, image = vidcap.read()

                if success:
                    if count%period == 0:
                        save_idx = count//period

                        image = image[0:720, 0:1280]

                        img_queue.add(image)
                        if img_queue.length() < 3:
                            count+=1
                            continue
                    
                        old_src = img_queue.get(3)
                        src = img_queue.get(1)
                        img_queue.pop()

                        ECR = edge_change_ratio(old_src, src)
                        ecr_queue.add(ECR)
                        if ecr_queue.length() < 5:
                            count+=1
                            continue

                        # 出現黑頻或不合理畫面無條件捨去
                        if ECR >= 0.99 or ECR <= 0.01:
                            static_scene.append(save_idx)
                            ecr_queue.pop()
                            count+=1
                            continue

                        ECR_avg = ecr_queue.average()
                        ecr_queue.pop()
                        print(ECR_avg)

                        if ECR_avg <= thresh:
                            static_scene.append(save_idx)

                    if count%500 == 0:
                        print(f'{count} raw frames have been processed')
                    
                    count+=1
                else:
                    break
            
            results.append(static_scene)

        with open("test/ECR/" + file + ".txt", 'w') as f:
            for r in results:
                for line in r:
                    f.write(str(line) + " ")
                f.write("\n")

if __name__ == "__main__":
    #filepath = "Mobile361/CamA_20210203_072021_2x2_4ADAS"
    #filepath = "test/camA_20201127_142858_2x2_4ADAS"
    filepath = ["CamA_19700123_071334_2x2_4ADAS", "CamA_19700123_071634_2x2_4ADAS", "camA_20201127_142858_2x2_4ADAS", 
                 "CamA_20201231_071716_2x2_4ADAS", "CamA_20210203_070819_2x2_4ADAS"]
    test(filepath)