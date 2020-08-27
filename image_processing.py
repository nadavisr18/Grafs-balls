import cv2
import time
import numpy as np


class Processor:
    def __init__(self):
        self.vid = cv2.VideoCapture("video.mp4")
        self.downscale_factor = 50
        self.ball_radius = 100
        self.frame = 0
        self.color = np.array([116, 205, 251])

    def __del__(self):
        self.vid.release()
        cv2.destroyAllWindows()

    def display_video(self):
        while True:
            init_time = time.time()
            ret, frame = self.vid.read()
            if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
                break
            frame = self.get_ball(frame)
            cv2.imshow('frame', frame)
            time.sleep(0.02)
            print(1/(time.time()-init_time))

    def get_ball(self, frame):
        small = cv2.resize(frame, (0, 0), fx=1 / self.downscale_factor, fy=1 / self.downscale_factor)

        only_ball = np.apply_along_axis(self.threshold, 2, small)
        ball_indexes = np.where(only_ball == True)

        ball_x = np.average(ball_indexes[0])
        ball_y = np.average(ball_indexes[1])

        from_x, to_x, from_y, to_y = self.get_ball_range(ball_x, ball_y, frame)
        ball_in_frame = frame[from_x:to_x, from_y:to_y]
        # frame[from_x:to_x, from_y:to_y] = self.detect_ball(ball_in_frame)
        # return frame
        if 0 in ball_in_frame.shape:
            ball_in_frame = self.handle_no_ball()
        return self.detect_ball(ball_in_frame)

    def get_ball_range(self, x, y, frame):
        if str(x) == "nan":
            return 0, 1, 0, 1
        upscaled_x = int(np.floor(x * self.downscale_factor))+ self.downscale_factor//2
        upscaled_y = int(np.floor(y * self.downscale_factor))+ self.downscale_factor//2

        min_x = self.walk(frame, upscaled_x, upscaled_y, [-1,0])[0] - 30
        max_x = self.walk(frame, upscaled_x, upscaled_y, [ 1,0])[0] + 30
        min_y = self.walk(frame, upscaled_x, upscaled_y, [0,-1])[1] - 30
        max_y = self.walk(frame, upscaled_x, upscaled_y, [0, 1])[1] + 30

        return int(min_x), int(max_x), int(min_y), int(max_y)

    def walk(self, frame, x, y,direction):
        while True:
            x += direction[0]
            y += direction[1]
            if x < frame.shape[0] and y < frame.shape[1]:
                pixel = frame[x,y]
                if not self.threshold(pixel,5000):
                    return [x,y]
            else:
                return [0,0]

    def threshold(self,x,threshold = 1600):
        dist = self.get_distance(x,self.color)
        if dist < threshold:
            return True
        else:
            return False

    def detect_ball(self, frame):
        x = len(frame) // 2
        y = len(frame) // 2
        rightx,righty = self.detect_edges(frame, x, y, [3, 0])
        leftx,lefty = self.detect_edges(frame, x, y, [-3, 0])
        upx,upy = self.detect_edges(frame, x, y, [0, -3])
        downx,downy = self.detect_edges(frame, x, y, [0, 3])

        frame[rightx - 3:rightx + 3, righty - 3:righty + 3] = [0, 0, 255]
        frame[leftx - 3:leftx + 3, lefty - 3:lefty + 3] = [0, 0, 255]
        frame[upx - 3:upx + 3, upy - 3:upy + 3] = [0, 0, 255]
        frame[downx - 3:downx + 3, downy - 3:downy + 3] = [0, 0, 255]

        return frame

    def detect_edges(self,frame,x,y,direction):
        while True:
            x += direction[0]*5
            y += direction[1]*5
            if 0 < x < frame.shape[0] and 0 < y < frame.shape[1]:
                new_color = frame[x,y]
                avg_color = np.average(frame,axis = (0,1))
                dist = self.get_distance(new_color,avg_color)
                ball = self.threshold(new_color,dist)
                if not ball:
                    return [x,y]
            else:
                return [0,0]
    @staticmethod
    def handle_no_ball():
        return np.ones((2,2,3))

    @staticmethod
    def get_distance(x1,x2):
        return np.sum((x1 - x2)**2)