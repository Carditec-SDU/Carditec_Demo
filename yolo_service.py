import _thread
from threading import Thread, Semaphore

from yolo import YOLO 
from SSD import SSD

class Yolo_thread(Thread):
    
    def __init__(self, threadID, name):
        '''

            model: 线程当前装载的模型

        '''
        
        Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.model = None


    def run(self, f_model):

        print("启动线程："+self.name)

        if f_model == "yolo":
            print("初始化YOLO模型")
            self.model = YOLO()
            self.name = "yolo"
        elif f_model == "SSD":
            print("初始化SSD模型")
            self.model = SSD()
            self.name = "SSD"

    def destroy(self):

        del self.model
        return self.name

    def detect_image(self, frame):

        return self.model.detect_image(frame)



