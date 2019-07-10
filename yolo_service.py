import _thread
from threading import Thread, Semaphore

import tensorflow as tf
import keras

from yolo import YOLO 
from SSD import SSD

class Yolo_thread(Thread):
    
    def __init__(self):

        '''

            model: 线程当前装载的模型

        '''
        
        Thread.__init__(self)
        self.name = None
        self.model = None


    def run(self, f_model="SSD"):

        '''
            调用 .start 方法时，采用多线程机制加载模型
            保证程序开始时加载模型的速度

        '''

        # print("启动yolo线程")
        # print("初始化YOLO模型")

        if f_model=="SSD":

            # TF版的模型用tensorflow的方式调用clear_session

            tf.keras.backend.clear_session()
            self.model = SSD()
            self.name = "SSD"
        elif f_model=="yolo":

            # Keras版的模型用 keras.backend方式调用clear_session

            keras.backend.clear_session()   
            self.model = YOLO()
            self.name = "yolo"

        # print("YOLO模型初始化完成")

    def destroy(self):

        del self.model
        return self.name

    def detect_image(self, frame):

        return self.model.detect_image(frame)



