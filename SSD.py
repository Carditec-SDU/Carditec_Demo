import tensorflow as tf
import os
from utils import np_methods, visualization
import numpy as np
import math

class SSD:

    def __init__(self):

        self.net_shape = (300, 300)
        self.select_threshold = 0.5
        self.num_classes = 21
        self.rbbox_img = [0., 0., 1., 1.]
        self.nms_threshold = 0.45
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = self.init_sess()

    def init_sess(self):
        with tf.gfile.FastGFile('./model/frozen_model_car.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
            sess = tf.Session(config=self.config, graph=tf.get_default_graph())
            return sess

    def detect_image(self,img):
        ssd_anchors = self.Get_anchors(self.net_shape)
        output = self.sess.graph.get_tensor_by_name('ssd_300_vgg/final:0')
        prob = self.sess.run(output, feed_dict={"Placeholder:0": img})
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            prob, ssd_anchors, select_threshold=self.select_threshold, img_shape=self.net_shape, num_classes=self.num_classes,
            decode=True)
        rbboxes = np_methods.bboxes_clip(self.rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=self.nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(self.rbbox_img, rbboxes)
        # visualization.plt_bboxes(frame, rclasses, rscores, rbboxes)
        img = visualization.plt_bboxes_1(img, rclasses, rscores, rbboxes)
        return img



    def Get_anchors(self,net_shape):
        feat_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
        anchor_sizes = [(21.0, 45.0), (45.0, 99.0), (99.0, 153.0), (153.0, 207.0), (207.0, 261.0), (261.0, 315.0)]
        anchor_ratios = [[2, 0.5], [2, 0.5, 3, 0.3333333333333333], [2, 0.5, 3, 0.3333333333333333],
                         [2, 0.5, 3, 0.3333333333333333], [2, 0.5], [2, 0.5]]
        anchor_steps = [8, 16, 32, 64, 100, 300]
        anchor_offset = 0.5

        return self.ssd_anchors_all_layers(net_shape, feat_shapes, anchor_sizes,
                                      anchor_ratios, anchor_steps, anchor_offset, dtype=np.float32)

    def ssd_anchors_all_layers(self,img_shape,
                               layers_shape,
                               anchor_sizes,
                               anchor_ratios,
                               anchor_steps,
                               offset=0.5,
                               dtype=np.float32):
        """Compute anchor boxes for all feature layers.
        """
        layers_anchors = []
        for i, s in enumerate(layers_shape):
            anchor_bboxes = self.ssd_anchor_one_layer(img_shape, s,
                                                 anchor_sizes[i],
                                                 anchor_ratios[i],
                                                 anchor_steps[i],
                                                 offset=offset, dtype=dtype)
            layers_anchors.append(anchor_bboxes)

        return layers_anchors

    def ssd_anchor_one_layer(self,img_shape,
                             feat_shape,
                             sizes,
                             ratios,
                             step,
                             offset=0.5,
                             dtype=np.float32):

        y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
        y = (y.astype(dtype) + offset) * step / img_shape[0]
        x = (x.astype(dtype) + offset) * step / img_shape[1]

        # Expand dims to support easy broadcasting.
        y = np.expand_dims(y, axis=-1)
        x = np.expand_dims(x, axis=-1)

        num_anchors = len(sizes) + len(ratios)
        h = np.zeros((num_anchors, ), dtype=dtype)
        w = np.zeros((num_anchors, ), dtype=dtype)
        # Add first anchor boxes with ratio=1.
        h[0] = sizes[0] / img_shape[0]
        w[0] = sizes[0] / img_shape[1]
        di = 1
        if len(sizes) > 1:
            h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
            w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
            di += 1
        for i, r in enumerate(ratios):
            h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
            w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
        return y, x, h, w




