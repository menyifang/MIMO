# coding=utf-8
# date:
import tensorflow as tf
import numpy as np
import cv2
import os

if tf.__version__ >= '2.0':
    print("tf version >= 2.0")
    tf = tf.compat.v1
    tf.disable_eager_execution()


class human_segmenter(object):
    def __init__(self, model_path,is_encrypted_model=False):
        super(human_segmenter, self).__init__()
        f = tf.gfile.FastGFile(model_path, 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_graph = tf.import_graph_def(graph_def, name='')

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3  # 占用GPU 30%的显存
        self.sess = tf.InteractiveSession(graph=persisted_graph, config=config)

        # self.image_node = self.sess.graph.get_tensor_by_name("input_image:0")
        # # self.output_node = self.sess.graph.get_tensor_by_name("output_png:0")
        # # check if the nodename in model
        # if "output_png:0" in self.sess.graph_def.node:
        #     self.output_node = self.sess.graph.get_tensor_by_name("output_png:0")
        # else:
        #     self.output_node = self.sess.graph.get_tensor_by_name("output_alpha:0")
        # if "if_person:0" in self.sess.graph_def.node:
        #     self.logits_node = self.sess.graph.get_tensor_by_name("if_person:0")

        print("human_segmenter init done")
    
    def image_preprocess(self, img):
        if len(img.shape) == 2:
            img = np.dstack((img, img, img))
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        img = img[:, :, ::-1]
        img = img.astype(np.float32)
        return img
    
    def run(self, img):
        image_feed = self.image_preprocess(img)
        output_img_value, logits_value = self.sess.run([self.sess.graph.get_tensor_by_name("output_png:0"), self.sess.graph.get_tensor_by_name("if_person:0")],
                                                  feed_dict={self.sess.graph.get_tensor_by_name("input_image:0"): image_feed})
        # mask = output_img_value[:,:,-1]
        output_img_value = cv2.cvtColor(output_img_value, cv2.COLOR_RGBA2BGRA)
        return output_img_value

    def run_head(self, img):
        image_feed = self.image_preprocess(img)
        # image_feed = image_feed/255.0
        output_alpha = self.sess.run(self.sess.graph.get_tensor_by_name('output_alpha:0'),
                                     feed_dict={'input_image:0': image_feed})

        return output_alpha
    
    def get_human_bbox(self, mask):
        '''
        
        :param mask:
        :return: [x,y,w,h]
        '''
        print('dtype:{}, max:{},shape:{}'.format(mask.dtype, np.max(mask), mask.shape))
        ret, thresh = cv2.threshold(mask,127,255,0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        
        contoursArea = [cv2.contourArea(c) for c in contours]
        max_area_index = contoursArea.index(max(contoursArea))
        bbox = cv2.boundingRect(contours[max_area_index])
        return bbox
    
    
    def release(self):
        self.sess.close()


class head_segmenter(object):
    def __init__(self, model_path, is_encrypted_model=False):
        super(head_segmenter, self).__init__()
        f = tf.gfile.FastGFile(model_path, 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_graph = tf.import_graph_def(graph_def, name='')

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3  # 占用GPU 30%的显存
        self.sess = tf.InteractiveSession(graph=persisted_graph, config=config)

        print("human_segmenter init done")

    def image_preprocess(self, img):
        if len(img.shape) == 2:
            img = np.dstack((img, img, img))
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        img = img[:, :, ::-1]
        img = img.astype(np.float32)
        return img

    def run_head(self, img):
        image_feed = self.image_preprocess(img)
        # image_feed = image_feed/255.0
        output_alpha = self.sess.run(self.sess.graph.get_tensor_by_name('output_alpha:0'),
                                     feed_dict={'input_image:0': image_feed})

        return output_alpha

    def get_human_bbox(self, mask):
        '''

        :param mask:
        :return: [x,y,w,h]
        '''
        print('dtype:{}, max:{},shape:{}'.format(mask.dtype, np.max(mask), mask.shape))
        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        contoursArea = [cv2.contourArea(c) for c in contours]
        max_area_index = contoursArea.index(max(contoursArea))
        bbox = cv2.boundingRect(contours[max_area_index])
        return bbox

    def release(self):
        self.sess.close()


class hair_segmenter(object):
    def __init__(self, model_dir, is_encrypted_model=False):
        head_path = os.path.join(model_dir, 'Matting_headparser_6_18.pb')
        face_path = os.path.join(model_dir, 'segment_face.pb')
        detect_path = os.path.join(model_dir, 'face_detect.pb')

        self.sess = self.load_sess(head_path)
        image = np.ones((512, 512, 3))
        output_png = self.sess.run(self.sess.graph.get_tensor_by_name('output_alpha:0'),
                                   feed_dict={'input_image:0': image})

        self.sess_detect = self.load_sess(detect_path)
        oboxes, scores, num_detections = self.sess_detect.run(
            [self.sess_detect.graph.get_tensor_by_name('tower_0/boxes:0'),
             self.sess_detect.graph.get_tensor_by_name('tower_0/scores:0'),
             self.sess_detect.graph.get_tensor_by_name('tower_0/num_detections:0')],
            feed_dict={'tower_0/images:0': image[np.newaxis], 'training_flag:0': False})
        faceRects = []

        self.sess_face = self.load_sess(face_path)
        image = np.ones((512, 512, 3))
        output_alpha = self.sess_face.run(self.sess_face.graph.get_tensor_by_name('output_alpha_face:0'),
                                          feed_dict={'input_image_face:0': image})

    def load_sess(self, model_path):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        with tf.gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            sess.run(tf.global_variables_initializer())
        return sess

    def image_preprocess(self, img):
        if len(img.shape) == 2:
            img = np.dstack((img, img, img))
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        img = img[:, :, ::-1]
        img = img.astype(np.float32)
        return img

    def run_head(self, image):
        image_feed = self.image_preprocess(image)
        output_img_value = self.sess.run(self.sess.graph.get_tensor_by_name('output_alpha:0'),
                                     feed_dict={'input_image:0': image_feed})
        # mask = output_img_value[:,:,-1]
        output_img_value = cv2.cvtColor(output_img_value, cv2.COLOR_RGBA2BGRA)
        return output_img_value

    def run(self, image):
        h, w, c = image.shape
        faceRects = self.detect_face(image)
        face_num = len(faceRects)
        print('face_num:{}'.format(face_num))
        all_head_alpha = []
        all_face_mask = []
        for i in range(face_num):
            y1 = faceRects[i][0]
            y2 = faceRects[i][1]
            x1 = faceRects[i][2]
            x2 = faceRects[i][3]
            pad_y1, pad_y2, pad_x1, pad_x2 = self.pad_box(y1, y2, x1, x2, 0.15, 0.15, 0.15, 0.15, h, w)
            temp_img = image.copy()
            roi_img = temp_img[pad_y1:pad_y2, pad_x1:pad_x2]
            output_alpha = self.sess_face.run(self.sess_face.graph.get_tensor_by_name('output_alpha_face:0'),
                                              feed_dict={'input_image_face:0': roi_img[:, :, ::-1]})
            face_mask = np.zeros((h, w, 3))
            face_mask[pad_y1:pad_y2, pad_x1:pad_x2] = output_alpha
            all_face_mask.append(face_mask)
            # cv2.imwrite(str(i)+'face.jpg',face_mask)
            # cv2.imwrite(str(i)+'face_roi.jpg',roi_img)

        for i in range(face_num):
            y1 = faceRects[i][0]
            y2 = faceRects[i][1]
            x1 = faceRects[i][2]
            x2 = faceRects[i][3]
            pad_y1, pad_y2, pad_x1, pad_x2 = self.pad_box(y1, y2, x1, x2, 1.47, 1.47, 1.3, 2.0, h, w)
            temp_img = image.copy()
            for j in range(face_num):
                y1 = faceRects[j][0]
                y2 = faceRects[j][1]
                x1 = faceRects[j][2]
                x2 = faceRects[j][3]
                small_y1, small_y2, small_x1, small_x2 = self.pad_box(y1, y2, x1, x2, -0.1, -0.1, -0.1, -0.1, h, w)
                small_width = small_x2 - small_x1
                small_height = small_y2 - small_y1
                if (
                        small_x1 < 0 or small_y1 < 0 or small_width < 3 or small_height < 3 or small_x2 > w or small_y2 > h):
                    continue
                # if(i!=j):
                #     temp_img[small_y1:small_y2,small_x1:small_x2]=0
                if (i != j):
                    temp_img = temp_img * (1.0 - all_face_mask[j] / 255.0)

            roi_img = temp_img[pad_y1:pad_y2, pad_x1:pad_x2]
            output_alpha = self.sess.run(self.sess.graph.get_tensor_by_name('output_alpha:0'),
                                         feed_dict={'input_image:0': roi_img[:, :, ::-1]})
            head_alpha = np.zeros((h, w))
            head_alpha[pad_y1:pad_y2, pad_x1:pad_x2] = output_alpha[:, :, 2]
            all_head_alpha.append(head_alpha)

        print('all_head_alpha', all_head_alpha)
        # return all_head_alpha[0]



    def detect_face(self, img):
        h, w, c = img.shape
        input_img = cv2.resize(img[:, :, ::-1], (512, 512))
        boxes, scores, num_detections = self.sess_detect.run(
            [self.sess_detect.graph.get_tensor_by_name('tower_0/boxes:0'),
             self.sess_detect.graph.get_tensor_by_name('tower_0/scores:0'),
             self.sess_detect.graph.get_tensor_by_name('tower_0/num_detections:0')],
            feed_dict={'tower_0/images:0': input_img[np.newaxis], 'training_flag:0': False})
        faceRects = []
        for i in range(num_detections[0]):
            if scores[0, i] < 0.5:
                continue
            y1 = np.int(boxes[0, i, 0] * h)
            x1 = np.int(boxes[0, i, 1] * w)
            y2 = np.int(boxes[0, i, 2] * h)
            x2 = np.int(boxes[0, i, 3] * w)
            if x2 <= x1 + 3 or y2 <= y1 + 3:
                continue
            faceRects.append((y1, y2, x1, x2, y2 - y1, x2 - x1))
        sorted(faceRects, key=lambda x: x[4] * x[5], reverse=True)
        return faceRects

    def pad_box(self, y1, y2, x1, x2, left_ratio, right_ratio, top_ratio, bottom_ratio, h, w):
        box_w = x2 - x1
        box_h = y2 - y1
        pad_y1 = np.maximum(np.int(y1 - top_ratio * box_h), 0)
        pad_y2 = np.minimum(np.int(y2 + bottom_ratio * box_h), h - 1)
        pad_x1 = np.maximum(np.int(x1 - left_ratio * box_w), 0)
        pad_x2 = np.minimum(np.int(x2 + right_ratio * box_w), w - 1)
        return pad_y1, pad_y2, pad_x1, pad_x2



if __name__ == "__main__":
    img = cv2.imread('12345/images/0001.jpg')
    print(img.shape)
    fp = human_segmenter(model_path='assets/matting_human.pb')

    rgba = fp.run(img)
    # cv2.imwrite("human_mask1.png",mask)
    print("test done")
