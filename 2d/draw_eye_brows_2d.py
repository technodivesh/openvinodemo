#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from math import hypot


import logging as log
import os.path as osp
import sys
import time
from argparse import ArgumentParser
import pickle

import cv2
import numpy as np

from openvino.inference_engine import IENetwork
from ie_module import InferenceContext
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from head_pose_detector import HeadPoseDetector

DEVICE_KINDS = ['CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL']


def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group('General')
    general.add_argument('-i', '--input', metavar="PATH", default='0',
                         help="(optional) Path to the input video " \
                         "('0' for the camera, default)")
    general.add_argument('-o', '--output', metavar="PATH", default="",
                         help="(optional) Path to save the output video to")
    general.add_argument('--no_show', action='store_true',
                         help="(optional) Do not display output")
    general.add_argument('-tl', '--timelapse', action='store_true',
                         help="(optional) Auto-pause after each frame")

    models = parser.add_argument_group('Models')
    models.add_argument('-m_fd', metavar="PATH", default="", required=True,
                        help="Path to the Face Detection model XML file")
    models.add_argument('-m_lm', metavar="PATH", default="", required=True,
                        help="Path to the Facial Landmarks Regression model XML file")
    models.add_argument('-m_hp', metavar="PATH", default="", required=True,
                        help="Path to the Head Pose model XML file")


    infer = parser.add_argument_group('Inference options')
    infer.add_argument('-d_fd', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Face Detection model (default: %(default)s)")
    infer.add_argument('-d_lm', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Facial Landmarks Regression model (default: %(default)s)")
    infer.add_argument('-d_hp', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Head Pose model (default: %(default)s)")
    infer.add_argument('-d_reid', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Face Reidentification model (default: %(default)s)")
    infer.add_argument('-l', '--cpu_lib', metavar="PATH", default="",
                       help="(optional) For MKLDNN (CPU)-targeted custom layers, if any. " \
                       "Path to a shared library with custom layers implementations")
    infer.add_argument('-c', '--gpu_lib', metavar="PATH", default="",
                       help="(optional) For clDNN (GPU)-targeted custom layers, if any. " \
                       "Path to the XML file with descriptions of the kernels")
    infer.add_argument('-v', '--verbose', action='store_true',
                       help="(optional) Be more verbose")
    infer.add_argument('-pc', '--perf_stats', action='store_true',
                       help="(optional) Output detailed per-layer performance stats")
    infer.add_argument('-t_fd', metavar='[0..1]', type=float, default=0.6,
                       help="(optional) Probability threshold for face detections" \
                       "(default: %(default)s)")
    infer.add_argument('-t_id', metavar='[0..1]', type=float, default=0.3,
                       help="(optional) Cosine distance threshold between two vectors " \
                       "for face identification (default: %(default)s)")
    infer.add_argument('-exp_r_fd', metavar='NUMBER', type=float, default=1.15,
                       help="(optional) Scaling ratio for bboxes passed to face recognition " \
                       "(default: %(default)s)")


    return parser



class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args):
        used_devices = set([args.d_fd, args.d_lm, args.d_hp, args.d_reid])
        self.context = InferenceContext()
        context = self.context
        context.load_plugins(used_devices, args.cpu_lib, args.gpu_lib)
        for d in used_devices:
            context.get_plugin(d).set_config({
                "PERF_COUNT": "YES" if args.perf_stats else "NO"})

        log.info("Loading models")
        face_detector_net = self.load_model(args.m_fd)
        landmarks_net = self.load_model(args.m_lm)
        head_pose_net = self.load_model(args.m_hp)
        # face_reid_net = self.load_model(args.m_reid)

        self.face_detector = FaceDetector(face_detector_net,
                                          confidence_threshold=args.t_fd,
                                          roi_scale_factor=args.exp_r_fd)

        self.landmarks_detector = LandmarksDetector(landmarks_net)
        self.head_pose_detector = HeadPoseDetector(head_pose_net)
        self.face_detector.deploy(args.d_fd, context)
        self.landmarks_detector.deploy(args.d_lm, context,
                                       queue_size=self.QUEUE_SIZE)
        self.head_pose_detector.deploy(args.d_hp, context,
                                       queue_size=self.QUEUE_SIZE)

        log.info("Models are loaded")


    def load_model(self, model_path):
        model_path = osp.abspath(model_path)
        model_description_path = model_path
        model_weights_path = osp.splitext(model_path)[0] + ".bin"
        log.info("Loading the model from '%s'" % (model_description_path))
        assert osp.isfile(model_description_path), \
            "Model description is not found at '%s'" % (model_description_path)
        assert osp.isfile(model_weights_path), \
            "Model weights are not found at '%s'" % (model_weights_path)
        model = IENetwork(model_description_path, model_weights_path)
        log.info("Model is loaded")
        return model

    def process(self, frame):

        # print("frame.shape--", frame.shape)
        assert len(frame.shape) == 3, \
            "Expected input frame in (H, W, C) format"
        assert frame.shape[2] in [3, 4], \
            "Expected BGR or BGRA input"

        orig_image = frame.copy()
        frame = frame.transpose((2, 0, 1)) # HWC to CHW
        # print("frame.shape--", frame.shape)
        frame = np.expand_dims(frame, axis=0)
        # print("frame.shape--", frame.shape)

        self.face_detector.clear()
        self.landmarks_detector.clear()
        self.head_pose_detector.clear()
        # self.face_identifier.clear()

        self.face_detector.start_async(frame)
        rois = self.face_detector.get_roi_proposals(frame)
        if self.QUEUE_SIZE < len(rois):
            log.warning("Too many faces for processing." \
                    " Will be processed only %s of %s." % \
                    (self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]
        self.landmarks_detector.start_async(frame, rois)
        self.head_pose_detector.start_async(frame, rois)
        landmarks = self.landmarks_detector.get_landmarks()
        head_pose = self.head_pose_detector.get_head_pose()

        outputs = [rois, landmarks, head_pose, 'test']

        return outputs


    def get_performance_stats(self):
        stats = {
            'face_detector': self.face_detector.get_performance_stats(),
            'landmarks': self.landmarks_detector.get_performance_stats(),
            'face_identifier': self.face_identifier.get_performance_stats(),
        }
        return stats


class Visualizer:
    BREAK_KEY_LABELS = "q(Q) or Escape"
    BREAK_KEYS = {ord('q'), ord('Q'), 27}


    def __init__(self, args):
        self.frame_processor = FrameProcessor(args)
        self.display = not args.no_show
        self.print_perf_stats = args.perf_stats
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.frame_num = 0
        self.frame_count = -1
        self.frame_timeout = 0 if args.timelapse else 1

        self.eye_brow_right = cv2.imread("images/eyebrows/e10.png")#, self.eye_brow_obj.get_image()
        self.eye_brow_left = cv2.flip(self.eye_brow_right,1)


    def fliped(self,frame):
        return cv2.flip(frame,1)
        
    def get_distance(self,b,a):
        print(b,a)
        distance = abs(b-a)
        # time.sleep(10)
        distance = map(int,distance)
        print(distance)
        return tuple(distance)

    def get_angle(self,b,a):

        c = np.array([a[0],b[1]])
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(cosine_angle))
        angle = angle if b[1]>=a[1] else 360-angle
        # ####print(angle,'----------------------------------angle--')
        #time.sleep(10)
        return angle

    def rotateImage(self,image, angle,frame='left'):
        angle = angle if angle>0 else 360+angle

        #print(angle)
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        # cv2.imshow(frame,result)
        # cv2.waitKey(100)
        return result
        
    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_text_with_background(self, frame, text, origin,
                                  font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.0,
                                  color=(0, 0, 0), thickness=1, bgcolor=(255, 255, 255)):
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(frame,
                      tuple((origin + (0, baseline)).astype(int)),
                      tuple((origin + (text_size[0], -text_size[1])).astype(int)),
                      bgcolor, cv2.FILLED)
        cv2.putText(frame, text,
                    tuple(origin.astype(int)),
                    font, scale, color, thickness)
        return text_size, baseline

    def draw_detection_roi(self, frame, roi, identity):

        # Draw face ROI border
        cv2.rectangle(frame,
                      tuple(roi.position), tuple(roi.position + roi.size),
                      (0, 220, 0), 2)

    def draw_detection_keypoints(self, frame, roi, landmarks, head_pose):
        keypoints = [landmarks.one,
                     landmarks.two,
                     landmarks.three,
                     landmarks.four,
                     landmarks.five,
                     landmarks.six]

        print('.',end = '')
        for point in keypoints:
            #print("point------", point, roi.position, roi.size)
            center = roi.position + roi.size * point
            # print("center------", center)
            cv2.circle(frame, tuple(center.astype(int)), 2, (0, 255, 255), 2)

    def draw_eye_brows(self, frame, roi, landmarks, head_pose):


        # angle_up_down, angle_head_tilt, angle_left_right = map(lambda x:x//20*20 ,map(int,self.get_head_pose_angles(frame, roi, head_pose)))
        angle_up_down, angle_head_tilt, angle_left_right = self.get_head_pose_angles(frame, roi, head_pose)
        angle_para = {
            'angle_up_down':angle_up_down,
            'angle_head_tilt':angle_head_tilt,
            'angle_left_right':angle_left_right
        }

        # center point for each eye
        center_r = roi.position + roi.size * landmarks.five
        center_l = roi.position + roi.size * landmarks.two

        center_r1 = roi.position + roi.size * landmarks.four
        center_r2 = roi.position + roi.size * landmarks.six
        center_l1 = roi.position + roi.size * landmarks.one
        center_l2 = roi.position + roi.size * landmarks.three

        height, width, channels_f = frame.shape


        ############################
        mask = np.zeros((height, width), np.uint8)
        mask.fill(0)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eb_l_width = int(hypot(center_l1[0] - center_l2[0],
                           center_l1[1] - center_l2[1]) * 1.7)
        eb_l_height = int(eb_l_width * 0.4)


        # New nose position
        top_left = (int(center_l[0] - eb_l_width / 2),
                              int(center_l[1] - eb_l_height / 2))
        bottom_right = (int(center_l[0] + eb_l_width / 2),
                       int(center_l[1] + eb_l_height / 2))


        eye_brow_left = self.rotateImage(self.eye_brow_left,-angle_head_tilt)
        # Adding the new nose
        eb_left = cv2.resize(eye_brow_left, (eb_l_width, eb_l_height))
        eb_left_gray = cv2.cvtColor(eb_left, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(eb_left_gray, 25, 255, cv2.THRESH_BINARY_INV)
        eb_l_area = frame[top_left[1]: top_left[1] + eb_l_height,
                    top_left[0]: top_left[0] + eb_l_width]
        eb_l_area_no_eb = cv2.bitwise_and(eb_l_area, eb_l_area, mask=mask)
        final_eb_left = cv2.add(eb_l_area_no_eb, eb_left)
        frame[top_left[1]: top_left[1] + eb_l_height,
                    top_left[0]: top_left[0] + eb_l_width] = final_eb_left


        ############################

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eb_r_width = int(hypot(center_r1[0] - center_r2[0],
                           center_r1[1] - center_r2[1]) * 1.7)
        eb_r_height = int(eb_r_width * 0.4)


        # New nose position
        top_left = (int(center_r[0] - eb_r_width / 2),
                              int(center_r[1] - eb_r_height / 2))
        bottom_right = (int(center_r[0] + eb_r_width / 2),
                       int(center_r[1] + eb_r_height / 2))


        # Adding the new nose
        eye_brow_right = self.rotateImage(self.eye_brow_right,-angle_head_tilt)
        eb_right = cv2.resize(eye_brow_right, (eb_r_width, eb_r_height))
        eb_right_gray = cv2.cvtColor(eb_right, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(eb_right_gray, 25, 255, cv2.THRESH_BINARY_INV)
        nose_area = frame[top_left[1]: top_left[1] + eb_r_height,
                    top_left[0]: top_left[0] + eb_r_width]
        nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=mask)
        final_nose = cv2.add(nose_area_no_nose, eb_right)
        frame[top_left[1]: top_left[1] + eb_r_height,
                    top_left[0]: top_left[0] + eb_r_width] = final_nose



    def get_head_pose_angles(self, frame, roi, head_pose):

        angle_p_fc, angle_r_fc, angle_y_fc = [ next(iter(obj))[0] for obj in head_pose]

        return angle_p_fc, angle_r_fc, angle_y_fc

    def draw_detections(self, frame, detections):
        for roi, landmarks, head_pose, identity in zip(*detections):
            # self.draw_detection_roi(frame, roi, identity)
            # self.draw_detection_keypoints(frame, roi, landmarks, head_pose)
            try:
                self.draw_eye_brows(frame, roi, landmarks, head_pose)
            except Exception as ex:
                print(ex)
            #self.draw_detection_head_pose(frame, roi, head_pose)

    def display_interactive_window(self, frame):
        #frame = cv2.flip(frame,1)
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text = "Press '%s' key to exit" % (self.BREAK_KEY_LABELS)
        thickness = 2
        text_size = cv2.getTextSize(text, font, text_scale, thickness)
        origin = np.array([frame.shape[-2] - text_size[0][0] - 10, 10])
        line_height = np.array([0, text_size[0][1]]) * 1.5
        cv2.putText(frame, text,
                    tuple(origin.astype(int)), font, text_scale, color, thickness)

        cv2.imshow('openvino demo', frame)

    def should_stop_display(self):
        key = cv2.waitKey(self.frame_timeout) & 0xFF
        return key in self.BREAK_KEYS

    def process(self, input_stream, output_stream):
        frame_rate = 10
        prev = 0
        self.input_stream = input_stream
        self.output_stream = output_stream

        while input_stream.isOpened():

            time_elapsed = time.time() - prev
            has_frame, frame = input_stream.read()

            # if time_elapsed > 1./frame_rate:
            #     prev = time.time()
            # else:
            #     continue

            if not has_frame:
                break
            frame = self.fliped(frame) # manually added by SG to make mirror like effect

            # if self.input_crop is not None:
            #     frame = Visualizer.center_crop(frame, self.input_crop)

            detections = self.frame_processor.process(frame)

            self.draw_detections(frame, detections)

            if output_stream:
                output_stream.write(frame)
            if self.display:
                self.display_interactive_window(frame)
                if self.should_stop_display():
                    break

            self.update_fps()
            self.frame_num += 1

    def run(self, args):

        input_stream = Visualizer.open_input_stream(args.input)
        if input_stream is None or not input_stream.isOpened():
            log.error("Cannot open input stream: %s" % args.input)
        fps = input_stream.get(cv2.CAP_PROP_FPS)

        frame_size = (int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.frame_count = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        log.info("Input stream info: %d x %d @ %.2f FPS" % \
            (frame_size[0], frame_size[1], fps))
        output_stream = Visualizer.open_output_stream(args.output, fps, frame_size)

        self.process(input_stream, output_stream)

        # Release resources
        if output_stream:
            output_stream.release()
        if input_stream:
            input_stream.release()

        cv2.destroyAllWindows()


    @staticmethod
    def open_input_stream(path):
        log.info("Reading input data from '%s'" % (path))
        stream = path
        try:
            stream = int(path)
        except ValueError:
            pass
        return cv2.VideoCapture(stream)

    @staticmethod
    def open_output_stream(path, fps, frame_size):

        print("path, fps, frame_size---", path, fps, frame_size)
        output_stream = None
        if path != "":
            # pass
            if not path.endswith('.avi'):
                log.warning("Output file extension is not 'avi'. " \
                        "Some issues with output can occur, check logs.")
            log.info("Writing output to '%s'" % (path))
            output_stream = cv2.VideoWriter(path,
                                            cv2.VideoWriter.fourcc(*'MJPG'), fps, frame_size)
        return output_stream


def main():
    args = build_argparser().parse_args()

    log.basicConfig(format="[ %(levelname)s ] %(asctime)-15s %(message)s",
                    level=log.INFO if not args.verbose else log.DEBUG, stream=sys.stdout)

    log.debug(str(args))
    visualizer = Visualizer(args)
    visualizer.run(args)

if __name__ == '__main__':
    main()
