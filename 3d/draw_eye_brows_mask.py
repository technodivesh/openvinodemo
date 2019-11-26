import dlib
import cv2
import numpy as np
import time as tm
from math import sin, cos

import models
import NonLinearLeastSquares
import ImageProcessing

from drawing import *

import FaceRendering
import utils_fswap

from argparse import ArgumentParser
import logging as log
import sys
import os.path as osp
from openvino.inference_engine import IENetwork
from ie_module import InferenceContext
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from head_pose_detector import HeadPoseDetector


print ("Press T to draw the keypoints and the 3D model")
print ("Press R to start recording to a video file")


#loading the keypoint detection model, the image and the 3D model
predictor_path = "models/shape_predictor_68_face_landmarks.dat"
image_name = "images/face/face01.jpg"

#the smaller this value gets the faster the detection will work
#if it is too small, the user's face might not be detected
maxImageSizeForDetection = 320

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils_fswap.load3DFaceModel("models/candide.npz")

# print(np.shape(mean3DShape), np.shape(blendshapes), np.shape(mesh), np.shape(idxs3D), np.shape(idxs2D))
# tm.sleep(10)

# print("mean3DShape", mean3DShape)
# print("blendshapes", blendshapes)
# print("mesh",mesh, len(mesh))
# print("idxs3D", idxs3D, type(idxs3D))
# print("idxs2D", idxs2D, type(idxs2D))

projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

modelParams = None
lockedTranslation = False
drawOverlay = False
cap = cv2.VideoCapture(0)
writer = None
cameraImg = cap.read()[1]

textureImg = cv2.imread(image_name)

textureCoords = utils_fswap.getFaceTextureCoords(textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)

# To get textureCoords(ie. landmarks of sample image )
##########################################################
                    
DEVICE_KINDS = ['CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL']

def generate_points(r, n, center):
        "generate n-sided polygon and translate it to center"
        for angle in range(0, 360, 360//n):
            yield (int(center[0] + r*cos(angle)), int(center [1] + r*sin(angle)))

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


args = build_argparser().parse_args()
log.basicConfig(format="[ %(levelname)s ] %(asctime)-15s %(message)s",
            level=log.INFO if not args.verbose else log.DEBUG, stream=sys.stdout)

# print(args)


class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args):
        used_devices = set([args.d_fd, args.d_lm, args.d_hp, args.d_reid])
        self.context = InferenceContext()
        context = self.context
        context.load_plugins(used_devices, args.cpu_lib, args.gpu_lib)
        if args.output:
            writer = True
        writer = True
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

        # orig_image = frame.copy()
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

        outputs = [rois, landmarks, head_pose]

        return outputs

frame_processor = FrameProcessor(args)


###########################################################


def draw_detection_roi(frame, roi):

        # Draw face ROI border
        cv2.rectangle(frame,
                      tuple(roi.position), tuple(roi.position + roi.size),
                      (0, 220, 0), 1)

def draw_detection_keypoints(frame, roi, landmarks, head_pose):

    # print("landmarks--", landmarks.get_array())
    keypoints = [landmarks.one,
                 landmarks.two,
                 landmarks.three,
                 landmarks.four,
                 landmarks.five,
                 landmarks.six]

    # print('.',end = '')
    # for point in keypoints:
    for point in landmarks.get_array():
        #print("point------", point, roi.position, roi.size)
        center = roi.position + roi.size * point
        # print("center------", center)
        cv2.circle(frame, tuple(center.astype(int)), 2, (0, 255, 255), 1)

def draw_detections(frame, detections):
    for roi, landmarks, head_pose in zip(*detections):
        draw_detection_roi(frame, roi)
        draw_detection_keypoints(frame, roi, landmarks, head_pose)
        # try:
        #     self.draw_eye_brows(frame, roi, landmarks, head_pose)
        # except Exception as ex:
        #     print(ex)
        # #self.draw_detection_head_pose(frame, roi, head_pose)


# textureImg = cv2.imread('images/eyebrows/e10.png')#textureImg  # right eyebrow 
# textureCoords= ""# textureCoords   points 15,16,17
# mesh = "" # mesh

renderer = FaceRendering.FaceRenderer(cameraImg, textureImg, textureCoords, mesh)


while True:
    cameraImg = cv2.flip(cap.read()[1],1)

    frame = cameraImg
    detections = frame_processor.process(frame)
    # draw_detections(frame, detections)

    # print(cameraImg)
    # shapes2Dorg = utils_fswap.getFaceKeypoints(frame, detector, predictor, maxImageSizeForDetection)
    # print("shapes2D-----org", shapes2D)

    # if shapes2Dorg is not None:
    #     for shape2Dorg in shapes2Dorg:
    #         print('shape2D--org',shape2Dorg, len(shape2Dorg[0]))
    # # To draw default
    # for i in shapes2Dorg[0].T:
    #     cv2.circle(frame, tuple(i), 4, (0, 0, 255), 1)

    # print(detections[1][0].get_array().T)
    if detections[1]:

        for roi, landmarks, head_pose in zip(*detections):
            landmarks = landmarks.get_array()
            centers = np.array( list(map(lambda xy: roi.position + roi.size * [xy[0],xy[1]], landmarks)) )
            # print(centers)
            centers = centers.astype('int64')
            lm = centers
            p = lm
            P = np.vstack((
                p[18:35] ,
                p[12] , (p[12]+p[13])*0.5 , p[13] , (p[13]+p[14])*0.5 , p[14] ,
                p[15] , (p[15] + p[16])*0.5, p[16] , (p[16] + p[17])*0.5 , p[17] ,
                p[14] , p[14] , p[14] , p[4] ,
                p[6] , p[6] , p[5] , p[7] , p[7] ,
                p[1] , p[1] , p[1] , p[0] , p[0] , p[0] ,
                p[2] , p[2] , p[2] , p[3] , p[3] , p[3] ,
                p[8] , p[8] , p[8] , p[8] , p[9] , p[9] , p[9] ,
                p[11] , p[11] , p[11] , p[11] , p[11] ,
                p[10] , p[10] , p[10] , p[10] , p[10] , p[10] , p[10]
                )).astype(int)#, axis=0)
            # print(f"[--------{P}\n\n\n\n\n{temp}---------]")
            # tm.sleep(10)
            shape2D = P.T

            # for i in P[17:27]:
            #     cv2.circle(frame, tuple(i), 2, (0, 255, 255), 1)



            # for i in range(len(landmarks)):
            #     landmarks[i] = roi.position + roi.size * landmarks[i]
            #     # landmarks[i][0],landmarks[i][1] = int(landmarks[i][0]),int(landmarks[i][1])

            # # print(landmarks.T)
            # shape2D = landmarks.T



            # shape2D = next(iter(shapes2Dorg))
            # print('shape2D--org',shapes2Dorg[0], len(shapes2Dorg[0][0]))
            # print("shape2D-----22222",shape2D, len(shape2D[0]))
            # continue

            # print(shape2D[0], len(shape2D[0]))   # list of [[x1,x2,x3.....xn],[y1,y2,y3.....yn]]
            #3D model parameter initialization
            modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])

            # print(f'mean3DShape[:, idxs3D]\n{mean3DShape[:, idxs3D[:2]]},\n\n\n\n shape2D[:, idxs2D])\n{shape2D[:, idxs2D[:2]]}\n\n{modelParams}')
            # print("modelParams--------pass")
            #3D model parameter optimization
            modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]), verbose=0)
            # print(f"{modelParams}--222------pass")

            # break

            #rendering the model to an image
            # print(blendshapes.shape)
            shape3D = utils_fswap.getShape3D(mean3DShape, blendshapes, modelParams)
            renderedImg = renderer.render(shape3D)
            # cv2.imshow('',renderedImg)

            #blending of the rendered face with the image
            mask = np.copy(renderedImg[:, :, 0])
            renderedImg = ImageProcessing.colorTransfer(cameraImg, renderedImg, mask)

            # apply rendered image on cameraImg
            # cv2.imshow('',cameraImg)
            if False:
                c1,r1 = tuple(map(int,roi.position))
                c2,r2 = tuple(map(int,roi.position + roi.size))
                t_ = r2+r1
                r1 = int(t_*0.32)
                r2 = int(t_*0.5)
                # print('\n\n\n\n\n\n\n\n\n',r1,r2,'----', c1,c2,'\n\n\n\n\n\n\n\n\n')
                # r1,r2,c1,c2 = [234,401,87,318]
                # print('\n\n\n\n\n\n\n\n\n',r1,r2,'----', c1,c2,'\n\n\n\n\n\n\n\n\n')
                cameraImg1 = ImageProcessing.blendImages(
                    renderedImg[r1:r2,c1:c2,:], 
                    cameraImg[r1:r2,c1:c2,:], 
                    mask[r1:r2,c1:c2])
                row,col,ch = cameraImg1.shape
                print('--------------------------')
                # cv2.imshow('fgds',cameraImg1)
                # cv2.waitKey(0)

                cameraImg[r1:row+r1,c1:col+c1] = cameraImg1
            else:
                th = roi.size[0]*0.05
                E1 = (P[21][0],  P[21][1]+th  )
                E2 = (P[19][0],  P[19][1]+th  )
                E3 = (P[17][0],  P[17][1]+th  )
                E4 = (P[17][0]-th,  P[17][1]+th  )
                E5 = (P[17][0]-th,  P[17][1]  )
                E6 = (P[17][0]-th,  P[17][1]-th  )
                E7 = (P[17][0],  P[17][1]-th  )
                E8 = (P[19][0],  P[19][1]-th  )
                E9 = (P[21][0],  P[21][1]-th  )

                E10 = (P[22][0],  P[22][1]-th  )
                E11 = (P[24][0],  P[24][1]-th  )
                E12 = (P[26][0],  P[26][1]-th  )
                E13 = (P[26][0]+th,  P[26][1]-th  )
                E14 = (P[26][0]+th,  P[26][1]  )
                E15 = (P[26][0]+th,  P[26][1]+th  )
                E16 = (P[26][0],  P[26][1]+th  )
                E17 = (P[24][0],  P[24][1]+th  )
                E18 = (P[22][0],  P[22][1]+th  )
                j = (np.vstack((
                    P[21], E1,E2, E3,E4,E5,E6,E7, E8,E9, P[21],
                    P[22], E10, E11, E12,E13,E14,E15,E16, E17, E18, P[22]
                    
                    )) ).astype('int32')
                # print([point for point in generate_points(th,8,P[17])])
                
                # cv2.fillPoly(frame, [j], (0,255,0))

                # cv2.fillPoly(frame, [np.array([point for point in generate_points(th,8,P[17])])], (0,0,255))
                # for i in j:
                #     cv2.circle(frame, tuple(i), 5, (0, 255, 255), 1)
                # print(j)
                image = renderedImg
                mask1 = np.zeros(image.shape, dtype=np.uint8)
                roi_corners = np.array([j], dtype=np.int32)
                channel_count = image.shape[2]
                ignore_mask_color = (255,)*channel_count
                cv2.fillPoly(mask1, roi_corners, ignore_mask_color)
                masked_image = cv2.bitwise_and(image, mask1)
                # cv2.imshow('image_masked.png', masked_image)

                # print(masked_image.shape,cameraImg.shape)
                # gray = cv2.cvtColor(masked_image,cv2.COLOR_BGR2GRAY)
                # cameraImg = (cameraImg * (masked_image==0)) #(masked_image * (masked_image>5))


                cameraImg = ImageProcessing.blendImages(masked_image, cameraImg, mask1,featherAmount = 0.1)

            # print(f"renderedImg, {renderedImg.shape}, cameraImg, {cameraImg.shape}, mask, {mask.shape}")
            # print(cameraImg.shape)
            # cv2.imshow('',cameraImg)
            

            #drawing of the mesh and keypoints
            if drawOverlay:
                drawPoints(cameraImg, shape2D.T)
                drawProjectedShape(cameraImg, [mean3DShape, blendshapes], projectionModel, mesh, modelParams, lockedTranslation)

    # if True:#writer is not None:
    #     print('save ho rha hai')
    #     f_w, f_h, _ = cameraImg.shape
    #     # fourcc = cv2.CV_FOURCC(*'XVID')
    #     writer = cv2.VideoWriter("outoutout.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 10, (f_w,f_h))
    #     writer.write(cameraImg)

    if writer is not None:
        writer.write(cameraImg)

    cv2.imshow('image', cameraImg)
    key = cv2.waitKey(1)

    if key == 27 or key==ord('q'):
        writer.release()
        break
    if key == ord('t'):
        drawOverlay = not drawOverlay
    if key == ord('r'):
        if writer is None:
            print ("Starting video writer")
            writer = cv2.VideoWriter("out.avi", cv2.VideoWriter.fourcc(*'MJPG'), 25, (cameraImg.shape[1], cameraImg.shape[0]))

            if writer.isOpened():
                print ("Writer succesfully opened")
            else:
                writer = None
                print ("Writer opening failed")
        else:
            print ("Stopping video writer")
            writer.release()
            writer = None