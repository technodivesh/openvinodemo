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

import numpy as np

from utils import cut_rois, resize_input
from ie_module import Module
import time
class HeadPoseDetector(Module):
    # POINTS_NUMBER = 5

    class Result:
        def __init__(self, outputs):
            self.angle_y_fc = outputs[0]
            # self.angle_p_fc = outputs[1]
            # self.angle_r_fc = outputs[2]
            self.points = outputs
            # print(outputs.shape,end='')

            p = lambda i: self[i]
            self.one = p(0)
            # self.two = p(13)
            # self.three = p(14)
            # self.four = p(15)
            # self.five = p(16)
            # self.six = p(17)


        def __getitem__(self, idx):
            return self.points[idx]

        def get_array(self):
            return np.array(self.points, dtype=np.float64)

    def __init__(self, model):
        super(HeadPoseDetector, self).__init__(model)

        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) > 1, "Expected 1 output blob"
        self.input_blob = next(iter(model.inputs))
        # self.output_blob = next(iter(model.outputs))
        self.output_blob = list(model.outputs.keys())
        self.input_shape = model.inputs[self.input_blob].shape
        # print(list(model.outputs.keys())[1])
        self.output_shape = [
            model.outputs[self.output_blob[0]].shape,
            model.outputs[self.output_blob[1]].shape,
            model.outputs[self.output_blob[2]].shape
            ]

        # print(self.output_shape)
        # assert np.array_equal([1, 70],
        #                       model.outputs[self.output_blob].shape), \
        #     "Expected model output shape %s, but got %s" % \
        #     ([1, 70],
        #      model.outputs[self.output_blob].shape)

    def preprocess(self, frame, rois):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        inputs = cut_rois(frame, rois)
        inputs = [resize_input(input, self.input_shape) for input in inputs]
        return inputs

    def enqueue(self, input):
        return super(HeadPoseDetector, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        for input in inputs:
            self.enqueue(input)

    def get_head_pose(self):


        try:
            outputs = self.get_outputs()
            output = outputs[0]
            #print('\n\n     ',[out[self.output_blob].shape for out in outputs],'\n\n')
            #time.sleep(1000)
            # results = [[HeadPoseDetector.Result(out[self.output_blob[1]]),HeadPoseDetector.Result(out[self.output_blob[0]])] for out in outputs]

            results = [
                [
                    HeadPoseDetector.Result(output[self.output_blob[0]]),
                    HeadPoseDetector.Result(output[self.output_blob[1]]),
                    HeadPoseDetector.Result(output[self.output_blob[2]])
                ]
            ]
        except Exception as e:
            # raise e
            return []


        return results
