# smpl export
import numpy as np
import pickle
import torch
from torch.nn import Module
import os
import math
from torch.autograd import Variable
from tqdm import tqdm
from blazepalm import PalmDetector


class PalmDetectorScaled(Module):
	def __init__(self, bfModel):
		super(PalmDetectorScaled, self).__init__()
		self.bfModel = bfModel
		self.anchors = torch.nn.Parameter(self.bfModel.anchors)

	def forward(self, x):
		out = self.bfModel(x)
		c = out[0].sigmoid() # .squeeze() # 2944
		bb = self.decodeBoxModel(out[1].squeeze()) # .unsqueeze(0) # .view(2944, 9, 2)
		return bb, c # torch.cat([r, c], dim=-1) # 2944, 17

	def decodeBoxModel(self, raw_boxes):
		"""Converts the predictions into actual coordinates using
		the anchor boxes. Processes the entire batch at once.
		"""
		# print(raw_boxes.size())
		# boxes = torch.zeros(size=(896, 16)) # raw_boxes.clone() # torch.zeros_like(raw_boxes)

		# self.anchors[:, 2/3] is 1 so skipping it

		x_center = (raw_boxes[:, 8] / 256.0) + self.anchors[:, 0]
		y_center = (raw_boxes[:, 9] / 256.0) + self.anchors[:, 1]

		w = (raw_boxes[:, 2] / 256.0) * 2.6
		h = (raw_boxes[:, 3] / 256.0) * 2.6

		bb = []

		bb.append(x_center - w / 2.0)
		bb.append(y_center - h / 2.0)
		bb.append(x_center + w / 2.0)
		bb.append(y_center + h / 2.0)

		# concat_stuff = []

		# for k in range(7):
		#     offset = 4 + k*2
		#     # raw_boxes[:, offset    ] = (raw_boxes[:, offset    ] / 128.0) * self.anchors[:, 2] + self.anchors[:, 0] # x
		#     # raw_boxes[:, offset + 1] = (raw_boxes[:, offset + 1] / 128.0) * self.anchors[:, 3] + self.anchors[:, 1] # y
		#     concat_stuff.append((raw_boxes[:, offset    ] / 256.0) + self.anchors[:, 0])
		#     concat_stuff.append((raw_boxes[:, offset + 1] / 256.0) + self.anchors[:, 1])

		# concat_stuff.append(x_center - w / 2.0)
		# concat_stuff.append(y_center - h / 2.0) # now it'll have 16 values so SIMD 16 can be used

		return torch.stack(bb, dim=-1) # , torch.stack(concat_stuff, dim=-1)

import coremltools as ct
from coremltools.converters.onnx import convert

bfModel = PalmDetector()
bfModel.load_weights("./palmdetector.pth")
bfModel.load_anchors("./anchors.npy")

bfs = PalmDetectorScaled(bfModel)
bfs.eval()

traced_model = torch.jit.trace(bfs, torch.rand(1, 3, 256, 256), check_trace=True)
# print(traced_model)
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="image", shape=ct.Shape(shape=(1, 3, 256, 256,)), bias=[-1,-1,-1], scale=1/127.5)]
)
mlmodel.save('../App/BlazePalm CoreML/BlazePalmScaled.mlmodel')

print(mlmodel)
# Save converted CoreML model


# result = mlmodel.predict({"betas_pose_trans": x, "v_personal": y}, usesCPUOnly=True)