import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Taken from https://github.com/vidursatija/BlazeFace-CoreML/blob/master/ML/blazeface.py
class ResModule(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1):
		super(ResModule, self).__init__()
		self.stride = stride
		self.channel_pad = out_channels - in_channels
		# kernel size is always 3
		kernel_size = 5

		if stride == 2:
			self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
			padding = 0
		else:
			padding = (kernel_size - 1) // 2

		self.convs = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
						kernel_size=kernel_size, stride=stride, padding=padding, 
						groups=in_channels, bias=True),
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
						kernel_size=1, stride=1, padding=0, bias=True),
		)

		self.act = nn.ReLU(inplace=True)

	def forward(self, x):
		if self.stride == 2:
			h = F.pad(x, (1, 2, 1, 2), "constant", 0)
			x = self.max_pool(x)
		else:
			h = x

		if self.channel_pad > 0:
			x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

		return self.act(self.convs(h) + x)


class ResBlock(nn.Module):
	def __init__(self, in_channels, number=2):
		super(ResBlock, self).__init__()
		layers = [ResModule(in_channels, in_channels) for _ in range(number)]

		self.f = nn.Sequential(*layers)

	def forward(self, x):
		return self.f(x)


# From https://github.com/google/mediapipe/blob/master/mediapipe/models/palm_detection.tflite
class HandLandmarks(nn.Module):
	def __init__(self):
		super(HandLandmarks, self).__init__()

		self.backbone1 = nn.Sequential(
			nn.ConstantPad2d((0, 1, 0, 1), value=0.0),
			nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=0, bias=True),
			nn.ReLU(inplace=True),

			ResBlock(24),
			ResModule(24, 48, stride=2)
		) # 64x64

		self.backbone2 = nn.Sequential(
			ResBlock(48),
			ResModule(48, 96, stride=2)
		) # 32x32

		self.backbone3 = nn.Sequential(
			ResBlock(96),
			ResModule(96, 96, stride=2)
		) # 16x16

		self.backbone4 = nn.Sequential(
			ResBlock(96),
			ResModule(96, 96, stride=2), # 8x8
			nn.Upsample(scale_factor=2, mode='bilinear') # align_corners = false
		) # 16x16
		# add output of backbone3 here

		self.backbone5 = nn.Sequential(
			ResModule(96, 96),
			nn.Upsample(scale_factor=2, mode='bilinear')
		) # 32x32
		# add output of backbone2 here

		self.backbone6 = nn.Sequential(
			ResModule(96, 96),
			nn.Conv2d(in_channels=96, out_channels=48, kernel_size=1, stride=1, padding=0, bias=True),
			nn.Upsample(scale_factor=2, mode='bilinear')
		) # 64x64
		# add output of backbone1 here

		ff_layers = []
		ResBlockChannels = [48, 96, 288, 288, 288]
		ResModuleChannels = [96, 288, 288, 288, 288]
		for rbc, rmc in zip(ResBlockChannels, ResModuleChannels):
			ff_layers.append(ResBlock(rbc, number=4))
			ff_layers.append(ResModule(rbc, rmc, stride=2))
		ff_layers.append(ResBlock(288, number=4))

		self.ff = nn.Sequential(*ff_layers)

		self.handflag = nn.Conv2d(in_channels=288, out_channels=1, kernel_size=2, stride=1, padding=0, bias=True)
		self.handedness = nn.Conv2d(in_channels=288, out_channels=1, kernel_size=2, stride=1, padding=0, bias=True)
		self.reg_3d = nn.Conv2d(in_channels=288, out_channels=63, kernel_size=2, stride=1, padding=0, bias=True)


	def forward(self, x):
		b1 = self.backbone1(x) # 64x64
		# print(b1.size())

		b2 = self.backbone2(b1) # 32x32
		# print(b2.size())

		b3 = self.backbone3(b2) # 16x16
		# print(b3.size())

		b4 = self.backbone4(b3) + b3 # 16x16
		# print(b4.size())

		b5 = self.backbone5(b4) + b2 # 32x32
		# print(b5.size())

		b6 = self.backbone6(b5) + b1 # 64x64
		# print(b6.size())

		ff = self.ff(b6) # 1x288x2x2

		hand = self.handflag(ff) # 1x1x1x1
		hand = hand.squeeze().sigmoid().reshape(-1, 1)

		handedness = self.handedness(ff) # 1x1x1x1
		handedness = handedness.squeeze().sigmoid().reshape(-1, 1)

		reg_3d = self.reg_3d(ff) # 1x63x1x1
		reg_3d = reg_3d.permute(0, 2, 3, 1).reshape(-1, 63) / 256.0

	def load_weights(self, path):
	    self.load_state_dict(torch.load(path))
	    self.eval()        


if __name__ == '__main__':
	m = HandLandmarks()
	import coremltools as ct
	# m.load_weights("./HandLandmarks.pth")
	# m.load_anchors('./anchors.npy')
	m.eval()

	# np.random.seed(0)
	# a = torch.from_numpy(np.random.rand(1, 3, 256, 256).astype(np.float32))
	# bb = m(a)
	# np.save('npseed0reg_torch.npy', bb[1].detach().numpy())
	traced_model = torch.jit.trace(m, torch.rand(1, 3, 256, 256), check_trace=True)
	print(traced_model)
	# mlmodel = ct.convert(
	#     traced_model,
	#     inputs=[ct.ImageType(name="image", shape=ct.Shape(shape=(1, 3, 256, 256,)), bias=[-1,-1,-1], scale=1/127.5)],
	#     minimum_ios_deployment_target='14'
	# )
	# print(mlmodel)
	# mlmodel.save('./BlazeLandmarks.mlmodel')