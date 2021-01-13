from handlandmarks import HandLandmarks
import torch

m = HandLandmarks()
import coremltools as ct
m.load_weights("./HandLandmarks.pth")
m.eval()

# np.random.seed(0)
# a = torch.from_numpy(np.random.rand(1, 3, 256, 256).astype(np.float32))
# bb = m(a)
# np.save('npseed0reg_torch.npy', bb[1].detach().numpy())
traced_model = torch.jit.trace(m, torch.rand(1, 3, 256, 256), check_trace=True)
#print(traced_model)
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="image", shape=ct.Shape(shape=(ct.RangeDim(1,4), 3, 256, 256,)), bias=[-1,-1,-1], scale=1/127.5)],
    minimum_ios_deployment_target='14'
)
print(mlmodel)
mlmodel.save('../App/BlazePalm CoreML/BlazeLandmarks.mlmodel')