import base64
import numpy as np
from liveness_detection.dataset.sequence import LivenessDataset
from liveness_detection.augmentation import Preprocessor

# images = np.random.randint(0, 255, (5, 144, 144, 3), 'uint8')
# images_encoded = str(base64.b64encode(images), 'utf-8')
# data = pd.DataFrame(
#         data=[images_encoded], columns=["image"]
#     )#.to_json(orient="split")
# detector = mlflow.pyfunc.load_model('../models/cc04b77233224bce9a6d8cd20528351c/artifacts/last')
#
# print(detector.predict(data))
# from liveness_detection.dataset import LivenessDataset
# from liveness_detection.augmentation import Preprocessor
# preprocessor = Preprocessor(augment=True)
# train_dataset = LivenessDataset('../data/train_live.txt',
#                                               '../data/train_spoofed.txt',
#                                               preprocessor, 5)

dataset = LivenessDataset('../data/train_live.txt', '../data/train_spoofed.txt', Preprocessor(True), 5)
print(len(dataset))
labels = np.array([i[1] for i in dataset]).squeeze()
print(len(labels))
print(np.unique(labels, return_counts=True))

# model1 = LivenessDetector.load_from_checkpoint('../last.ckpt')
# model2 = LivenessDetector.load_from_checkpoint('../models/lightning/last.ckpt').to('cuda')
# mlflow.pytorch.sa
# print(list(model1.parameters())[-4], list(model2.parameters())[-4])
# model = L('../model/model/data/model.pth')

# model1.eval()
# model2.eval()
# # print(model.swa_model)
# # model.freeze()
# #m
# def hook(model, input, output):
#     print(output)
# print(list(model1.swa_model.module._bn0.parameters()))
# print(list(model2.swa_model.module._bn0.parameters()))
# torch.optim.swa_utils.update_bn(torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8), model2, 'cuda')
# print(list(model2.swa_model.module._bn0.parameters()))
# print('BN')
#
# # print(model1.swa_model.module._bn0)
# # print(model2.swa_model.module._bn0)
# model1.swa_model.module._conv_stem.register_forward_hook(hook)
# model2.swa_model.module._conv_stem.register_forward_hook(hook)
# #
# # # print(list(model.swa_model.parameters()))
# # # LivenessDetectorWrapper.export_model('../model/last/artifacts/last.ckpt', name='last', image_width=144)
# in_ = torch.rand(1, 5, 144, 144)
# print(model1.swa_model(in_))
# print(model2.swa_model(in_))