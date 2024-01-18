from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import matplotlib.pyplot as plt

img = read_image("Chihuahua_dog_1.jpg")

weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights)
model.eval()

preprocess = weights.transforms()

batch = preprocess(img).unsqueeze(0)

prediction = model(batch)["out"]
normalized_masks = prediction.softmax(dim=1)
class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
mask = normalized_masks[0, class_to_idx["dog"]]

plt.imshow(mask.detach().numpy(), cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()
