import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

model = torchvision.models.detection.maskrcnn_resnet50_fpn(
    weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Error: File not found at {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        try:
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)
        except Exception as e:
            raise FileNotFoundError(f"❌ Error loading image: {e}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def preprocess_image(image):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    return transform(image).to(device)

def visualize_results(image, boxes, masks, scores, threshold=0.5):
    for i in range(len(boxes)):
        if scores[i] > threshold:
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            mask = masks[i][0] > 0.5
            color = np.random.randint(0, 255, (1, 3), dtype=int).tolist()[0]
            image[mask] = [color[0], color[1], color[2]]

            label_text = f"Score: {scores[i]:.2f}"
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

image_path = r"C:\Users\user\Desktop\INNOVATE\test.jpg"

try:
    image = load_image(image_path)
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model([input_tensor])

    output = output[0]
    boxes = output['boxes'].cpu().numpy()
    masks = output['masks'].cpu().numpy()
    scores = output['scores'].cpu().numpy()

    visualize_results(image, boxes, masks, scores)

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
