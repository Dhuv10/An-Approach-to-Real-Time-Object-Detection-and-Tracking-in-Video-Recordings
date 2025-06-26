import cv2
import torch
import torchvision.models.detection as model
import torchvision.transforms as T
import numpy as np

# Load the pre-trained Faster R-CNN model
model = model.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load class labels from a text file
classlabels = []
file_name = 'C:/Users/dell/Downloads/labels.txt'
with open(file_name, 'rt') as fpt:
    classlabels = fpt.read().rstrip('\n').split('\n')

# Set input size and scaling for the model
input_width, input_height = 800, 800  # Adjust this based on your model's requirements

# Open a video capture object
cap = cv2.VideoCapture('C:/Users/dell/Downloads/F1AutobahnFullHD.m4v')

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Cannot open the video')

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()

    # Convert the frame to a PyTorch tensor and preprocess for Faster R-CNN
    image_tensor = torch.from_numpy(frame.transpose((2, 0, 1))).float()
    image_tensor /= 255.0  # Normalize pixel values to [0, 1]
    image_tensor = T.functional.normalize(image_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Apply standard ImageNet normalization
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # Perform object detection with Faster R-CNN
    with torch.no_grad():
        predictions = model(image_tensor)

    for prediction in predictions[0]['boxes']:
        left, top, right, bottom = map(int, prediction.tolist())

        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

    for idx, label_id in enumerate(predictions[0]['labels']):
        class_name = classlabels[label_id - 1]  # Adjust for 0-based indexing
        score = predictions[0]['scores'][idx].item()

        # Draw label
        cv2.putText(frame, f'{class_name}: {score:.2f}', (left, top - 10), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)

    cv2.imshow('Object_Detected', frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
