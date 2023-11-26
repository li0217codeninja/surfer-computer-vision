from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model 
results = model.train(data='yolo1pt1.yaml', epochs=100)