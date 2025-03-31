
# A very simple Flask Hello World app for you to get started with...

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = models.efficientnet_b1(weights=None)  # Initialize model
model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=7)  # Modify classifier for 7 classes
model.load_state_dict(torch.load("efficientnet_b1.pth", map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode

# Define class labels
class_labels = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions ', 'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions']


# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for EfficientNet
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Folder to store uploaded images
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Prediction function
# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Open image
    print(image.size)
    print(f"Image size: {image.size}, Mode: {image.mode}")
    image = transform(image).unsqueeze(0).to(device)  # Preprocess

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # Convert to probabilities
        predicted_class_index = torch.argmax(probabilities).item()  # Get highest probability class index

    predicted_class_name = class_labels[predicted_class_index]  # Get class label
    return predicted_class_name, probabilities[predicted_class_index].item()

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Predict
            predicted_class, confidence = predict_image(file_path)

            return render_template("index.html", filename=file.filename, prediction=predicted_class, confidence=float(confidence))

            #return render_template("index.html", filename=file.filename, prediction='AAA', confidence=float('0.895'))

    return render_template("index.html", filename=None)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for("static", filename=f"uploads/{filename}"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)