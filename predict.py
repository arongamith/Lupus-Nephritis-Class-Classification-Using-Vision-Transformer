import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageTk
from tkinter import Tk, Label, Frame, Canvas, Scrollbar, NW, BOTH, LEFT, RIGHT, Y

# Import your custom classes/variables
from model import ViTClassifier
from transforms import val_transforms  # Ensure this is in your transforms.py

# ----- SETTINGS -----
test_folder = "data/test"  
class_names = [
    "diffuse_proliferative",
    "membranous_pattern",
    "mesangial_hypercellularity",
    "minimal_changes"
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Predicting on: {device}")

# ----- LOAD MODEL -----
# num_classes must match your training (4)
model = ViTClassifier(num_classes=len(class_names))

# Load the weights saved from your training script
try:
    model.load_state_dict(torch.load("vit_kidney_classifier.pth", map_location=device))
    print("Weights loaded successfully.")
except FileNotFoundError:
    print("Error: vit_kidney_classifier.pth not found. Train the model first.")

model.to(device)
model.eval()

# ----- FUNCTION TO PREDICT -----
def predict(image_path):
    # Open and ensure RGB (removes alpha channels if present)
    img = Image.open(image_path).convert("RGB")
    
    # Use the same transforms as used in validation
    # (Resize 224, ToTensor, Normalize)
    x = val_transforms(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(x)
        # Convert raw logits to probabilities
        probs = F.softmax(outputs, dim=1)
        
        # Get the index of the highest probability
        conf, pred = torch.max(probs, dim=1)
        
        pred_idx = pred.item()
        confidence_score = conf.item() * 100
        
    return class_names[pred_idx], confidence_score, img

# ----- TKINTER GUI -----
root = Tk()
root.title("Renal Biopsy Vision Transformer Classifier")
root.geometry("500x800")

# Scrollable canvas setup
canvas = Canvas(root)
scrollbar = Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor=NW)
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side=LEFT, fill=BOTH, expand=True)
scrollbar.pack(side=RIGHT, fill=Y)

# Process images in the test folder
if not os.path.exists(test_folder):
    print(f"Directory {test_folder} does not exist.")
else:
    image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        Label(scrollable_frame, text="No images found in test folder.", fg="red").pack()

    for img_file in image_files:
        img_path = os.path.join(test_folder, img_file)
        
        try:
            pred_class, confidence, img = predict(img_path)

            # Create a UI Container for each result
            result_container = Frame(scrollable_frame, bd=2, relief="groove", pady=10)
            result_container.pack(fill="x", padx=10, pady=5)

            # Display Image
            img_disp = img.resize((200, 200)) # Small preview
            img_tk = ImageTk.PhotoImage(img_disp)
            
            img_label = Label(result_container, image=img_tk)
            img_label.image = img_tk  # Persistent reference
            img_label.pack()

            # Display Prediction and Confidence
            # Color code: High confidence = Green, Low = Orange/Red
            text_color = "green" if confidence > 80 else "orange" if confidence > 50 else "red"
            
            result_text = f"File: {img_file}\nPred: {pred_class}\nConf: {confidence:.2f}%"
            info_label = Label(result_container, text=result_text, font=("Helvetica", 12, "bold"), fg=text_color)
            info_label.pack()
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

root.mainloop()