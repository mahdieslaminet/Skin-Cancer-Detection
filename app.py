from flask import Flask, render_template, request
from PIL import Image
import torch
from torchvision import transforms, models
import os
import base64
import io
app = Flask(__name__)
#
# # مسیر مدل آموزش دیده (فرض کنیم اسم فایل skin_cancer_model.pth است)
MODEL_PATH = "skin_cancer_model.pth"
LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # کلاس‌ها

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, len(LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

LABELS_FA = {
    "akiec": "کراتوز اکتینیک (ضایعه پیش‌سرطانی)",
    "bcc": "سرطان سلول بازال",
    "bkl": "ضایعه خوش‌خیم کراتوزی",
    "df": "درماتوفیبروما (توده خوش‌خیم)",
    "mel": "ملانوما (خطرناک‌ترین نوع سرطان پوست)",
    "nv": "خال ملانوسیتی (خال معمولی)",
    "vasc": "ضایعه عروقی"
}

@app.route("/predict", methods=["POST"])
def predict():
    # if "file" not in request.files:
    #     return "فایلی آپلود نشده است", 400
    # file = request.files["file"]
    image_base64 = request.form.get("image_base64")

    if not image_base64:
        return "تصویری ارسال نشده است", 400

    # decode base64
    image_bytes = base64.b64decode(image_base64.split(",")[1])
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)  # محاسبه احتمال هر کلاس
        pred = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, pred].item() * 100  # درصد اطمینان

        label_en = LABELS[pred]
        result_fa = LABELS_FA[label_en]

    # تعیین خطر (UX)
    if label_en in ["mel", "bcc", "akiec"]:
        risk = "high"
    else:
        risk = "low"

    # ارسال result و confidence به HTML
    return render_template("index.html", result=result_fa, risk=risk, confidence=round(confidence, 2),image_base64=image_base64)



if __name__ == "__main__":
    app.run(debug=True)
