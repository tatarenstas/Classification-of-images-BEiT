from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image
import requests

url = 'https://media.istockphoto.com/id/155276879/photo/modern-passanger-airplane-flying-above-clouds.jpg?s=612x612&w=0&k=20&c=tjIs1P72CZyPvms_Li9b9pCFp_YCHRF3-8eeKVhJ_NE='
image = Image.open(requests.get(url, stream=True).raw)
image.save("image.jpg")
feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx]) #Predicted class: airplane, aeroplane, plane
