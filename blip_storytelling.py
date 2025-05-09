import requests
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Path to the downloaded image (replace with your actual file path)
image_path = r"C:\Users\ashish\Downloads\parrots.png"# Use raw string to avoid unicode error

# Load the image from local storage
try:
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess the image and generate a story
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs, max_length=50)
    story = processor.decode(out[0], skip_special_tokens=True)

    # Output the result
    print("Generated Story:", story)

except Exception as e:
    print(f"Failed to process image: {e}")
