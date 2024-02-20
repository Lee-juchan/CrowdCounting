from flask import Flask, request, send
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import io

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    image = Image.open(file.stream)  # Open the image file
    transform = transforms.ToTensor()
    d = transform(image)  # Convert the image to tensor

    output = model(d.unsqueeze(0)).detach()[0]  # Apply the model
    plt.imshow(output)  # Create a plot

    # Convert the plot to a PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)

    # Save the image to a temporary file and send it
    im.save('output.png')
    return send_file('output.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(port=5000)
