from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the VGGNet model
model_path = os.path.join('models', 'my_model.h5')
model = load_model(model_path)

# Function to process the uploaded image and make predictions
def process_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    # Check if the file name is empty
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded file
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the uploaded image
        img_array = process_image(file_path, target_size=(224, 224))

        # Make predictions using the model
        prediction = model.predict(img_array)
        
        # Pass the prediction result to the result.html page
        return redirect(url_for('show_result', filename=filename, prediction=prediction))

@app.route('/result/<filename>')
def show_result(filename):
    prediction = request.args.get('prediction')
    return render_template('result.html', filename=filename, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
