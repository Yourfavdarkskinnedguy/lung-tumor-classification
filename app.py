from flask import Flask, request, render_template
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import load_img
from tensorflow.keras.models import load_model
import numpy as np


app= Flask(__name__)


interpreter= tf.lite.Interpreter(os.path.join(os.getcwd(), 'converted_model.tflite'))
interpreter.allocate_tensors()


@app.route('/')
def home():
    
    return render_template('index.html')

def binarizer(file):
    with open(file, 'rb') as f:
        read_file= f.read()
    return read_file
        
    

@app.route('/predict', methods=['POST'])
def predict():
    
    file= request.files['mri_scan']
    print(type(file))
    
    temp_filepath = os.path.join(file.filename)
    file.save(temp_filepath)
    
                
    loaded_image= load_img(temp_filepath, target_size=(256,256))
    img_to_array= keras.preprocessing.image.img_to_array(loaded_image)
    img_to_array_expanded= np.expand_dims(img_to_array, axis=0)
    
    input_details= interpreter.get_input_details()
    output_details= interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img_to_array_expanded)
    
    interpreter.invoke()
    
    
    result = interpreter.get_tensor(output_details[0]['index'])
    result= np.argmax(result)
    class_name= ["Benign", "Malignant", "Healthy"]
    prediction= class_name[result]
    print(prediction)
    
 
    
    os.remove(temp_filepath)
    
    return render_template('index.html', prediction= prediction)
    

    
if __name__ == "__main__":
    app.run(debug=True)