from flask import Flask, request, render_template, redirect, url_for
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import os
from dotenv import load_dotenv

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

load_dotenv()
prediction_endpoint = "https://fruitforyou-prediction.cognitiveservices.azure.com/"
prediction_key = "2c7e18296d9741e38ca2ae9ceeaa263e"
project_id = "97805f87-648b-48f4-b3f3-abd1a10b276a"
model_name = "calssify fruit"
iteration_name = "fruit calssifier"

credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
prediction_client = CustomVisionPredictionClient(endpoint=prediction_endpoint, credentials=credentials)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            with open(filepath, "rb") as image_data:
                results = prediction_client.classify_image(project_id, iteration_name, image_data.read())
                predictions = [(prediction.tag_name, prediction.probability) for prediction in results.predictions if prediction.probability > 0.5]
            os.remove(filepath)  # Clean up uploaded file
            return render_template('index.html', predictions=predictions)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
