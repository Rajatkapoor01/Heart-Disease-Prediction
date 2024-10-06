from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

#app = Flask(__name__)
app = Flask(__name__, template_folder='templates')

# Set the upload folder
app.config['UPLOAD_FOLDER'] = 'static/image'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit the upload size to 16MB

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the pre-trained model
filename = 'heart_disease_model.sav'
model = pickle.load(open(filename, 'rb'))

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from form

        age = float(request.form['age'])
        cp = float(request.form['cp'])
        chol = float(request.form['chol'])
        thalach = float(request.form['thalach'])
        oldpeak = float(request.form['oldpeak'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])

        # Format the input data into a numpy array
        input_data = np.array([[age, cp, chol, thalach, oldpeak, ca, thal]])

        # Make prediction using the loaded model
        prediction = model.predict(input_data)
        print(prediction)
        # Map prediction to result
        result = 'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease'
        #result=prediction
        print(result)

        return render_template('output.html', result=result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
