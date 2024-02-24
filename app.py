from flask import Flask,request,jsonify
import numpy as np
import pickle

model = pickle.load(open('food_spoilage_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict', methods=['POST'])
def predict():
    methane = request.form.get('methane')
    temperature = request.form.get('temperature')
    humidity = request.form.get('humidity')

    # result={'methane':methane,'temperature':temperature,'humidity':humidity}
    input_query = np.array([[methane,temperature,humidity]])

    result = model.predict(input_query)[0]

    return jsonify({'outcome':str(result)})
    #return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
