from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
from fastai2.vision.all import *
# Initialize the Flask application
app = Flask(__name__)
inf_model = load_learner("export.pkl")
# route http posts to this method
@app.route('/', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        r = request
        # convert string of image data to uint8
        nparr = np.fromstring(r.data, np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # do some fancy processing here....
        
        
        pred = inf_model.predict(img);
        print(pred[0])
        print(pred[1])
        print(pred[2])
        # build a response dict to send back to client
        response = {'Main Prediction': int(pred[0]), 'Accuracies': {'0': float(pred[2][0]), '1': float(pred[2][1]), '2': float(pred[2][2]), '3': float(pred[2][3]), '4': float(pred[2][4]), '5': float(pred[2][5]), '6': float(pred[2][6]), '7': float(pred[2][7]), '8': float(pred[2][8]), '9': float(pred[2][9])},
                    }
        # encode response using jsonpickle
        response_pickled = jsonpickle.encode(response)

        return Response(response=response_pickled, status=200, mimetype="application/json")
    elif request.method == 'GET':
        
        return "<h1> This is the Penman API </h1>"


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
