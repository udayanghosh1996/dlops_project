import streamlit as st

if not hasattr(st, 'already_started_server'):
    # Hack the fact that Python modules (like st) only load once to
    # keep track of whether this file already ran.
    st.already_started_server = True

    st.write('''
        The first time this script executes it will run forever because it's
        running a Flask server.

        Just close this browser tab and open a new one to see your Streamlit
        app.
    ''')

    import os

    from flask import Flask, jsonify, request, render_template
    from predict_image import *
    import numpy as np
    import cv2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    app = Flask(__name__)

    @app.route('/')
    def home():
        return render_template('home.html')


    @app.route('/prediction', methods=['GET', 'POST'])
    def prediction():
        '''if request.get_json() is not None:
            json_ = request.json
            pred = image_prediction(json_['image1'])
            if pred is not None:
                return jsonify(pred)'''

        if request.data is not None:
            npar = np.fromstring(request.data, np.uint8)
            img = cv2.imdecode(npar, cv2.IMREAD_COLOR)
            #path = os.path.join(os.path.join(os.getcwd(), '../webpages'), 'img.jpg')
            #imge = cv2.imread(path)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = image_prediction(image)
            return jsonify(pred)


    @app.route('/dummy', methods=['GET', 'POST'])
    def dummy():
        if request.data is not None:
            prob = np.random.random_sample(size=10)
            clas = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
            dic = dict(zip(clas, prob))
            return dic


    if __name__ == '__main__':
        app.run(host='https://udayanghosh1996-dlops-project--simclrapp-9aqgnl.streamlit.app', debug=True)
x = st.slider('Pick a number')
st.write('You picked:', x)