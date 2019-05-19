from flask import Flask, render_template, request
from models.model import predict_maliciousness

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_score', methods=['POST'])
def get_score():
    if request.method == 'POST':
        mal_features = ['domain_age', 'entropy', 'suffix', 'designation', 'number_suffix', 'digits_percentage', 'specials']
        url = request.get_data().decode("utf-8")
        mal_url_output = predict_maliciousness(url, mal_features)
        result = mal_url_output[1]
    else:
        result = 'Empty'
    return result


@app.route('/home')
def home():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)