from flask import Flask, request, jsonify, render_template
from api import process_user_query

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_input = request.get_json().get('query')

    # Call the function to process the user query
    result = process_user_query(user_input)
    return jsonify(result)


if __name__ == '__main__':
    # Run the app, you can access it at http://localhost:5000/
    app.run(host='0.0.0.0', port=5000, debug=True)
