from flask import Flask, request, jsonify
from model import get_best_places

app = Flask(__name__)


@app.route("/recommendations", methods=["GET"])
def get_recommendations():
    request_data = request.json

    bucket_list = request_data.get('bucket_list', [])
    preferred_activities = request_data.get('preferred_activities', [])

    return jsonify(get_best_places(preferred_activities, bucket_list))


if __name__ == "__main__":
    app.run(debug=True)