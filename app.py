import tempfile

import librosa
import numpy as np
import redis
from flask import Flask, jsonify, render_template, request
from redis.commands.search.query import Query

redis_client = redis.Redis()
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/get_recommendations", methods=['POST'])
def get_recommendations():
    tmp = tempfile.NamedTemporaryFile(delete=True)
    request.files.get("music_file").save(tmp)

    y, sr = librosa.load(tmp.name)
    feature = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=200)
    vector = np.mean(feature.T, axis=0)
    vector = vector.astype(np.float32).tobytes()

    q = Query(
        f"(*)=>[KNN 3 @vec $vec_param as vector_score]"
    ).sort_by("vector_score").return_fields("url", "vector_score").dialect(2)

    params = {
        "vec_param": vector
    }

    results = []
    query_results = redis_client.ft("idx:music").search(query=q, query_params=params)

    for result in query_results.docs:
        results.append(result.url)

    tmp.close()
    return jsonify({
        "results": results
    })

if __name__ == '__main__':
    app.run(debug=True)
