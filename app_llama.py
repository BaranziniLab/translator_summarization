from flask import Flask, Response, jsonify, request
import os
import json
import requests
from utility import *

app = Flask(__name__)

system_prompt = HIGH_LEVEL_SUMMARY_SYSTEM_PROMPT_WITHOUT_JSON_RESPONSE

# Llama API configuration
URI = os.environ.get('LLAMA_URI')
API_KEY = os.environ.get('LLAMA_KEY')

@app.route('/', methods=['GET'])
def home():
    endpoints = sorted(r.rule for r in app.url_map.iter_rules())
    return jsonify(endpoints=endpoints)

@app.route("/summary", methods=["POST"])
def summary():
    if not API_KEY:
        return jsonify({
            "error": "Server configuration error.",
            "message": "Llama API key is not configured in the server."
        }), 500

    data = request.json
    if not isinstance(data, dict) or "query" not in data or "results" not in data:
        return jsonify(error="Invalid request format. Must contain 'query' and 'results' keys."), 400

    prompt = get_token_limited_prompt(data, system_prompt)

    def generate():
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "stop": ["<|endoftext|>", "End of Response"],
            "presence_penalty": 0.5,
            "stream": True
        }

        with requests.post(URI, headers=headers, json=data, stream=True) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            json_str = line.decode('utf-8')
                            if json_str.startswith('data: '):
                                json_str = json_str[len('data: '):]
                            chunk = json.loads(json_str)

                            choices = chunk.get('choices', [])
                            if choices:
                                delta = choices[0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except (json.JSONDecodeError, IndexError, KeyError):
                            continue
            else:
                yield f"Error: Request failed with status code {response.status_code}"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000)