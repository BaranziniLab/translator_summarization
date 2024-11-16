from flask import Flask, Response, jsonify, request
from webargs.flaskparser import use_kwargs
from webargs.fields import Str
from webargs import fields, validate
from openai import OpenAI, OpenAIError
from utility import *

app = Flask(__name__)

system_prompt = HIGH_LEVEL_SUMMARY_SYSTEM_PROMPT_WITHOUT_JSON_RESPONSE

try:
    client = OpenAI(api_key = os.environ.get('API_KEY'))
except OpenAIError as e:
    error_message = str(e)
    app.logger.error(f"OpenAI API Error: {error_message}")
    client = None

    
@app.route('/', methods=['GET'])
def home():
    endpoints = sorted(r.rule for r in app.url_map.iter_rules())
    return jsonify(
        endpoints=endpoints
    )

@app.route("/summary", methods=["POST"])
def summary():
    if not client:
        return jsonify({
            "error": "Server configuration error.",
            "message": "OpenAI API key is not configured in the server."
        }), 500
    data = request.json
    if not isinstance(data, dict) or "query" not in data or "results" not in data:
        return jsonify(error="Invalid request format. Must contain 'query' and 'results' keys."), 400

    prompt = get_token_limited_prompt(data, system_prompt)
    
    def generate():
        stream = client.chat.completions.create(        
            temperature=TEMPERATURE,
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000)