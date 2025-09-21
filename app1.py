from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sr # type: ignore
from pydub import AudioSegment
analyzer = SentimentIntensityAnalyzer()
app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/analyze_text')
def analyze_text():
    return render_template('analyze_text.html')

@app.route('/analyze_excel')
def analyze_excel():
    return render_template('analyze_excel2.html')

@app.route('/analyze_speech')
def analyze_speech():
    return render_template('analyze_speech.html')

@app.route('/predict_text', methods=['POST'])
def predict_text():
    text = request.form['text']
    predicted_probability = analyzer.polarity_scores(text)['compound']
    xgb_prediction = "Positive" if predicted_probability > 0 else "Negative" if predicted_probability < 0 else "Neutral"

    result = {
        'vader_sentiment': xgb_prediction,
        'probability_score': predicted_probability
    }
    return jsonify(result)

@app.route('/predict_excel', methods=['POST'])
def predict_excel():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    try:
        df = pd.read_excel(filepath)
        if 'Review' not in df.columns:
            return jsonify({'error': "The Excel file must have a 'Review' column"}), 400

        df['Sentiment Score'] = df['Review'].apply(lambda x: analyzer.polarity_scores(x)['compound'] if pd.notnull(x) else 0)
        df['Sentiment'] = df['Sentiment Score'].apply(lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral")

        results = df.to_dict(orient='records')
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_speech_data', methods=['POST'])
def analyze_speech_data():
    return jsonify({"message": "Speech input analysis is not implemented yet. This is a placeholder."})
@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    try:
 
        # Convert audio to text
        recognizer = sr.Recognizer()
        with sr.AudioFile(filepath) as source:
            audio_data = recognizer.record(source)
            recognized_text = recognizer.recognize_google(audio_data)

        # Perform sentiment analysis
        sentiment_score = analyzer.polarity_scores(recognized_text)['compound']
        sentiment = (
            "Positive" if sentiment_score > 0 else
            "Negative" if sentiment_score < 0 else
            "Neutral"
        )

        return jsonify({
            'text': recognized_text,
            'sentiment': sentiment,
            'sentiment_score': sentiment_score
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
