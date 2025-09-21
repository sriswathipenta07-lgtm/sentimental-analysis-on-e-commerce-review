from flask import Flask, render_template, request, jsonify
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # type: ignore
import os
import pandas as pd
app = Flask(__name__)

model = joblib.load("D:\\projects\\e commerce sentiment analysis\\sentiment analysis on e commerce\\Xgboost.joblib")
 
tfidf_vectorizer = joblib.load('D:\\projects\\e commerce sentiment analysis\\sentiment analysis on e commerce\\tfidf_vectorizer.joblib')

analyzer = SentimentIntensityAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    

    predicted_probability = analyzer.polarity_scores(text)['compound']
    xgb_prediction = "Positive" if predicted_probability > 0 else "Negative" if predicted_probability < 0 else "Neutral"

    result = {
        'vader_sentiment': xgb_prediction,
        'probability_score': predicted_probability
    }
    return jsonify(result)
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded file
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    try:
        # Read Excel file
        reviews_df = pd.read_excel(filepath)
        if 'Review' not in reviews_df.columns:
            return jsonify({"error": "The Excel file must have a 'Review' column"}), 400

        # Perform sentiment analysis
        sentiment_results = []
        for review in reviews_df['Review']:
            if pd.isnull(review):
                sentiment_results.append({"review": "", "score": 0, "sentiment": "Neutral"})
                continue
            
            score = analyzer.polarity_scores(review)['compound']
            sentiment = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
            sentiment_results.append({"review": review, "score": score, "sentiment": sentiment})

        # Add results to the DataFrame
        reviews_df['Sentiment Score'] = [result['score'] for result in sentiment_results]
        reviews_df['Sentiment'] = [result['sentiment'] for result in sentiment_results]

        # Calculate summary
        positive_count = len(reviews_df[reviews_df['Sentiment'] == 'Positive'])
        negative_count = len(reviews_df[reviews_df['Sentiment'] == 'Negative'])
        neutral_count = len(reviews_df[reviews_df['Sentiment'] == 'Neutral'])
        total_reviews = len(reviews_df)
        overall_performance = "Good" if positive_count > negative_count else "Bad"

        # Save analyzed file
        output_filepath = os.path.join("uploads", "analyzed_reviews.xlsx")
        reviews_df.to_excel(output_filepath, index=False)

        # Render results on the webpage
        return render_template(
            'results.html',
            total_reviews=total_reviews,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            overall_performance=overall_performance,
            table_data=reviews_df.to_dict(orient='records'),
            download_link=output_filepath
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
