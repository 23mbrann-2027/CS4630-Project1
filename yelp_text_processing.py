from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from tqdm import tqdm


def text_processing(df):
    tqdm.pandas()
    df = pd.read_parquet("data/processed/cleaned_reviews.parquet", engine = "pyarrow")

    print(df.head())

    analyzer = SentimentIntensityAnalyzer()

    df["sentiment_score"] = df["text"].progress_apply(
        lambda text: analyzer.polarity_scores(text)["compound"]
    )

    def classifier(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else: 
            return "neutral"
        
    df["sentiment_label"] = df["sentiment_score"].apply(classifier)

    return df

def run_sentiment_pipeline(input_path, output_path):
    df = pd.read_parquet(
        input_path,
        engine = "pyarrow"
    )

    df = text_processing(df)

    df.to_parquet(
        output_path,
        engine = "pyarrow",
        compression = "snappy",
        index = False
    )

    print("Senitiment analysis complete")


if __name__ == "__main__":

    input_path = "data/processed/cleaned_reviews.parquet"
    output_path = "data/processed/reviews_with_sentiment.parquet"


    run_sentiment_pipeline(
        input_path,
        output_path
    )