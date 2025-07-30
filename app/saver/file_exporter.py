def save_to_csv(df, path="recommendations.csv"):
    df.to_csv(path, index=False)