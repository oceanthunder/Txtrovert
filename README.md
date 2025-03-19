# Txtrovert   
**These reviews have a lot to say.**  

##  Overview  
Txtrovert is a sentiment analysis model trained on the [IMDB Movies Dataset (50k+ reviews)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). It classifies reviews as **positive** or **negative** using **logistic regression**. 
It is deployed on [Streamlit Community Cloud](txtrovert.streamlit.app) and [Huggingface Spaces](https://huggingface.co/spaces/sahilgarje/txtrovert).

## How It Works  
1. The input review is transformed using a **TF-IDF vectorizer**.  
2. The logistic regression model predicts whether the sentiment is **positive** or **negative**.  
3. The result is displayed with delightful animations!  

## Files  
- `txtrovertModel.pkl` - Trained sentiment analysis model  
- `tfidfVectorizer.pkl` - Pretrained TF-IDF vectorizer  
- `app.py` - Gradio app for interactive sentiment analysis  
- `requirements.txt` - Dependencies for this repo 
- `IMDBDataset.csv` - The review dataset from Kaggle
- `admin.jsonl` - Converted the dataset to jsonl format, because why not
- `predict.py` - The predictions without interfacing, if you're into that
- `train.py` - The logistic regression model that created vecorizer and pkl files
- `predictedReviews.jsonl` - An example output of the predict.py file
