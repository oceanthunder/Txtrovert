import json
import joblib
import gradio as gr
import numpy as np

def load_model():
    model = joblib.load("txtrovertModel.pkl")
    vectorizer = joblib.load("tfidfVectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

def predict_sentiment(text):
    X_input = vectorizer.transform([text])
    prediction = model.predict(X_input)[0]
    label = "Positive" if prediction == 1 else "Negative"
    return label

def get_animation(label):
    if label == "Positive":
        return "<div class='positive-anim'>(☉‿☉) Hoh? A man of culture! Truly, this is the work of an ally of justice! ☆</div>"
    else:
        return "<div class='negative-anim'>(╬ಠ益ಠ) MUDA MUDA MUDA! Reality has rejected your hopes… WRYYYYY!</div>"


def predict_with_animation(text):
    label = predict_sentiment(text)
    return label, get_animation(label)

demo = gr.Interface(
    fn=predict_with_animation,
    inputs=gr.Textbox(placeholder="Enter your review here...", label="Review Input"),
    outputs=[gr.Text(label="Sentiment"), gr.HTML()],
    title="Txtrovert",
    description="These reviews have a lot to say.",
    theme="default",
    examples=[
        ["The movie was an absolute masterpiece!"],
        ["Worst experience ever. Totally regret it!"],
        ["It was okay, nothing special but not bad either."],
    ],
    allow_flagging="never",
    css="""
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body { 
            background-color: #ffd700; 
            color: #ffffff; 
            font-family: 'Roboto', sans-serif; 
            text-align: center;
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
        }
        .gradio-container { 
            width: 95vw;
            max-width: 1400px;
            height: 90vh;
            padding: 40px;
            background: #222;
            border-radius: 12px;
            box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        input, textarea {
            background: #333;
            color: #fff;
            border: 1px solid #fff;
            border-radius: 8px;
            padding: 15px;
            width: 100%;
            font-size: clamp(14px, 2vw, 18px);
            resize: none;
        }
        button {
            background: #ffcc00;
            color: #222;
            font-weight: bold;
            border: none;
            padding: 15px 25px;
            border-radius: 8px;
            cursor: pointer;
            font-size: clamp(14px, 2vw, 18px);
            transition: background 0.3s;
        }
        button:hover {
            background: #e6b800;
        }
        .positive-anim {
            font-size: 20px;
            color: white;
            animation: happy-bounce 1s infinite alternate;
        }
        .negative-anim {
            font-size: 20px;
            color: white;
            animation: sad-fade 1s infinite alternate;
        }
        @keyframes happy-bounce {
            0% { transform: translateY(0); }
            80% { transform: translateY(-4px); }
        }
        @keyframes sad-fade {
            0% { opacity: 1; }
            100% { opacity: 0.5; }
        }
    """
)

demo.launch()
