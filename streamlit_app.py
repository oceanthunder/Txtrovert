import streamlit as st
import joblib
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

st.markdown("""
    <style>
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
            min-height: 100vh;
        }
        .stApp { 
            width: 90%;
            max-width: 1200px;
            min-width: 300px;
            padding: 5%;
            background: #222;
            border-radius: 12px;
            box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            margin: auto;
        }
        .stTextArea textarea {
            background: #333 !important;
            color: #fff !important;
            border: 1px solid #fff !important;
            border-radius: 8px !important;
            padding: 12px !important;
            width: 100% !important;
            font-size: 16px !important;
            resize: none !important;
        }
        .stButton button {
            background: #ffcc00 !important;
            color: #222 !important;
            font-weight: bold !important;
            border: none !important;
            padding: 12px 20px !important;
            border-radius: 8px !important;
            cursor: pointer !important;
            font-size: 16px !important;
            transition: background 0.3s !important;
        }
        .stButton button:hover {
            background: #e6b800 !important;
        }

        /* Responsive Design */
        @media screen and (max-width: 1024px) {
            .stApp {
                width: 95%;
                padding: 4%;
            }
        }
        @media screen and (max-width: 600px) {
            .stApp {
                width: 95%;
                padding: 20px;
            }
            .stTextArea textarea {
                font-size: 14px !important;
                padding: 10px !important;
            }
            .stButton button {
                font-size: 14px !important;
                padding: 10px 15px !important;
            }
        }

        /* Happy/Sad Animations */
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
        .predict-button{
        background-color: black !important;
        color: white !important;
        border-radius: 10px !important;
        font-size: 16px !important;
        font-weight: bold !important;
        padding: 10px 15px !important;
        border: 2px solid white !important;
        transition: all 0.3s ease-in-out;
    }
        @keyframes sad-fade {
            0% { opacity: 1; }
            100% { opacity: 0.5; }
        }
    </style>
""", unsafe_allow_html=True)



if "text_input" not in st.session_state:
    st.session_state.text_input = ""

def update_text_input(example_text):
    st.session_state.text_input = example_text

st.title("Txtrovert")
st.markdown("These reviews have a lot to say.")

text_input = st.text_area(
    "Review Input",
    placeholder="Enter your review here...",
    height=100,
    key="text_input"
)

examples = [
    "The movie was an absolute masterpiece!",
    "Worst experience ever. Totally regret it!",
    "It was okay, nothing special but not bad either."
]

st.markdown("**Try an example:**")
col1, col2, col3 = st.columns(3)
with col1:
    st.button(
        examples[0],
        key="example1",
        on_click=update_text_input,
        args=(examples[0],),
        help="Click to use this example"
    )
with col2:
    st.button(
        examples[1],
        key="example2",
        on_click=update_text_input,
        args=(examples[1],),
        help="Click to use this example"
    )
with col3:
    st.button(
        examples[2],
        key="example3",
        on_click=update_text_input,
        args=(examples[2],),
        help="Click to use this example"
    )

if st.markdown('<button class="predict-button">Predict Sentiment</button>', unsafe_allow_html=True):
    if st.session_state.text_input:
        label, animation = predict_with_animation(st.session_state.text_input)
        st.markdown(f"""
    <p style='font-size:24px; font-weight:bold; text-align:center;'>
        Sentiment: {label}
    </p>
""", unsafe_allow_html=True)

        st.markdown(animation, unsafe_allow_html=True)
    else:
        st.warning("Please enter a review or select an example.")
