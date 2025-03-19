import json
import joblib
import numpy as np

model = joblib.load("txtrovertModel.pkl")
vectorizer = joblib.load("tfidfVectorizer.pkl")

new_reviews = [
    "This movie was absolutely amazing! The cinematography was breathtaking, and the performances were so compelling that I was completely immersed in the story from start to finish. I would definitely watch it again!",
    
    "This was the worst customer service experience I have ever had. Not only did they take forever to respond, but when they finally did, they were incredibly rude and unhelpful. I had to escalate the issue multiple times just to get a simple resolution. Never buying from them again!",
    
    "The product quality was decent; the material feels sturdy and durable. However, the delivery was significantly delayed, and the packaging was quite damaged when it arrived. It still works fine, but I expected better from a reputed brand.",
    
    "I absolutely loved this experience! From the moment I walked in, the ambiance was warm and welcoming. The staff was friendly, the service was prompt, and the food was absolutely delicious. Definitely coming back again soon!",
    
    "Horrible! I regret watching this film. The plot was all over the place, the characters lacked depth, and the dialogues felt forced. I kept waiting for it to get better, but it just got worse. Complete waste of time!",
    
    "Surprisingly, this turned out to be one of the most enjoyable purchases I’ve made. The craftsmanship is excellent, and the attention to detail is evident. I was skeptical at first, but now I’m glad I went for it. Would highly recommend!",
    
    "I was so disappointed with this book. The premise seemed interesting, but the execution was terrible. The pacing was inconsistent, the writing felt rushed, and the ending was incredibly predictable. I wouldn’t recommend it to anyone looking for a deep and engaging read.",
    
    "The resort was beyond expectations! The rooms were spotless, the view was breathtaking, and the staff went above and beyond to ensure we had a comfortable stay. The only downside was that some amenities were not available, but overall, it was a fantastic experience.",
    
    "Absolutely infuriating! I booked a flight through this airline, and they canceled it at the last minute without any proper explanation. The refund process has been a nightmare, and their customer support is nonexistent. I will never use their services again!",
    
    "The event was well-organized, and the speakers were truly insightful. I walked away feeling inspired and equipped with new knowledge. The only improvement I’d suggest is a better seating arrangement, as it was a bit cramped. Overall, a great experience!"
]


X_new = vectorizer.transform(new_reviews)

predictions = model.predict(X_new)

doccano_ready = [{"text": review, "label": ["positive" if pred == 1 else "negative"]} for review, pred in zip(new_reviews, predictions)]

with open("predictedReviews.jsonl", "w", encoding="utf-8") as f:
    for item in doccano_ready:
        f.write(json.dumps(item) + "\n")

print("Predicted labels saved to predictedReviews.jsonl. You can now do RLHF or some shit.")

