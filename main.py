import pandas as pd
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Class Definitions
class VirtualUser:
    def __init__(self, id, susceptibility_to_fake_news):
        self.id = id
        self.susceptibility_to_fake_news = susceptibility_to_fake_news


class EnhancedVirtualUser(VirtualUser):
    def __init__(self, id, susceptibility_to_fake_news, interests):
        super().__init__(id, susceptibility_to_fake_news)
        self.interests = interests


class DynamicVirtualUser(EnhancedVirtualUser):
    def __init__(self, id, susceptibility_to_fake_news, interests):
        super().__init__(id, susceptibility_to_fake_news, interests)
        self.read_articles = []

    def read_article(self, article_id):
        self.read_articles.append(article_id)


# Function Definitions
def create_dynamic_virtual_users(num_users, interests):
    dynamic_users = []
    for i in range(num_users):
        susceptibility = random.uniform(0, 1)
        user_interests = random.sample(interests, k=random.randint(1, len(interests)))
        dynamic_users.append(DynamicVirtualUser(i, susceptibility, user_interests))
    return dynamic_users


def simulate_user_behavior(users, df, read_prob=0.1, share_prob_base=0.05, comment_prob_base=0.03):
    interactions = defaultdict(list)
    for user in users:
        share_prob = share_prob_base + user.susceptibility_to_fake_news * 0.1
        comment_prob = comment_prob_base + user.susceptibility_to_fake_news * 0.05
        articles_read = df.sample(frac=read_prob)
        for article_id in articles_read.index:
            user.read_article(article_id)
            interactions["reads"].append((user.id, article_id))
            if random.random() < share_prob:
                interactions["shares"].append((user.id, article_id))
            if random.random() < comment_prob:
                interactions["comments"].append((user.id, article_id))
    return interactions


# Load the Dataset
file_path = 'news_articles.csv'  # Adjust the path according to your file location
df = pd.read_csv(file_path)
df = df.dropna(subset=['text'])

# Define a set of interests
interests = ["politics", "technology", "health", "entertainment", "sports"]

# Create dynamic virtual users
num_dynamic_users = 100
dynamic_virtual_users = create_dynamic_virtual_users(num_dynamic_users, interests)

# Simulate user behavior
dynamic_user_interactions = simulate_user_behavior(dynamic_virtual_users, df)

# Machine Learning Model Implementation
X = df['text']  # Feature for fake news detection
y = df['label'].apply(lambda x: 1 if x == 'fake' else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = RandomForestClassifier()
model.fit(X_train_vec, y_train)

# Model Evaluation
y_pred = model.predict(X_test_vec)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nModel Classification Report:\n", classification_report(y_test, y_pred))

# Personalized Fake News Detection and User Data Display
for user in dynamic_virtual_users:
    user_read_articles = df.loc[user.read_articles]
    print(f"\nUser {user.id} (Susceptibility: {user.susceptibility_to_fake_news:.2f}):")
    print("Interactions:")

    for article_id, article in user_read_articles.iterrows():
        vectorized_text = vectorizer.transform([article['text']])
        prediction = model.predict(vectorized_text)[0]

        # Display the user interaction with each article
        article_title = article['title'] if 'title' in article else "N/A"
        article_label = "Fake" if prediction == 1 else "Real"
        print(f" - Article: {article_title} | Classified as: {article_label}")
