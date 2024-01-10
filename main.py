import random

import matplotlib
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

matplotlib.use('Agg')  # Dla środowisk bez GUI, lub 'TkAgg' dla środowisk z GUI


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
    def __init__(self, id, susceptibility_to_fake_news, interests, friends=None):
        super().__init__(id, susceptibility_to_fake_news, interests)
        self.read_articles = []
        self.friends = friends if friends is not None else []
        self.infection_time = None  # Track when the user shares the news

    def read_article(self, article_id, current_time):
        print(f"User {self.id} read article {article_id} at time {current_time}")
        self.read_articles.append(article_id)
        if self.infection_time is None:
            self.infection_time = current_time

    def post_article(self, article_id, current_time, network):
        print(f"User {self.id} posted article {article_id} at time {current_time}")
        self.read_article(article_id, current_time)
        for friend_id in self.friends:
            friend = network.nodes[friend_id]['user']
            if random.random() < 0.7:  # Higher chance for friends to read the posted article
                print(f"Friend {friend_id} of User {self.id} read article {article_id} posted by User {self.id}")
                friend.read_article(article_id, current_time)


# Function Definitions
def create_dynamic_virtual_users(num_users, interests):
    dynamic_users = []
    num_users = min(num_users, 30)
    for i in range(num_users):
        susceptibility = random.uniform(0, 1)
        user_interests = random.sample(interests, k=random.randint(1, len(interests)))
        dynamic_users.append(DynamicVirtualUser(i, susceptibility, user_interests))
    return dynamic_users

def assign_friends_to_users(users, max_friends=5):
    for user in users:
        potential_friends = [u for u in users if u.id != user.id]
        user.friends = random.sample(potential_friends, k=min(max_friends, len(potential_friends)))
        user.friends = [friend.id for friend in user.friends]  # Store only friend IDs


def create_social_network(users, average_friends=5):
    G = nx.Graph()
    for user in users:
        G.add_node(user.id, user=user)
        friends = random.sample([u.id for u in users if u.id != user.id], k=average_friends)
        for friend in friends:
            G.add_edge(user.id, friend)
    return G


def simulate_and_export_results(users, network, df, num_days=30):
    results = defaultdict(list)

    for day in range(1, num_days + 1):
        current_time = day
        for user in users:
            share_prob = 0.05 + user.susceptibility_to_fake_news * 0.1
            if random.random() < share_prob:
                article_id = df.sample(n=1).index[0]
                user.post_article(article_id, current_time, network)
                user.read_article(article_id, current_time)
                results[user.id].append(('read', article_id, day))
                for friend_id in network[user.id]:
                        friend = network.nodes[friend_id]['user']
                        if article_id not in friend.read_articles:
                            friend.read_article(article_id, current_time + 1)
                            results[friend_id].append(('received', article_id, day, user.id))

    # Zapis do CSV
    with open('simulation_results.csv', 'w') as file:
        file.write('User,Action,Article,Day,Origin\n')
        for user_id, interactions in results.items():
            for action, article_id, day, *origin in interactions:
                origin_id = origin[0] if origin else ''
                file.write(f"{user_id},{action},{article_id},{day},{origin_id}\n")

    return results


def plot_simulation_results(results):
    plt.figure(figsize=(10, 6))
    read_counts = defaultdict(int)
    share_counts = defaultdict(int)

    for interactions in results.values():
        for interaction in interactions:
            action = interaction[0]
            article_id = interaction[1]
            if action == 'read':
                read_counts[article_id] += 1
            elif action == 'received':
                share_counts[article_id] += 1

    # Przygotowanie danych do wykresu
    articles = list(set(read_counts.keys()).union(share_counts.keys()))
    read_freq = [read_counts[art] for art in articles]
    share_freq = [share_counts[art] for art in articles]

    # Rysowanie wykresu
    plt.bar(articles, read_freq, label='Read Count')
    plt.bar(articles, share_freq, bottom=read_freq, label='Share Count', alpha=0.5)
    plt.xlabel('Article ID')
    plt.ylabel('Frequency')
    plt.title('Article Read and Share Frequency')
    plt.legend()
    plt.savefig('results.jpg')


def main():
    # Load the Dataset
    file_path = 'news_articles.csv'
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['text'])

    # Define a set of interests
    interests = ["politics", "technology", "health", "entertainment", "sports"]

    # Create dynamic virtual users
    num_dynamic_users = 100
    dynamic_virtual_users = create_dynamic_virtual_users(num_dynamic_users, interests)
    assign_friends_to_users(dynamic_virtual_users)

    # Machine Learning Model Implementation
    X = df['text']
    y = df['label'].apply(lambda x: 1 if x == 'fake' else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = RandomForestClassifier()
    model.fit(X_train_vec, y_train)

    # Create a network for dynamic virtual users
    network = create_social_network(dynamic_virtual_users)

    # Symulacja i eksport wyników
    simulation_results = simulate_and_export_results(dynamic_virtual_users, network, df)

    # Generowanie wykresu
    plot_simulation_results(simulation_results)


if __name__ == '__main__':
    main()
