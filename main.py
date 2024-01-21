import random
from collections import defaultdict
import matplotlib
import networkx as nx
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from plots import plot_simulation_results, plot_infection_spread, plot_article_frequency_with_interests, \
    plot_infection_spread_and_percentage

matplotlib.use('Agg')


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
        self.infection_time = None

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
            if random.random() < 0.7:
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


def simulate_and_export_results(
        users,
        network,
        df,
        iteration,
        share_probability=0.7,
        is_probability_changed=False,
        num_days=30):
    results = defaultdict(list)

    for day in range(1, num_days + 1):
        current_time = day
        for user in users:
            if random.random() < share_probability:
                article_id = df.sample(n=1).index[0]
                user.post_article(article_id, current_time, network)
                user.read_article(article_id, current_time)
                results[user.id].append(('read', article_id, day))
                for friend_id in network[user.id]:
                    friend = network.nodes[friend_id]['user']
                    if article_id not in friend.read_articles:
                        friend.read_article(article_id, current_time + 1)
                        results[friend_id].append(('received', article_id, day, user.id))
                        results[friend_id].append(('read', article_id, day, user.id))

    filename = str(iteration) + '_simulation_results.csv'
    if is_probability_changed:
        filename = str(share_probability) + '_prob_simulation_results.csv'
    with open(filename, 'w') as file:
        file.write('User,Action,Article,Day,Origin\n')
        for user_id, interactions in results.items():
            for action, article_id, day, *origin in interactions:
                origin_id = origin[0] if origin else ''
                file.write(f"{user_id},{action},{article_id},{day},{origin_id}\n")
    total_read = sum(1 for interactions in results.values() for action in interactions if action[0] == 'read')
    total_received = sum(1 for interactions in results.values() for action in interactions if action[0] == 'received')
    print(f"Total 'read' actions: {total_read}")
    print(f"Total 'received' actions: {total_received}")
    return results


def main():
    # Wczytajmy dataset
    file_path = 'news_articles.csv'
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['text'])

    # Dodaj sztucznie kolumnę 'interests'
    interests = ["politics", "technology", "health", "entertainment", "sports"]
    df['interests'] = [random.choice(interests) for _ in range(len(df))]

    # Utwórz kolumnę 'id' jako unikalny identyfikator dla każdego artykułu
    df['id'] = range(len(df))

    # Menu wyboru
    print("Wybierz opcję:")
    print("1: Uruchomienie wielu symulacji z różnymi prawdopodobieństwami.")
    print("2: Uruchomienie jednej symulacji z określonym prawdopodobieństwem.")
    user_choice = input("Wpisz 1 lub 2 i naciśnij Enter: ")

    if user_choice == '1':
        num_simulations = 4
        for i in range(1, num_simulations):
            # Dynamiczna liczba uzytkownikow
            num_dynamic_users = 100
            dynamic_virtual_users = create_dynamic_virtual_users(num_dynamic_users, interests)
            assign_friends_to_users(dynamic_virtual_users)

            # Machine Learning
            X = df['text']
            y = df['label'].apply(lambda x: 1 if x == 'fake' else 0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            vectorizer = TfidfVectorizer(stop_words='english')
            X_train_vec = vectorizer.fit_transform(X_train)
            model = RandomForestClassifier()
            model.fit(X_train_vec, y_train)

            # Generowanie sieci z virtualnymi uzytkownikami
            network = create_social_network(dynamic_virtual_users)

            # Symulacja i eksport wyników
            simulation_results = simulate_and_export_results(dynamic_virtual_users, network, df, i)

            # Generowanie wykresu
            plot_simulation_results(simulation_results, df, i)  # Wywołanie funkcji z nowym df
            plot_article_frequency_with_interests(simulation_results, df, i)
            plot_infection_spread(simulation_results, i)
            plot_infection_spread_and_percentage(simulation_results, num_dynamic_users, i)
    elif user_choice == '2':
        # Dynamiczna liczba uzytkownikow
        num_dynamic_users = 100
        dynamic_virtual_users = create_dynamic_virtual_users(num_dynamic_users, interests)
        assign_friends_to_users(dynamic_virtual_users)

        # Machine Learning
        X = df['text']
        y = df['label'].apply(lambda x: 1 if x == 'fake' else 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        vectorizer = TfidfVectorizer(stop_words='english')
        X_train_vec = vectorizer.fit_transform(X_train)
        model = RandomForestClassifier()
        model.fit(X_train_vec, y_train)

        # Generowanie sieci z virtualnymi uzytkownikami
        network = create_social_network(dynamic_virtual_users)

        share_probabilities = [0.5, 0.6, 0.7, 0.8]
        for share_probability in share_probabilities:
            simulate_and_export_results(
                dynamic_virtual_users, network, df, 1, share_probability, True
            )
    else:
        print("Nieprawidłowy wybór. Uruchom program ponownie.")


if __name__ == '__main__':
    main()
