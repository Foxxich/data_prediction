import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
matplotlib.use('Agg')


def plot_simulation_results(simulation_results):
    plt.figure(figsize=(10, 6))
    read_counts = defaultdict(int)
    share_counts = defaultdict(int)

    # Przetwarzanie wyników symulacji
    for user_id, interactions in simulation_results.items():
        for action, article_id, day, *origin in interactions:
            if action == 'read':
                read_counts[article_id] += 1
            elif action == 'received':
                share_counts[article_id] += 1

    # Sort articles by ID for consistent plotting
    articles = sorted(set(read_counts.keys()).union(share_counts.keys()))
    read_freq = [read_counts[art] for art in articles]
    share_freq = [share_counts[art] for art in articles]

    # Adjust 'share_freq' to never exceed 'read_freq'
    share_freq = [min(shares, reads) for shares, reads in zip(share_freq, read_freq)]

    # Rysowanie wykresu
    plt.bar(articles, read_freq, label='Read Count', color='yellow')
    plt.bar(articles, share_freq, label='Share Count', color='red', alpha=0.5)

    plt.xlabel('Article ID')
    plt.ylabel('Frequency')
    plt.title('New Article Read and Share Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('articles.jpg')


def plot_infection_spread(simulation_results):
    plt.figure(figsize=(10, 6))
    infection_timeline = defaultdict(int)

    # Przetwarzanie wyników symulacji do stworzenia linii czasu zakażeń
    for user_id, interactions in simulation_results.items():
        for action, article_id, day, *origin in interactions:
            if action in ('read', 'received'):
                infection_timeline[day] += 1

    # Sortowanie dni, aby zapewnić poprawną kolejność na wykresie
    days = sorted(infection_timeline.keys())
    infections = [infection_timeline[day] for day in days]

    # Tworzenie wykresu liniowego pokazującego wzrost liczby zakażeń
    plt.plot(days, infections, marker='o', linestyle='-', color='red')

    plt.xlabel('Day')
    plt.ylabel('Cumulative Infections')
    plt.title('Infection Spread Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('infections.jpg')
