import matplotlib
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
matplotlib.use('Agg')


def plot_article_frequency_with_interests(simulation_results, df, iteration):
    # Przetwarzanie wyników symulacji
    read_counts = defaultdict(int)
    share_counts = defaultdict(int)
    for user_id, interactions in simulation_results.items():
        for action, article_id, day, *origin in interactions:
            if action == 'read':
                read_counts[article_id] += 1
            elif action == 'received':
                share_counts[article_id] += 1

    # Tworzenie DataFrame
    articles_data = []
    for article_id in set(read_counts.keys()).union(share_counts.keys()):
        articles_data.append({
            'Article ID': article_id,
            'Read Count': read_counts[article_id],
            'Share Count': share_counts[article_id],
            'Total Interactions': read_counts[article_id] + share_counts[article_id]
        })
    articles_df = pd.DataFrame(articles_data)

    # Dołączanie informacji o interesach
    articles_df = articles_df.merge(df[['id', 'interests']], left_on='Article ID', right_on='id', how='left')

    # Sortowanie według łącznej liczby interakcji
    articles_df = articles_df.sort_values(by='Total Interactions', ascending=False)

    # Tworzenie wykresu
    fig = go.Figure(data=[
        go.Bar(
            name='Read Count',
            x=articles_df['Article ID'].astype(str),
            y=articles_df['Read Count'],
            marker_color='yellow',
            hoverinfo='text',
            hovertext=articles_df['interests']
        ),
        go.Bar(
            name='Share Count',
            x=articles_df['Article ID'].astype(str),
            y=articles_df['Share Count'],
            marker_color='red',
            hoverinfo='text',
            hovertext=articles_df['interests']
        )
    ])

    fig.update_layout(
        barmode='stack',
        title='Article Read and Share Frequency with Interests',
        xaxis_title='Article ID',
        yaxis_title='Frequency',
        legend_title='Actions',
        hovermode='closest'
    )

    # Zapis do pliku HTML
    html_file = str(iteration) + '_articles_with_interests_interactive.html'
    fig.write_html(html_file)

    return html_file


def plot_simulation_results(simulation_results, df, iteration):
    # Przetwarzanie wyników symulacji
    read_counts = defaultdict(int)
    share_counts = defaultdict(int)
    for user_id, interactions in simulation_results.items():
        for action, article_id, day, *origin in interactions:
            if action == 'read':
                read_counts[article_id] += 1
            elif action == 'received':
                share_counts[article_id] += 1

    # Tworzenie DataFrame
    data = {'Article ID': [], 'Read Count': [], 'Share Count': []}
    for article_id in set(read_counts.keys()).union(share_counts.keys()):
        data['Article ID'].append(article_id)
        data['Read Count'].append(read_counts[article_id])
        data['Share Count'].append(share_counts[article_id])

    df_stats = pd.DataFrame(data)
    df_stats = df_stats.merge(df[['id', 'interests']], left_on='Article ID', right_on='id', how='left')

    # Tworzenie wykresu skrzypcowego z plotly
    fig = px.violin(df_stats, y=['Read Count', 'Share Count'], box=True, points="all",
                    labels={'value': 'Count', 'variable': 'Action'},
                    title='Read and Share Counts of Articles')

    # Zapis do pliku HTML
    html_file = str(iteration) + '_plot.html'
    fig.write_html(html_file)

    return html_file


def plot_infection_spread(simulation_results, iteration):
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
    plt.savefig(str(iteration) + '_infections.jpg')


import matplotlib.pyplot as plt
from collections import defaultdict


def plot_infection_spread_and_percentage(simulation_results, num_users, iteration):
    plt.figure(figsize=(12, 6))

    # Przetwarzanie wyników symulacji do stworzenia linii czasu zakażeń
    infection_timeline = defaultdict(int)
    infected_users = set()
    percentage_infected = []

    for user_id, interactions in simulation_results.items():
        for action, article_id, day, *origin in interactions:
            if action in ('read', 'received'):
                infected_users.add(user_id)
                infection_timeline[day] += 1
        percentage_infected.append(len(infected_users) / num_users * 100)

    # Sortowanie dni, aby zapewnić poprawną kolejność na wykresie
    days = sorted(infection_timeline.keys())
    infections = [infection_timeline[day] for day in days]

    # Wykres liczby zakażeń
    plt.subplot(1, 2, 1)
    plt.plot(days, infections, marker='o', linestyle='-', color='red')
    plt.xlabel('Day')
    plt.ylabel('Number of Infections')
    plt.title('Daily Infections')
    plt.grid(True)

    # Wykres procentu populacji, która zachorowała
    plt.subplot(1, 2, 2)
    plt.plot(days, percentage_infected[:len(days)], marker='o', linestyle='-', color='blue')
    plt.xlabel('Day')
    plt.ylabel('Percentage of Population Infected')
    plt.title('Cumulative Percentage of Population Infected')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(str(iteration) + '_infections_and_percentage.jpg')
