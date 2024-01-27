import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
from collections import defaultdict

matplotlib.use('Agg')


def plot_article_frequency_with_interests(simulation_results, df, iteration):
    # Przetwarzanie wyników symulacji
    read_counts = defaultdict(int)
    share_counts = defaultdict(int)

    # Zliczanie akcji 'read' i 'received' zgodnie z logiką podaną przez użytkownika
    for user_id, interactions in simulation_results.items():
        for action, article_id, day, *origin in interactions:
            if action == 'read':
                read_counts[article_id] += 1
            elif action == 'received':
                share_counts[article_id] += 1

    # Dodanie wydruków do weryfikacji poprawności zliczania
    print(f"Corrected Total 'read' actions: {sum(read_counts.values())}")
    print(f"Corrected Total 'received' actions: {sum(share_counts.values())}")

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

    # Zmiana barmode na 'group' zamiast 'stack'
    fig.update_layout(
        barmode='group',  # Zmienione z 'stack' na 'group'
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


def plot_infection_spread_and_percentage(simulation_results, num_users, iteration):
    infection_timeline = defaultdict(int)
    infected_users = set()
    percentage_infected = []

    max_day = max(max(day for _, _, day, *_ in interactions) for interactions in simulation_results.values())
    for day in range(1, max_day + 1):
        daily_infected_count = 0
        for user_id, interactions in simulation_results.items():
            if user_id not in infected_users:
                interactions_today = [interaction for interaction in interactions if interaction[2] == day]
                if any(action in ('read', 'received') for action, _, _, *_ in interactions_today):
                    infected_users.add(user_id)
                    daily_infected_count += 1
        infection_timeline[day] += daily_infected_count
        percentage_infected.append(len(infected_users) / num_users * 100)

    days = sorted(infection_timeline.keys())
    infections = [infection_timeline[day] for day in days]

    mu, sigma = np.mean(infections), np.std(infections)
    if sigma == 0:
        print("Odchylenie standardowe wynosi zero, nie można wygenerować dystrybuanty i gęstości rozkładu normalnego.")
        return

    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    p = norm.cdf(x, mu, sigma)
    density = norm.pdf(x, mu, sigma)

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x, p, 'k-', lw=2)
    plt.title('Dystrybuanta rozkładu normalnego')
    plt.xlabel('Możliwe wyniki zmiennej (cechy)')
    plt.ylabel('Wartość dystrybuanty')
    plt.subplot(1, 2, 2)
    plt.plot(x, density, 'k-', lw=2)
    plt.fill_between(x, density, 0, alpha=0.1)
    plt.title('Gęstość rozkładu normalnego')
    plt.xlabel('Możliwe wyniki zmiennej (cechy)')
    plt.ylabel('Gęstość')

    plt.tight_layout()
    plt.savefig(f"{iteration}_infection_spread_and_percentage.png")
    plt.close()


def plot_infections(simulation_results, num_users, iteration):
    plt.figure(figsize=(12, 6))

    # Przetwarzanie wyników symulacji
    new_infections_daily = defaultdict(int)
    infected_users = set()

    # Ustalanie zakresu dni
    max_day = max(max(day for _, _, day, *_ in interactions) for interactions in simulation_results.values())
    for day in range(1, max_day + 1):
        daily_infected_count = 0
        for user_id, interactions in simulation_results.items():
            if user_id not in infected_users:
                interactions_today = [interaction for interaction in interactions if interaction[2] == day]
                if any(action in ('read', 'received') for action, _, _, *_ in interactions_today):
                    infected_users.add(user_id)
                    daily_infected_count += 1
        new_infections_daily[day] = daily_infected_count

    days = sorted(new_infections_daily.keys())
    new_infections = [new_infections_daily[day] for day in days]

    # Wykres liczby nowych zakażeń każdego dnia
    plt.subplot(1, 2, 1)
    plt.bar(days, new_infections, color='red')
    plt.xlabel('Day')
    plt.ylabel('Number of New Infections')
    plt.title('Daily New Infections')
    plt.grid(True)
    plt.xticks(days)  # Ustawienie etykiet osi X jako liczby całkowite

    # Wykres kumulatywnego procentu populacji, która zachorowała
    cumulative_infections = np.cumsum(new_infections)
    percentage_infected = [count / num_users * 100 for count in cumulative_infections]

    plt.subplot(1, 2, 2)
    plt.plot(days, percentage_infected, marker='o', linestyle='-', color='blue')
    plt.xlabel('Day')
    plt.ylabel('Percentage of Population Infected')
    plt.title('Cumulative Percentage of Population Infected')
    plt.grid(True)
    plt.xticks(days)  # Ustawienie etykiet osi X jako liczby całkowite

    plt.tight_layout()
    plt.savefig(f"{iteration}_plot_infections.jpg")
