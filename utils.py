import pandas as pd

file_path = 'news_articles.csv'
df = pd.read_csv(file_path)

print("Column Names:", df.columns.tolist())

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# Print a separator line
print("\n" + "-"*50 + "\n")

# Show the first 5 rows of the DataFrame
print(df.head())

