import matplotlib.pyplot as plt
import seaborn as sns

def add_histogram(df, column, bins=10):
    plt.hist(df[column].dropna(), bins=bins, color='skyblue', edgecolor='black')
    plt.title(f'Гистограмма для {column}')
    plt.xlabel(column)
    plt.ylabel('Частота')
    plt.show()

def add_scatter(df, x_column, y_column):
    plt.scatter(df[x_column], df[y_column], alpha=0.7)
    plt.title(f'Диаграмма рассеяния: {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()
    
def add_boxplot(df, x_column, y_column):
    plt.figure(figsize=(8,6))
    sns.boxplot(x=x_column, y=y_column, data=df)
    plt.title(f'Boxplot: {y_column} по {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()
