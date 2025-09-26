import pandas as pd

# Глобальные переменные для названий колонок
time_spend = '8. What is the average time you spend on social media every day?'
sleep_problem = '20. On a scale of 1 to 5, how often do you face issues regarding sleep?'

def convert_social_time(df, column):
    time_order = {
        'Less than an Hour': 1,
        'Between 1 and 2 hours': 2,
        'Between 2 and 3 hours': 3,
        'Between 3 and 4 hours': 4,
        'Between 4 and 5 hours': 5,
        'More than 5 hours': 6
    }
    df['social_time_num'] = df[column].map(time_order)
    return df

def check_missing(df, columns):
    missing = df[columns].isnull().sum()
    print("Пропущенные значения по колонкам:")
    print(missing)
    return missing

def fill_missing(df, column, method='mean'):
    if method == 'mean':
        df[column] = df[column].fillna(df[column].mean())
    elif method == 'median':
        df[column] = df[column].fillna(df[column].median())
    return df