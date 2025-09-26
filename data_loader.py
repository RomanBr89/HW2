import pandas as pd

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Файл успешно загружен")
        return df
    except Exception as e:
        print(f"Ошибка при загрузке CSV: {e}")
        return None