import pandas as pd
import zipfile
import json
import os

class DataLoaderFactory:
    @staticmethod
    def get_loader(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            return CSVLoader()
        elif ext == '.json':
            return JSONLoader()
        elif ext == '.zip':
            return ZIPLoader()
        elif ext == '.xlsx':
            return ExcelLoader()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

class CSVLoader:
    def load(self, file_path):
        return pd.read_csv(file_path)

class JSONLoader:
    def load(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)

class ZIPLoader:
    def load(self, file_path):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            for name in zip_ref.namelist():
                if name.endswith('.csv'):
                    with zip_ref.open(name) as f:
                        return pd.read_csv(f)
                elif name.endswith('.xlsx'):
                    with zip_ref.open(name) as f:
                        return pd.read_excel(f)
        raise ValueError("No supported file found in ZIP archive.")

class ExcelLoader:
    def load(self, file_path):
        return pd.read_excel(file_path)

def load_data(file_path):
    loader = DataLoaderFactory.get_loader(file_path)
    return loader.load(file_path)

# Example usage:
# df = load_data('your_file.xlsx')

if __name__ == "__main__":
    # Change the file name as needed
    df = load_data('card_transdata.csv')
    print(df.head())