import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(input_file, output_train, output_val, output_test):
    df = pd.read_excel(input_file)  
    
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.2, random_state=42)
    
    train.to_csv(output_train, index=False)
    val.to_csv(output_val, index=False)
    test.to_csv(output_test, index=False)

if __name__ == "__main__":
    input_file = r'C:\Users\CSE RGUKT\Documents\Chandu\MAML-FewShot-20240708T114734Z-001\MAML-FewShot\data\3merged_file.xlsx'
    prepare_data(input_file, 'data/MergedData_train.csv', 'data/MergedData_val.csv', 'data/MergedData_test.csv')
