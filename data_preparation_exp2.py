import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import wordnet
import random

def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if len(synonyms) >= 1:
            synonym = synonyms[0].lemmas()[0].name()
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_words)

def augment_data(df, n=1):
    df['Tweet'] = df['Tweet'].apply(lambda x: synonym_replacement(x, n))
    return df

def prepare_data(input_file, output_train, output_val, output_test):
    df = pd.read_excel(input_file)

    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.2, random_state=42)
    
    train = augment_data(train, n=1)  

    train.to_csv(output_train, index=False)
    val.to_csv(output_val, index=False)
    test.to_csv(output_test, index=False)

if __name__ == "__main__":
    input_file = r'C:\Users\CSE RGUKT\Documents\Chandu\MAML-FewShot-20240708T114734Z-001\MAML-FewShot\data\3merged_file.xlsx'
    prepare_data(input_file, 'data/MergedDataExp2_train.csv', 'data/MergedDataExp2_val.csv', 'data/MergedDataExp2_test.csv')
