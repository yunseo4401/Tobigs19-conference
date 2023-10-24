import torch
import pandas as pd

# Custom Dataset
class AdArticleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        # item['labels'] = item['labels'].squeeze(0)
        # item['attention_mask'] = item['labels'].squeeze(0)
        # item['token_type_ids'] = item['token_type_ids'].squeeze(0)

        return item

    def __len__(self):
        return len(self.labels)
    


def load(tokenizer):
    df_label0 = pd.read_csv('일반기사데이터.csv', index_col=0)
    df_label1 = pd.read_csv('data_processed.csv', index_col=0)

    df_label0.drop(columns=['newsTitle'], inplace=True)
    df_label0.rename({'newsContent' : 'content'}, axis=1, inplace=True)
    df_label0['label'] = df_label0['label'].map({1 : 0})

    df_label1.drop(columns=['매체명', '광고주', '상품명', 'link1'], inplace=True)
    df_label1.rename({'text1' : 'content'}, axis=1, inplace=True)
    df_label1['label'] = 1


    # 간단한 전처리
    df_label0['content'] = df_label0['content'].apply(lambda x : x.replace(' \n', '. ').strip()).tolist()
    df_label1['content'] = df_label1['content'].apply(lambda x : x.replace('\n', ' ').strip()).tolist()

    # 토큰 길이 칼럼 'toklen' 추가
    toklen = [len(tokenizer.tokenize(doc)) for doc in df_label1['content'].tolist()]
    df_label1['toklen'] = toklen
    
    # drop 23 rows
    tmp_index = df_label1[df_label1['toklen']>4096].index
    df_label1.drop(index=tmp_index, inplace=True)
    df_label1.drop(columns=['toklen'], inplace=True)

    # combine data
    texts = df_label1['content'].tolist() + df_label0['content'].tolist()
    labels = df_label1['label'].tolist() + df_label0['label'].tolist()

    return texts, labels



def split_and_tokenize(texts, labels, tokenizer, maxlen=512, random_state=2023):
    from sklearn.model_selection import train_test_split

    # train : valid : test = 7: 1.5 : 1.5 split
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.3, shuffle=True, random_state=random_state, stratify=labels)
    val_texts, test_texts, val_labels, test_labels = train_test_split(val_texts, val_labels, test_size=0.5, shuffle=True, random_state=random_state, stratify=val_labels)

    # tokenize
    print('-------------------------tokenizing-------------------------------')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=maxlen)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=maxlen)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=maxlen)

    train_dataset = AdArticleDataset(train_encodings, train_labels)
    val_dataset = AdArticleDataset(val_encodings, val_labels)
    test_dataset = AdArticleDataset(test_encodings, test_labels)

    return train_dataset, val_dataset, test_dataset