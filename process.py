from config import *
import pandas as pd


def format_sample(file_paths, output_path):
    text = bio = pola = ''
    items = []

    for file_path in file_paths:
        with open(file_path, encoding='utf-8') as f:
            for line in f.readlines():
                if line == '\n':
                    items.append({'text': text.strip(), 'bio': bio.strip(), 'pola': pola.strip()})
                    text = bio = pola = ''
                    continue

                t, b, p = line.split(' ')
                text += t + ' '
                bio += b + ' '

                p = str(1) if p.strip() == str(2) else p.strip()
                pola += p + ' '

    df = pd.DataFrame(items)
    df.to_csv(output_path, index=None)


def check_label():
    df = pd.read_csv(TRAIN_FILE_PATH)
    dct = {}
    for index, row in df.iterrows():
        for b, p in zip(row['bio'].split(), row['pola'].split()):
            if b == 'B-ASP' and p == '-1':
                print(index, row)
                df.drop(index=index, inplace=True)
            cnt = dct.get((b, p), 0)
            dct[(b, p)] = cnt + 1
    print(dct)

    df.to_csv(TRAIN_FILE_PATH, index=None)


def split_sample():
    file_name = './output/process/fine_grained_sentiment_analysis.sample.all.csv'
    df = pd.read_csv(file_name)
    df = df.sample(frac=1)
    df.reset_index(inplace=True, drop=True)
    n = len(df)
    df.loc[:int(n * 0.8), :].to_csv('./output/process/fine_grained_sentiment_analysis.sample.train.csv', index=None)
    df.loc[int(n * 0.8):, :].to_csv('./output/process/fine_grained_sentiment_analysis.sample.test.csv', index=None)


def check_label_sample():
    df = pd.read_csv('./output/process/fine_grained_sentiment_analysis.sample.all.csv')
    dct = {}

    for index, row in df.iterrows():
        for b, p in zip(row['bio'].split(), row['pola'].split()):
            if b == 'B-ASP' and p == '-1':
                print(index, row)
                df.drop(index=index, inplace=True)
            cnt = dct.get((b, p), 0)
            dct[(b, p)] = cnt + 1

    print(dct)
    df.to_csv('./output/process/fine_grained_sentiment_analysis.sample.all.csv', index=None)


if __name__ == '__main__':
    format_sample([
         './input/raw_data/spotify/spotify.fine_grained_sentiment_analysis.train.dat',
         './input/raw_data/threads/threads.fine_grained_sentiment_analysis.train.dat',
         './input/raw_data/chatGPT/chatGPT.fine_grained_sentiment_analysis.train.dat',
    ], TRAIN_FILE_PATH)

    format_sample([
         './input/origin/spotify/spotify.fine_grained_sentiment_analysis.test.dat',
         './input/origin/threads/threads.fine_grained_sentiment_analysis.test.dat',
         './input/origin/chatGPT/chatGPT.fine_grained_sentiment_analysis.test.dat',
     ], TEST_FILE_PATH)

    check_label()

    format_sample([
         './input/raw_data/spotify/spotify.fine_grained_sentiment_analysis.train.dat',
         './input/raw_data/threads/threads.fine_grained_sentiment_analysis.train.dat',
         './input/raw_data/chatGPT/chatGPT.fine_grained_sentiment_analysis.train.dat',
         './input/origin/spotify/spotify.fine_grained_sentiment_analysis.test.dat',
         './input/origin/threads/threads.fine_grained_sentiment_analysis.test.dat',
         './input/origin/chatGPT/chatGPT.fine_grained_sentiment_analysis.test.dat',
     ], './output/process/fine_grained_sentiment_analysis.sample.all.csv')

    check_label_sample()

    split_sample()