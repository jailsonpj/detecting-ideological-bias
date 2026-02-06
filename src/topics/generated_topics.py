import pandas as pd
import numpy as np
import json
import spacy
from gensim import corpora
from gensim.models import LdaModel
import joblib 

class LDAFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        self.dictionary = None
        self.lda_model = None

    def preprocess(self, texts):
        processed = []
        for doc in self.nlp.pipe(texts, batch_size=50):
            tokens = [token.lemma_.lower() for token in doc 
                      if token.pos_ in ['NOUN', 'ADJ'] and not token.is_stop and len(token.text) > 3]
            processed.append(tokens)
        return processed

    def get_vec(self, bow):
        vec = np.zeros(self.config['topics']['input_dim'])
        for topic_id, prob in self.lda_model.get_document_topics(bow, minimum_probability=0):
            if topic_id < self.config['topics']['input_dim']:
                vec[topic_id] = prob
        return vec

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

    extractor = LDAFeatureExtractor(config)

    
    df_train = pd.read_csv(config['dataset']['documents_train'])
    df_val = pd.read_csv(config['dataset']['documents_val'])
    df_test = pd.read_csv(config['dataset']['documents_test'])

    print("Processando Treino...")
    train_tokens = extractor.preprocess(
        df_train[config['dataset']['text_column']].astype(str)
    )
    
    extractor.dictionary = corpora.Dictionary(train_tokens)
    extractor.dictionary.filter_extremes(
        no_below=config['topics']['no_below'], 
        no_above=config['topics']['no_above']
    )
    
    train_corpus = [extractor.dictionary.doc2bow(text) for text in train_tokens]
    
    print("Treinando LDA...")
    extractor.lda_model = LdaModel(
        corpus=train_corpus,
        id2word=extractor.dictionary,
        num_topics=config['topics']['input_dim'],
        passes=config['topics']['passes'],
        random_state=42
    )

    def process_and_save(df, name):
        print(f"Inferindo tópicos para {name}...")
        tokens = extractor.preprocess(df[config['text_column']].astype(str))
        corpus = [extractor.dictionary.doc2bow(t) for t in tokens]
        
        probs = np.array(
            [extractor.get_vec(c) for c in corpus]
        ).astype(np.float32)
        
        topic_cols = [f'T_{i}' for i in range(config['topics']['input_dim'])]
        df_probs = pd.DataFrame(probs, columns=topic_cols)
        
        final_df = pd.concat(
            [df.reset_index(drop=True), df_probs], 
            axis=1
        )
        final_df.to_csv(
            f"{config['topics']['output_prefix']}_{name}.csv", 
            index=False
        )
        np.save(
            f"{config['topics']['output_prefix']}_{name}_features.npy", 
            probs
        )
        return probs

    X_train = process_and_save(df_train, "train")
    X_val = process_and_save(df_val, "val")
    X_test = process_and_save(df_test, "test")

    joblib.dump(extractor.lda_model, f"lda_model_{config['num_topics']}.pkl")
    extractor.dictionary.save(f"lda_dict_{config['num_topics']}.dict")

    print(f"Shape do X_train: {X_train.shape}") 
if __name__ == "__main__":
    main()