import streamlit as st
import gensim
import nltk
import numpy as np
import pandas as pd
import pyLDAvis.gensim_models
from gensim import corpora
import streamlit.components.v1 as components
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from wordcloud import WordCloud
import matplotlib.colors as mcolors
from utils.config import *
from utils.preprocess import *

DEFAULT_NUM_TOPICS = 4

DATASETS = {
    'Comcast Reviews': {
        'path': './data/comcast_consumeraffairs_complaints.csv',
        'column': 'text',
        'url': 'https://www.kaggle.com/datasets/archaeocharlie/comcastcomplaints',
        'description': (
            'Data on reviews posted online about Comcast services'
        )
    },
    'Comcast Consumer Complaints': {
        'path': './data/comcast_fcc_complaints_2015.csv',
        'column': 'Description',
        'url': 'https://www.kaggle.com/datasets/yasserh/comcast-telecom-complaints?select=Comcast.csv',
        'description': (
            'Data on complaints raise by the customers of Comcast'
        )
    }
}

MODELS = {
    'Latent Dirichlet Allocation': {
        'class': gensim.models.LdaMulticore,
        'help': 'https://radimrehurek.com/gensim/models/ldamodel.html'
    },
    'Bertopic': {
        'class': BERTopic,
        'help': 'https://maartengr.github.io/BERTopic/index.html'
    }
}

COLORS = [color for color in mcolors.XKCD_COLORS.values()]

WORDCLOUD_FONT_PATH = r'./data/Inkfree.ttf'

@st.cache_data()
def generate_texts_df(selected_dataset: str):
    dataset = DATASETS[selected_dataset]
    return pd.read_csv(f'{dataset["path"]}')

@st.cache_data()
def denoise_docs(texts_df: pd.DataFrame, text_column: str, model_key):
    #texts = texts_df[text_column].values.tolist()
    docs = texts_df[text_column].apply(lambda x: abbrev_conversion(x,lookup_dict))
    #docs = abbrev_conversion()
    if model_key == 'Latent Dirichlet Allocation':
        docs = docs.apply(cleanData_for_gensim)
    else:
        docs = docs.apply(cleanData_for_bert)
    return docs

@st.cache_data()
def generate_docs(texts_df: pd.DataFrame, text_column: str):
    docs = denoise_docs(texts_df, text_column)
    return docs


@st.cache_data()
def generate_wordcloud(docs, collocations: bool = False):
    wordcloud_text = (' '.join(' '.join(doc) for doc in docs))
    wordcloud = WordCloud(font_path=WORDCLOUD_FONT_PATH, width=700, height=600,
                          background_color='white', collocations=collocations).generate(wordcloud_text)
    return wordcloud


@st.cache_data()
def prepare_training_data(docs):
    id2word = corpora.Dictionary(docs)
    corpus = [id2word.doc2bow(doc) for doc in docs]
    return id2word, corpus

@st.cache_data()
def train_model_lda(modeldocs, base_model):

    id2word, corpus = prepare_training_data(docs)
    model = base_model(corpus=corpus, id2word=id2word, num_topics=DEFAULT_NUM_TOPICS)
    
    return id2word, corpus, model

@st.cache_data()
def train_model_bert(modeldocs, base_model):
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    # Create your representation model
    representation_model = KeyBERTInspired()
    # Use the representation model in BERTopic on top of the default pipeline
    model = base_model(representation_model=representation_model,ctfidf_model=ctfidf_model)
    topics, probabilities = model.fit_transform(df)

    return model,topics,probabilities



def clear_session_state():
    for key in ('model_kwargs', 'id2word', 'corpus', 'model'):
        if key in st.session_state:
            del st.session_state[key]


@st.cache_data()
def white_or_black_text(background_color):
    # https://stackoverflow.com/questions/3942878/how-to-decide-font-color-in-white-or-black-depending-on-background-color
    red = int(background_color[1:3], 16)
    green = int(background_color[3:5], 16)
    blue = int(background_color[5:], 16)
    return 'black' if (red * 0.299 + green * 0.587 + blue * 0.114) > 186 else 'white'


if __name__ == '__main__':
    st.set_page_config(page_title='Topic Modeling')
    st.title('Topic Modeling')
    st.header('What is topic modeling?')
    with st.expander('Hero Image'):
        st.image('./data/topic.png', use_column_width=True)
    st.markdown(
        'Topic modeling is a type of statistical modeling that uses unsupervised Machine Learning to identify clusters or groups of similar words within a body of text.'
        'This text mining method uses semantic structures in text to understand unstructured data without predefined tags or training data.'
        'Topic modeling analyzes documents to identify common themes and provide an adequate cluster.' 
        'For example, a topic modeling algorithm could identify whether incoming documents are contracts, invoices, complaints, or more based on their contents.'
    )

    st.header('Datasets')
    st.markdown('Preloaded a couple of small example datasets to illustrate.')
    selected_dataset = st.selectbox('Dataset', [None, *sorted(list(DATASETS.keys()))], on_change=clear_session_state)
    if not selected_dataset:
        st.write('Choose a Dataset to Continue ...')
        st.stop()

    model_key = st.sidebar.selectbox('Model', [None, *list(MODELS.keys())], on_change=clear_session_state)
    if not model_key:
        with st.sidebar:
            st.write('Choose a Model to Continue ...')
            train_model_clicked = st.form_submit_button('Train Model')
        st.stop()
    
    with st.expander('Dataset Description'):
        st.markdown(DATASETS[selected_dataset]['description'])
        st.markdown(DATASETS[selected_dataset]['url'])

    text_column = DATASETS[selected_dataset]['column']
    texts_df = generate_texts_df(selected_dataset)
    docs = generate_docs(texts_df, text_column)

    with st.expander('Sample Documents'):
        sample_texts = texts_df[text_column].sample(5).values.tolist()
        for index, text in enumerate(sample_texts):
            st.markdown(f'**{index + 1}**: _{text}_')

    with st.expander('Frequency Sized Corpus Wordcloud'):
        wc = generate_wordcloud(docs)
        st.image(wc.to_image(), caption='Dataset Wordcloud (Not A Topic Model)', use_column_width=True)
        st.markdown('These are the remaining words after document preprocessing.')

    with st.expander('Document Word Count Distribution'):
        len_docs = [len(doc) for doc in docs]
        fig, ax = plt.subplots()
        sns.histplot(data=pd.DataFrame(len_docs, columns=['Words In Document']), discrete=True, ax=ax)
        st.pyplot(fig)

    
    
    if train_model_clicked:
        if model_key == 'Latent Dirichlet Allocation':

            with st.spinner('Training Model ...'):
                id2word, corpus, model = train_model_lda(docs, MODELS[model_key]['class'])
            st.session_state.id2word = id2word
            st.session_state.corpus = corpus
            st.session_state.model = model

        else:
            with st.spinner('Training Model ...'):
                model,topics,probabilities = train_model_bert(docs, MODELS[model_key]['class'])
            st.session_state.model = model
            st.session_state.topics = topics
            st.session_state.probabilities = probabilities



    if 'model' not in st.session_state:
        st.stop()

    st.header('Model')
    st.write(type(st.session_state.model).__name__)
    #st.write(st.session_state.model_kwargs)

    st.header('Model Results')
    """
    topics = st.session_state.model.show_topics(formatted=False, num_words=50,
                                                num_topics=DEFAULT_NUM_TOPICS, log=False)
    """ 
    """                                        
    with st.expander('Topic Word-Weighted Summaries'):
        topic_summaries = {}
        for topic in topics:
            topic_index = topic[0]
            topic_word_weights = topic[1]
            topic_summaries[topic_index] = ' + '.join(
                f'{weight:.3f} * {word}' for word, weight in topic_word_weights[:10])
        for topic_index, topic_summary in topic_summaries.items():
            st.markdown(f'**Topic {topic_index}**: _{topic_summary}_')
    """
    """
    with st.expander('Visualize the topics'):
        if model_key == 'Latent Dirichlet Allocation':

            wc = pyLDAvis.gensim.prepare(model, corpus, id2word)
            st.image(wc.to_image(), caption='Dominant Topics for LDA model', use_column_width=True)
            st.markdown('These are the dominant words in wach topic generated by LDA')

        else:
            wc= model.visualize_barchart(top_n_topics=9, height=700)
            st.image(wc.to_image(), caption='Dominant Topics for Bertopic model', use_column_width=True)
            st.markdown('These are the dominant words in wach topic generated by Bertopic')
    """

    st.header('Model Results')
    if model_key == 'Latent Dirichlet Allocation':
        topics_lda = st.session_state.model.show_topics(formatted=False, num_words=50,
                                               num_topics=DEFAULT_NUM_TOPICS, log=False)

        colors = random.sample(COLORS, k=DEFAULT_NUM_TOPICS)
        with st.expander('Top N Topic Keywords Wordclouds'):
            cols = st.columns(3)
            for index, topic in enumerate(topics_lda):
                wc = WordCloud(font_path=WORDCLOUD_FONT_PATH, width=700, height=600,
                            background_color='white', collocations=collocations, prefer_horizontal=1.0,
                            color_func=lambda *args, **kwargs: colors[index])
                with cols[index % 3]:
                    wc.generate_from_frequencies(dict(topic[1]))
                    st.image(wc.to_image(), caption=f'Topic #{index}', use_column_width=True)

        if hasattr(st.session_state.model, 'inference'):  # gensim Nmf has no 'inference' attribute so pyLDAvis fails
            if st.button('Generate visualization'):
                with st.spinner('Creating pyLDAvis Visualization ...'):
                    py_lda_vis_data = pyLDAvis.gensim_models.prepare(st.session_state.model, st.session_state.corpus,
                                                                    st.session_state.id2word)
                    py_lda_vis_html = pyLDAvis.prepared_data_to_html(py_lda_vis_data)
                with st.expander('pyLDAvis', expanded=True):
                    components.html(py_lda_vis_html, width=1300, height=800)

    else:

        colors = random.sample(COLORS, k=DEFAULT_NUM_TOPICS)
        with st.expander('Top N Topic Keywords Wordclouds'):
            cols = st.columns(3)
            for index, topic in enumerate(st.session_state.topics):
                wc = WordCloud(font_path=WORDCLOUD_FONT_PATH, width=700, height=600,
                            background_color='white', collocations=collocations, prefer_horizontal=1.0,
                            color_func=lambda *args, **kwargs: colors[index])
                with cols[index % 3]:
                    wc.generate_from_frequencies(dict(topic[1]))
                    st.image(wc.to_image(), caption=f'Topic #{index}', use_column_width=True)
        
        if st.button('Generate visualization'):
                with st.spinner('Creating bertopic barchart ...'):
                    bertopic_vis = st.session_state.model.visualize_barchart(top_n_topics=10, height=700)
                with st.expander('visualize_barchart', expanded=True):
                    components.html(bertopic_vis, width=1300, height=800)


















