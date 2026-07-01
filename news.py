#########################################################################################
#          THIS IS A RECOMMENDATION SYSTEM BASED ON THE USER's COVERED TOPICS           #
#########################################################################################


from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.feature_extraction.text as text
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('wordnet')
from ddgs import DDGS
import pandas as pd
from umap import UMAP


def find_topics(user_questions):
    custom_stopwords = list(text.ENGLISH_STOP_WORDS)
    custom_stopwords.extend(["why", "are", "can", "be", "benefits", "what", "how", "is", "important"])

    vectorizer_model = CountVectorizer(stop_words=custom_stopwords)

    documents = user_questions
    num_docs = len(documents)

    # Initialize and train BERTopic using a lightweight embedding model
    #topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2")
    #topic_model = BERTopic(
    #    vectorizer_model=vectorizer_model,
    #    min_topic_size=2,                # Lower this so small clusters can become real topics
    #    calculate_probabilities=True
    #)
    if num_docs < 3:
        print(f"Too few documents ({num_docs}) for UMAP. Using keyword fallback.")
        # Create custom stopwords and vectorizer locally to get representations
        
        # Safe fallback DataFrame matching your downstream needs
        res = pd.DataFrame({
            "Topic": [0] * num_docs,
            "Count": [1] * num_docs,
            "Name": ["0_fallback_query"] * num_docs,
            "Representation": [documents] * num_docs,
            "Representative_Docs": [documents] * num_docs
        })
        return res
    # Keeps init='random' for small sets, but scales n_neighbors safely.
    chosen_init = 'spectral' if num_docs >= 20 else 'random'

    # 🌟 THE CRITICAL FIX: Ensure n_neighbors can NEVER drop below 2!
    chosen_neighbors = max(2, min(15, num_docs - 1))

    custom_umap = UMAP(
        n_neighbors=chosen_neighbors,
        n_components=2,
        init=chosen_init,
        random_state=42
    )

    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        umap_model=custom_umap,          # 👈 Pass your robust UMAP here
        min_topic_size=2,                
        calculate_probabilities=True
    )


    topics, probs = topic_model.fit_transform(documents)

    if set(topics) == {-1}:
        print("Only outliers detected.")
        return pd.DataFrame({
            "Representation": [documents]
        })
    elif -1 in topics:
        print("Outliers detected. Running outlier reduction strategy...")
        new_topics = topic_model.reduce_outliers(documents, topics, strategy="c-tf-idf")
        topic_model.update_topics(documents, topics=new_topics, vectorizer_model=vectorizer_model)
    else:
        print("No outliers detected! All documents successfully assigned to topics.")
        new_topics = topics

    res = topic_model.get_topic_info()
    # View details about the extracted topics
    #print(res)
    return res


#def best_topics(user_questions):
#  res = find_topics(user_questions)
#  topics = res["Representation"].to_list()
#  t_best = topics[0] + topics[1]
#  return t_best


def search_query(user_questions):
    topics = find_topics(user_questions)["Representation"].to_list()
    #topics = res["Representation"].to_list()
    if len(topics) == 1:
        ts = topics[0]
    else:
        ts = topics[0] + topics[1]
    t_best = [tpc for tpc in ts if tpc != ''] 
    #removing the words with the same roots; keeping only one
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = set(lemmatizer.lemmatize(w) for w in t_best)
    print(lemmatized_words)

    search_phrase = " ".join(list(lemmatized_words)[:5]) + " news article"
    return search_phrase


def fetch_live_news_recommendations(query: str, max_results: int = 3) -> list[dict]:
    recommendations = []
    
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results, timelimit='w')
            
            if results:
                for idx, result in enumerate(results):
                    recommendations.append({
                        "id": idx + 1,
                        "title": result.get("title"),
                        "url": result.get("href"),
                        #"source": result.get("source", "Unknown"),
                        "snippet": result.get("body")
                    })
    except Exception as e:
        # Catches 'No results found', timeouts, or network drops safely
        print(f"Search skipped for query '{query}': {e}")
        return []
        
    return recommendations

def chat_history_to_user_questions(chat_history):
    user_questions = []
    i = 0
    for cont in chat_history[::-1]:
        if i == 20:
            return user_questions
        else:
            if cont["role"] == "user":
                user_questions.append(cont["content"])
                i += 1  
    return user_questions

def suggest_news_article(user_questions):
    current_search_query = search_query(user_questions)
    print(current_search_query)

    # Execute the live crawl
    suggested_articles = fetch_live_news_recommendations(current_search_query)

    # Display recommendations to user
    print("\n--- Live Internet Recommendations ---")
    for article in suggested_articles:
        print(f"\n[{article['id']}] {article['title']}")
        print(f"URL: {article['url']}")
        print(f"Snippet: {article['snippet'][:120]}...")
    return suggested_articles