import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import random

class PromptSearchCluster:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=['prompt', 'url'])
        self.df['prompt'] = self.df['prompt'].astype(str)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['prompt'])
    
    def search(self, query, top_n=1000):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[:-top_n-1:-1]
        return self.df.iloc[top_indices]
    
    def cluster_results(self, results, n_clusters=10):
        result_tfidf = self.vectorizer.transform(results['prompt'])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        results['cluster'] = kmeans.fit_predict(result_tfidf)
        return results
    
    def get_cluster_keywords(self, clustered_results, top_n=5):
        keywords = {}
        for cluster in clustered_results['cluster'].unique():
            cluster_docs = clustered_results[clustered_results['cluster'] == cluster]['prompt']
            tfidf_matrix = self.vectorizer.transform(cluster_docs)
            feature_array = np.array(self.vectorizer.get_feature_names_out())
            tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
            top_keywords = feature_array[tfidf_sorting][:top_n]
            keywords[cluster] = top_keywords.tolist()
        return keywords

    def get_cluster_image(self, cluster_results, cluster):
        cluster_images = cluster_results[cluster_results['cluster'] == cluster]['url']
        return random.choice(cluster_images.tolist())

@st.cache_resource
def load_data(csv_path):
    return PromptSearchCluster(csv_path)

def main():
    st.title("Prompt Search and Cluster")

    csv_path = "diffusion_prompts.csv"  # Replace with your CSV file path
    service = load_data(csv_path)

    query = st.text_input("Enter your search query:")
    
    if st.button("Search"):
        if query:
            search_results = service.search(query)
            clustered_results = service.cluster_results(search_results)
            cluster_keywords = service.get_cluster_keywords(clustered_results)

            st.write(f"Found {len(search_results)} results, clustered into 10 groups:")

            for cluster in range(10):
                st.subheader(f"Cluster {cluster}")
                st.write(f"Keywords: {', '.join(cluster_keywords[cluster])}")
                image_url = service.get_cluster_image(clustered_results, cluster)
                st.image(image_url, caption=f"Example image from cluster {cluster}", use_column_width=True)

if __name__ == "__main__":
    main()
