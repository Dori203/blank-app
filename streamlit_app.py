import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

class PromptSearchCluster:
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir
        self.csv_files = [f'split_{i:03d}.csv' for i in range(1, 11)]  # Files from split_001.csv to split_010.csv
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.fit_vectorizer()
    
    def fit_vectorizer(self):
        sample_size = 100000  # Adjust based on your needs and machine capabilities
        samples = []
        for file in self.csv_files:
            file_path = os.path.join(self.csv_dir, file)
            if os.path.exists(file_path):
                df_sample = pd.read_csv(file_path, nrows=sample_size // len(self.csv_files))
                samples.append(df_sample['prompt'].astype(str))
        self.vectorizer.fit(pd.concat(samples))
    
    def search(self, query, top_n=1000):
        query_vec = self.vectorizer.transform([query])
        results = []
        for file in self.csv_files:
            file_path = os.path.join(self.csv_dir, file)
            if os.path.exists(file_path):
                for chunk in pd.read_csv(file_path, chunksize=10000):
                    chunk = chunk.dropna(subset=['prompt', 'url'])
                    chunk_tfidf = self.vectorizer.transform(chunk['prompt'].astype(str))
                    similarities = cosine_similarity(query_vec, chunk_tfidf).flatten()
                    top_indices = similarities.argsort()[:-top_n-1:-1]
                    chunk_results = chunk.iloc[top_indices]
                    chunk_results['similarity'] = similarities[top_indices]
                    results.append(chunk_results)
        
        all_results = pd.concat(results)
        return all_results.sort_values('similarity', ascending=False).head(top_n)
    
    def cluster_results(self, results, n_clusters=10):
        result_tfidf = self.vectorizer.transform(results['prompt'].astype(str))
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
def load_data(csv_dir):
    return PromptSearchCluster(csv_dir)

def main():
    st.title("Prompt Search and Cluster")

    csv_dir = "split_csv_files"  # Directory containing split CSV files
    
    with st.spinner("Loading and processing the CSV files. This may take a while..."):
        service = load_data(csv_dir)

    st.success("Data loaded successfully!")

    query = st.text_input("Enter your search query:")
    
    if st.button("Search"):
        if query:
            with st.spinner("Searching and clustering results..."):
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
