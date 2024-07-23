import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import random
import pickle

# Constants
EMBEDDINGS_URL = "https://drive.google.com/uc?export=download&id=11o5XVqgWqOm-XEYTueR0F13taQb9KKLw"  # Replace with your actual Google Drive file ID
EMBEDDINGS_FILE = "embeddings.npy"

class PromptSearchCluster:
    def __init__(self, data, embeddings):
        self.data = data
        self.embeddings = embeddings
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def search(self, query, similarity_threshold=0.5):
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding.astype(np.float32)  # Ensure same dtype as embeddings
        
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        above_threshold = similarities >= similarity_threshold
        top_indices = above_threshold.nonzero()[0]
        
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        results = self.data.iloc[top_indices].copy()
        results['similarity'] = similarities[top_indices]
        return results

    def cluster_results(self, results, n_clusters=10):
        if len(results) < n_clusters:
            n_clusters = max(2, len(results) // 2)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        results['cluster'] = kmeans.fit_predict(self.embeddings[results.index])
        
        cluster_centers = kmeans.cluster_centers_
        distances = cosine_similarity(self.embeddings[results.index], cluster_centers)
        results['distance_to_center'] = [distances[i, label] for i, label in enumerate(results['cluster'])]
        
        return results

    def get_cluster_keywords(self, clustered_results, top_n=5):
        keywords = {}
        for cluster in clustered_results['cluster'].unique():
            cluster_prompts = clustered_results[clustered_results['cluster'] == cluster]['prompt']
            cluster_embeddings = self.embeddings[clustered_results[clustered_results['cluster'] == cluster].index]
            centroid = np.mean(cluster_embeddings, axis=0)
            distances = cosine_similarity([centroid], cluster_embeddings)[0]
            top_indices = distances.argsort()[-top_n:][::-1]
            keywords[cluster] = cluster_prompts.iloc[top_indices].tolist()
        return keywords

    def get_cluster_image(self, cluster_results, cluster):
        cluster_data = cluster_results[cluster_results['cluster'] == cluster]
        representative_row = cluster_data.sort_values('distance_to_center', ascending=False).iloc[len(cluster_data) // 2]
        return representative_row['url']

    def get_cluster_description(self, keywords):
        return " | ".join(keywords[:3])

    def get_diverse_prompts(self, cluster_prompts, n=3):
        if len(cluster_prompts) <= n:
            return cluster_prompts.tolist()
        
        prompt_embeddings = self.embeddings[cluster_prompts.index]
        similarities = cosine_similarity(prompt_embeddings)
        
        selected_indices = [random.randint(0, len(cluster_prompts) - 1)]
        
        for _ in range(1, n):
            candidates = list(range(len(cluster_prompts)))
            candidates = [c for c in candidates if c not in selected_indices]
            
            max_min_sim = float('-inf')
            best_candidate = None
            
            for candidate in candidates:
                min_sim = min(similarities[candidate][i] for i in selected_indices)
                if min_sim > max_min_sim:
                    max_min_sim = min_sim
                    best_candidate = candidate
            
            selected_indices.append(best_candidate)
        
        return cluster_prompts.iloc[selected_indices].tolist()


def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB

    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)

@st.cache_resource
def load_data():
    print("Loading data and embeddings...")
    
    # Check if embeddings file exists, if not, download it
    if not os.path.exists(EMBEDDINGS_FILE):
        st.info(f"Downloading embeddings file... This may take a while.")
        download_file(EMBEDDINGS_URL, EMBEDDINGS_FILE)
        st.success("Embeddings file downloaded successfully!")
    
    embeddings = np.load(EMBEDDINGS_FILE)
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    search_cluster = PromptSearchCluster(data, embeddings)
    print("Data and embeddings loaded successfully.")
    return search_cluster

def main():
    st.title("Prompt Search and Clustering")

    search_cluster = load_data()

    query = st.text_input("Enter your search query:")
    similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.01)

    if st.button("Search and Cluster"):
        if query:
            search_results = search_cluster.search(query, similarity_threshold)
            
            if len(search_results) == 0:
                st.write(f"No results found with similarity threshold of {similarity_threshold}.")
            else:
                clustered_results = search_cluster.cluster_results(search_results)
                cluster_keywords = search_cluster.get_cluster_keywords(clustered_results)
                
                st.write(f"Found {len(search_results)} results, clustered into {len(clustered_results['cluster'].unique())} groups:")
                
                for cluster in clustered_results['cluster'].unique():
                    st.subheader(f"Cluster {cluster}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        description = search_cluster.get_cluster_description(cluster_keywords[cluster])
                        st.write(f"Description: {description}")
                        st.write(f"Keywords: {', '.join(cluster_keywords[cluster])}")
                        st.write(f"Number of elements: {sum(clustered_results['cluster'] == cluster)}")
                    
                    with col2:
                        image_url = search_cluster.get_cluster_image(clustered_results, cluster)
                        st.image(image_url, caption="Representative Image", use_column_width=True)
                    
                    st.write("Sample prompts:")
                    cluster_prompts = clustered_results[clustered_results['cluster'] == cluster]['prompt']
                    sample_prompts = search_cluster.get_diverse_prompts(cluster_prompts)
                    for prompt in sample_prompts:
                        st.write(f"- {prompt}")
                    
                    st.markdown("---")

if __name__ == "__main__":
    main()