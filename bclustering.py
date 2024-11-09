import pandas as pd
import sys
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

def main(csv_path, title_col, description_col):
    # Load the data from CSV
    df = pd.read_csv(csv_path)
    
    # Check if the specified columns exist in the DataFrame
    if title_col not in df.columns or description_col not in df.columns:
        print(f"Error: Columns '{title_col}' and/or '{description_col}' not found in the CSV file.")
        sys.exit(1)

    # Concatenate 'title' and 'description' with tags
    df['combined_text'] = '<title> ' + df[title_col].fillna('') + ' </title> <description> ' + df[description_col].fillna('') + ' </description>'

    # Load the pre-trained SBERT model
    model = SentenceTransformer('all-MiniLM-L12-v2')

    # Generate embeddings for the 'combined_text' column
    embeddings = model.encode(df['combined_text'].tolist(), convert_to_tensor=False)

    # Clustering using DBSCAN
    dbscan = DBSCAN(eps=0.4, min_samples=2, metric='cosine')
    labels = dbscan.fit_predict(embeddings)

    # Add the cluster labels to the DataFrame
    df['cluster_label'] = labels

    # Print the number of unique clusters
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Number of clusters found: {num_clusters}")

    # Save the DataFrame with cluster labels to a new CSV
    output_path = csv_path.replace('.csv', '_clustered.csv')
    df.to_csv(output_path, index=False)
    print(f"Clustered data saved to: {output_path}")

if __name__ == "__main__":
    # Make sure the script is called with the correct number of arguments
    if len(sys.argv) != 4:
        print("Usage: python script.py <path_to_csv> <title_column_name> <description_column_name>")
        sys.exit(1)

    # Get arguments from command line
    csv_path = sys.argv[1]
    title_col = sys.argv[2]
    description_col = sys.argv[3]

    # Run the main function
    main(csv_path, title_col, description_col)
