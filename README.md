# Project: Ghost Clusters (Work In Progress)

This project is designed to cluster AI use cases based on their titles and descriptions and perform visual analysis of the resulting clusters. The clustering is performed using SBERT for embedding generation and DBSCAN for clustering. We also provide a Jupyter Notebook for further exploration and visualization of the clustered data.

## Overview

The project includes:
- **`bclustering.py`**: A Python script that reads a CSV file of project data, generates embeddings using SBERT, clusters the projects using DBSCAN, and saves the clustered data to a new CSV file.
- **`analysis.ipynb`**: A Jupyter Notebook that provides visualizations and analyses of the clustered data, including heatmaps, network graphs, and other insights.

## Requirements

Ensure the following Python libraries are installed:
- `pandas`
- `sentence-transformers`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `networkx`
- `nltk`

To install these libraries, run:
```bash
pip install pandas sentence-transformers scikit-learn matplotlib seaborn networkx nltk
```

## Usage

### Running `bclustering.py`

`bclustering.py` is a Python script that reads a CSV file containing project data, generates embeddings using SBERT, and clusters the data using DBSCAN. The script takes the path to the CSV file and the names of the columns containing the project title and description as arguments.

#### Command-Line Usage
```bash
python bclustering.py <path_to_csv> <title_column_name> <description_column_name>
```

### Script Features
- **Embedding Generation**: Uses Sentence-BERT (`all-MiniLM-L12-v2`) to create embeddings from the combined title and description.
- **Clustering**: Applies DBSCAN with cosine similarity to cluster the embeddings.
- **Output**: Saves the clustered data to a new CSV file with an additional column indicating the cluster label.

### Example
```bash
python bclustering.py projects.csv title description
```

## Visual Analysis

The `analysis.ipynb` notebook provides a detailed analysis of the clustered data, including:
- **Cluster Distribution**: Visualize the number of projects per cluster using bar plots.
- **Heatmap**: Generate a heatmap that shows the number of projects per department and cluster.
- **Network Graphs**: Visualize department interactions based on shared clusters with node size proportional to the number of projects and edge thickness representing the number of shared clusters.

## Notes

- The `analysis.ipynb` notebook assumes the output CSV file from `bclustering.py` is available.
- Custom stopwords, including `<title>` and `<description>` tags, are added to improve cluster insights.

## Future Enhancements

- Add more clustering algorithms for comparison.
- Implement interactive visualizations using `plotly` or `bokeh` for a more dynamic analysis experience.


