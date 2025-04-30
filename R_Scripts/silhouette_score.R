library(readr)
library(cluster)

# Load the embeddings
sarcasm_embeddings <- read_csv('sarcasm_embeddings.csv')
ethos_embeddings <- read_csv('ethos_embeddings.csv')

# Create labels
sarcasm_labels <- rep('Sarcasm', nrow(sarcasm_embeddings))
ethos_labels <- rep('Hate', nrow(ethos_embeddings))

# Combine embeddings and labels to perform t-SNE comparison
combined_embeddings <- rbind(sarcasm_embeddings, ethos_embeddings)
combined_labels <- c(sarcasm_labels, ethos_labels)

# Remove duplicates and downsample data
sarcasm_embeddings_unique <- sarcasm_embeddings[!duplicated(sarcasm_embeddings), ]
combined_embeddings_unique <- combined_embeddings[!duplicated(combined_embeddings), ]
combined_labels_unique <- combined_labels[!duplicated(combined_embeddings)]

# Perform dimensionality reduction
set.seed(42)
subset_idx <- sample(1:nrow(combined_embeddings_unique), size = 1000)
combined_embeddings_subset <- combined_embeddings_unique[subset_idx, ]
combined_labels_subset <- combined_labels_unique[subset_idx]

# Encode labels numerically
label_numeric <- as.numeric(as.factor(combined_labels_subset))

# Calcuate distance between points
dist_matrix <- dist(as.matrix(combined_embeddings_subset))

# Compute silhouette
silhouette_scores <- silhouette(label_numeric, dist_matrix)
mean(silhouette_scores[, "sil_width"])
