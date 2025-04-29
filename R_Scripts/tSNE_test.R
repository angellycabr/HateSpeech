library(readr)
library(Rtsne)

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
combined_embeddings_unique <- combined_embeddings[!duplicated(combined_embeddings), ]
combined_labels_unique <- combined_labels[!duplicated(combined_embeddings)]

# Perform dimensionality reduction
set.seed(42)
subset_idx <- sample(1:nrow(combined_embeddings_unique), size = 1000)
combined_embeddings_subset <- combined_embeddings_unique[subset_idx, ]
combined_labels_subset <- combined_labels_unique[subset_idx]

# Run t-SNE on subset
tsne_out <- Rtsne(as.matrix(combined_embeddings_subset))

# Plot
library(ggplot2)
ggplot(data.frame(x = tsne_out$Y[,1], 
                  y = tsne_out$Y[,2], 
                  label = combined_labels_subset),
       aes(x = x, y = y, color = label)) +
  geom_point(alpha = 0.7) +
  labs(title = "t-SNE of Sarcasm vs. Hate Embeddings") +
  theme_minimal()
