# BERT-Base-Uncased Quantized Model for social media sentiment analysis

This repository hosts a quantized version of the **bert-base-uncased** model, fine-tuned for social media sentiment analysis tasks. The model has been optimized for efficient deployment while maintaining high accuracy, making it suitable for resource-constrained environments.

## Model Details

- **Model Architecture:** BERT Base Uncased  
- **Task:** Social Media Sentiment Analysis
- **Dataset:** Social Media Sentiments Analysis Dataset [Kaggle]  
- **Quantization:** Float16  
- **Fine-tuning Framework:** Hugging Face Transformers  

## Usage

### Installation

```sh
pip install transformers torch
```


### Loading the Model

```python

from transformers import BertForSequenceClassification, BertTokenizer
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load quantized model
model_name = "AventIQ-AI/bert-social-media-sentiment-analysis"
model = BertForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = BertTokenizer.from_pretrained(model_name)

#Function to make analysis
def predict_sentiment(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Move tensors to GPU if available
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted class
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Map back to sentiment labels
    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_labels[predicted_class]

# Define a test sentence
test_sentence = "Spending time with family always brings me so much joy."
print(f"Predicted Sentiment: {predict_sentiment(text)}")
```

## Performance Metrics

- **Accuracy:** 0.82
- **Precision:** 0.68
- **Recall:** 0.82
- **F1 Score:** 0.73

## Fine-Tuning Details

### Dataset

The dataset is taken from Kaggle Social Media Sentiment Analysis.

### Training

- Number of epochs: 6  
- Batch size: 8  
- Evaluation strategy: epoch  
- Learning rate: 3e-5  

### Quantization

Post-training quantization was applied using PyTorch's built-in quantization framework to reduce the model size and improve inference efficiency.

## Repository Structure

```
.
├── model/               # Contains the quantized model files
├── tokenizer_config/    # Tokenizer configuration and vocabulary files
├── model.safensors/     # Fine Tuned Model
├── README.md            # Model documentation
```

## Limitations

- The model may not generalize well to domains outside the fine-tuning dataset.  
- Quantization may result in minor accuracy degradation compared to full-precision models.  

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.

