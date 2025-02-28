# Title Generator using LLM

A machine learning project for generating news article titles from content using a fine-tuned Flan-T5-small model.

## Project Overview

This project uses a fine-tuned Google Flan-T5-small model to generate concise, relevant titles for news articles based on their content. The model was trained on a dataset of space news articles, making it particularly effective for generating space-related news titles, though it can be adapted for other domains.

## Features

- Automated title generation for news articles
- Fine-tuned on space news data
- Built with Hugging Face Transformers library
- Handles input of varying lengths
- Deployed to Hugging Face Hub for easy access

## Dataset

The project uses the [Space News Dataset](https://www.kaggle.com/datasets/patrickfleith/space-news-dataset) from Kaggle, which contains articles related to space exploration, astronomy, and related topics. The dataset provides pairs of article content and corresponding titles, making it suitable for this sequence-to-sequence learning task.

## Model Architecture

The project utilizes Google's Flan-T5-small, a sequence-to-sequence model pre-trained on a diverse mix of tasks. The model was fine-tuned on our specific task of generating titles from article content.

- Base model: `google/flan-t5-small`
- Max input length: 512 tokens
- Max output length: 128 tokens

## Installation and Setup

### Prerequisites

- Python 3.7+
- PyTorch
- Transformers
- Datasets
- KaggleHub

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AhemdMahmoud/Title_Generator-_Fn.git
   cd title-generator
   ```

2. Install dependencies:
   ```bash
   pip install torch transformers datasets pandas numpy kagglehub
   ```

3. Download the dataset:
   ```python
   import kagglehub
   path = kagglehub.dataset_download("patrickfleith/space-news-dataset")
   ```

## Usage

### Loading the Model

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("yourusername/title-generator")
model = AutoModelForSeq2SeqLM.from_pretrained("yourusername/title-generator")
```

### Generating Titles

```python
def generate_title(article_content, max_length=128):
    # Tokenize the input
    inputs = tokenizer(article_content, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    
    # Generate the title
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    
    # Decode the generated title
    generated_title = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_title

# Example usage
article_content = "NASA announced today that they have discovered evidence of water on Mars. The discovery was made using the Perseverance rover which landed on the planet in February 2021..."
title = generate_title(article_content)
print(f"Generated title: {title}")
```

## Training Process

The model was trained with the following configuration:

- Learning rate: 5e-5
- Batch size: 8
- Epochs: 3
- Weight decay: 0.01
- Evaluation strategy: Evaluate at each epoch

The training process included:
1. Data preprocessing and cleaning
2. Removing duplicates and handling missing values
3. Splitting into train (80%) and test (20%) sets
4. Tokenization with special handling for input and target sequences
5. Fine-tuning using the Hugging Face Trainer API

## Model Performance

After training for 3 epochs, the model achieved the following evaluation metrics:
- [Add your evaluation metrics here, e.g., ROUGE score, BLEU score, etc.]

## Future Improvements

- Fine-tune on a larger and more diverse dataset
- Experiment with larger models (Flan-T5-base, Flan-T5-large)
- Implement beam search and other decoding strategies for better title generation
- Add a web interface for easy title generation
- Extend to other languages or specialized domains

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[Specify your license here]

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for the Transformers library
- [Google](https://ai.google/research/) for the Flan-T5 model
- [Kaggle](https://www.kaggle.com/) and [Patrick Fleith](https://www.kaggle.com/patrickfleith) for the Space News Dataset
