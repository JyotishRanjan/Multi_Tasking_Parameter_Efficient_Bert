# Multi-Task Fine-Tuning of BERT with LoRA Adapters

## Overview

This project demonstrates multi-task fine-tuning of BERT using LoRA (Low-Rank Adaptation) adapters. The fine-tuned model handles three tasks: sentiment analysis, semantic similarity, and question answering. By using LoRA adapters, we achieve efficient model training with significantly fewer trainable parameters compared to full model fine-tuning.

## Tasks

1. **Sentiment Analysis** - Classify text into positive or negative sentiments.
2. **Semantic Similarity** - Measure the similarity between text pairs.
3. **Question Answering** - Provide answers to questions based on context.

## Datasets

- **SST-2**: Stanford Sentiment Treebank for sentiment analysis.
- **STS**: Semantic Textual Similarity dataset for semantic similarity.
- **SQuAD**: Stanford Question Answering Dataset for question answering.

## Results

- **Number of Trainable Parameters**: 2000 times lower than that of the full fine-tuned model.
- **Model Performance**: Achieved comparable results to the fully fine-tuned BERT model across all tasks.

## Observations

The use of LoRA adapters allows for efficient fine-tuning with a reduced number of trainable parameters. Despite this, the model performs comparably to a fully fine-tuned BERT model, making it a cost-effective solution for multi-task learning.

## Uses

This multi-task fine-tuned model can be deployed in various applications where large language models (LLMs) are needed for related activities:

- **Customer Support**: 
  - **Sentiment Analysis**: Analyze the sentiment of customer responses.
  - **Semantic Similarity**: Identify similar previously asked questions and route customers accordingly.
  - **Question Answering**: Provide answers to customer queries from an FAQ database.

## Installation

```bash
# Clone the repository
git clone https://github.com/JyotishRanjan/Multi_Tasking_Parameter_Efficient_Bert.git

```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [LoRA](https://arxiv.org/abs/2006.10041) for efficient parameter adaptation.
- [BERT](https://arxiv.org/abs/1810.04805) for pre-trained transformer architecture.

## Contact

For any questions or feedback, please contact [MAIL](mailto:jyovicran@gmail.com).
