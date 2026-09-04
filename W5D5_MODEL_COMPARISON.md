# W5D5 Model Comparison

## Models Tested

- llama3.2:3b
- qwen2.5:3b

## Questions Used

The same three questions were given to both models:

1. Explain RAG in simple terms.
2. What is semantic search?
3. What is overfitting?

## Comparison

| Criteria            | llama3.2:3b                                                     | qwen2.5:3b                                                  |
| ------------------- | --------------------------------------------------------------- | ----------------------------------------------------------- |
| RAG explanation     | Incorrectly interpreted RAG as "Relative Acquisition Gain"      | Correctly explained RAG as "Retrieval-Augmented Generation" |
| Semantic search     | Response was not completed because of a timeout                 | Provided a clear explanation of semantic search             |
| Overfitting         | Provided a detailed explanation                                 | Provided a concise and clear explanation                    |
| Response length     | Generally detailed                                              | Generally concise                                           |
| Response speed      | Slower in several tests                                         | Faster in several tests                                     |
| Overall observation | Detailed responses but one incorrect RAG answer and one timeout | Clear responses and correctly identified RAG                |

## Observations

The two models produced different responses even when given the same questions and system prompt.

For the RAG question, qwen2.5:3b provided the correct meaning of Retrieval-Augmented Generation, while llama3.2:3b produced an incorrect interpretation.

For the overfitting question, both models provided relevant explanations, although llama3.2:3b was more detailed and qwen2.5:3b was more concise.

During the semantic search comparison, llama3.2:3b encountered a timeout, while qwen2.5:3b successfully generated a response.

## Conclusion

Both models were successfully tested locally using Ollama. The comparison shows that response quality, accuracy, response length, and response time can vary between local LLMs. The results were documented based on the actual outputs from the tests.
