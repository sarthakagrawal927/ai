from smolagents import LiteLLMModel
from transformers import pipeline
import time

classifier = pipeline("sentiment-analysis")
generator = pipeline("text-generation")
ner = pipeline("ner", grouped_entities=True)
question_answerer = pipeline("question-answering")

model = LiteLLMModel(
    model_id="ollama_chat/qwen2:7b",  # Or try other Ollama-supported models
    api_base="http://127.0.0.1:11434",  # Default Ollama local server
    num_ctx=8192,
)


def time_and_print(task_name, func, *args, **kwargs):
    """Execute a function, time it, and print the results."""
    print("\n" + "=" * 60)
    print(f"Running {task_name}...")
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    print(f"Result: {result}")
    print(f"Time: {elapsed:.3f}s")
    return result


def main():
    # LLM Generation
    time_and_print(
        "LiteLLM Model",
        model.generate,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello, how are you?"}],
            }
        ],
        kwargs={"max_tokens": 100},
    )

    # Sentiment Analysis
    time_and_print(
        "Sentiment Analysis",
        classifier,
        "I've been waiting for a HuggingFace course my whole life.",
    )

    # Text Generation
    time_and_print(
        "Text Generation",
        generator,
        "In this course, we will teach you how to",
    )

    # Named Entity Recognition
    time_and_print(
        "NER",
        ner,
        "My name is Sylvain and I work at Hugging Face in Brooklyn.",
    )

    # Question Answering
    time_and_print(
        "Question Answering",
        question_answerer,
        question="Where do I work?",
        context="My name is Sylvain and I work at Hugging Face in Brooklyn",
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
