# Large Language Models

Best resource for [this](https://www.youtube.com/watch?v=zjkBMFhNj_g)
They are trained on a large chunk of test to gain auto-completing sentences capabilities. Fine tune the model with high quality QnA responses to obtain assistant model.

## My learnings

I tried to fine tune my model on Mac m1 pro (16gb). Here are my learnings: only way to do this is via a fine tuning method called lora/qlora. For doing this also you need powerful GPUs. [Then I found a way to do it on Apple Silicon.](https://github.com/ml-explore/mlx). I followed a nice blog [series](https://apeatling.com/articles/part-4-testing-and-interacting-with-your-fine-tuned-llm/) to fine tune my model.

Tried with Mistral-7b model, took about 4 hours to fine tune with 10k dataset and 100 iterations. Tried running the fine tuned model, turns out 7b parameter model is 13gb which too much for my 16gb machine. This causes to do a lot of hot swaps in memory resulting in extremely slow performance.

Then I found about [quantized models (4bit)](https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.2-4-bit), which are much smaller and workable, this time fine tuning took under 30minutes and performance was bearable on the fine tuned model. If the model was not extremely fast on my machine, I realized that the machines used by large companies are way more suited for this kind of work. Then I cam about this blog that discusses about [hosting charges](https://blog.lytix.co/posts/self-hosting-llama-3) for such a model. Reading this article made me realize, that fine tuning my own model unless it is absolutely required or we have a team to maintain in house servers, is not a good idea.

I am better off using something like [Replicate](https://replicate.com/), if I just want something really specific to my use-case.

Also for more info, check out [reddit](https://www.reddit.com/r/LocalLLaMA/search/?q=fine-tuning&sort=top)