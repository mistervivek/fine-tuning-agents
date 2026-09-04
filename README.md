# efficient-AI-agents.

## Overview
This project is a customized AI banking assistant. It demonstrates how to use **Supervised Fine-Tuning (SFT)** to make a general AI model act as an expert in a specific domain. 

Instead of relying on long, expensive prompts to tell the AI how to act, this project trains the model on a specialized dataset. This makes the AI faster, cheaper to run, and more accurate when answering customer support questions for a bank.

## Key Features
*   **Domain Expert:** Understands banking rules and gives safe, accurate answers.
*   **Fewer Mistakes:** Reduces "hallucinations" (made-up answers) in financial topics.
*   **Structured Output:** Reliably generates correct formats like JSON or SQL when asked.
*   **Cost Effective:** Uses fewer tokens because it does not need a massive set of instructions for every message.

## How Works
1.  **Data Preparation:** We created a high-quality dataset that shows the AI exactly how a banking assistant should talk and solve problems.
2.  **Fine-Tuning:** We trained a base AI model on this specific dataset.
3.  **Integration:** We connected the fine-tuned model to a workflow to act as a customer support agent.

## Prerequisites
*   Python 3.10+
*   Jupyter Notebook (for exploring the data and running tests)
