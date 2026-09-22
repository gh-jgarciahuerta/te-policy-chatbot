# 💼 T&E Policy Chatbot

## 📌 Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot built to assist with questions related to Travel & Expense (T&E) policies.

It is designed to:

* 💬 Help employees quickly find answers about T&E policy rules
* ✅ Point users to the exact policy section that supports each answer

---

## 🧠 Use Cases

* Users can ask:

  * “Is this expense reimbursable?”
  * “What are the limits for meals or travel?”
  * “What do I do if I'm missing a receipt?”

---

## ⚙️ Tech Stack

* 🐍 Python
* 🔗 LangChain
* 🧠 Claude (via AWS Bedrock)
* 📚 Vector Database (for semantic search)

---

## 🔍 How It Works

1. 📄 Policy documents are stored in `code/` alongside the scripts, numbered in pipeline order
2. 🔢 Documents are converted into embeddings and stored in a vector database
3. 🔎 User queries are matched against relevant policy sections
4. 💡 The LLM generates context-aware responses

---

## 📁 Project Structure

* `code/` – Pipeline scripts and their input/output files, numbered in the order they run
* `vectorstore/` – Generated embeddings (ignored in Git)
* `Requirements.txt` – Project dependencies

---

## 🚀 Setup

1. Create and activate a virtual environment
2. Install dependencies:

   ```
   pip install -r Requirements.txt
   ```
3. Add your `.env` file with required API keys
4. Run the application

---

## ⚠️ Notes

* 🚫 `vectorstore/` should not be committed (generated locally)
* 🔒 Do not commit sensitive policy data files (they are git-ignored under `code/`)

---
