# 💼 T&E Policy Chatbot

## 📌 Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot built to assist with questions related to Travel & Expense (T&E) policies.

It is designed to:

* 💬 Help employees quickly find answers about T&E policy rules
* ✅ Support approvers in determining how to handle issues with expense reports

---

## 🧠 Use Cases

* **Employees** can ask:

  * “Is this expense reimbursable?”
  * “What are the limits for meals or travel?”

* **Approvers** can ask:

  * “How should I handle a missing receipt?”
  * “Is this expense compliant with policy?”

---

## ⚙️ Tech Stack

* 🐍 Python
* 🔗 LangChain
* 🧠 Claude (via AWS Bedrock)
* 📚 Vector Database (for semantic search)

---

## 🔍 How It Works

1. 📄 Policy documents are stored in `data/`
2. 🔢 Documents are converted into embeddings and stored in a vector database
3. 🔎 User queries are matched against relevant policy sections
4. 💡 The LLM generates context-aware responses

---

## 📁 Project Structure

* `code/` – Core application logic
* `data/` – Source policy documents
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
* 🔒 Do not include sensitive data in `data/`

---
