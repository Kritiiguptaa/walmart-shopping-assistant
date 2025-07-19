# 🛒 Walmart AI Shopping Assistant
A smart AI-powered chatbot that helps users explore Walmart products, apply filters, and get answers to common customer queries — all in a conversational interface.

## 🚀 Features
- 🔍 **Product Search:** Type in product names like "blankets", "tshirts", "curtains" and get top relevant results.
- 🎨 **Smart Filtering:** Filter products by color, size, price range, minimum rating, and return policy.
- ❤️ **Wishlist Functionality:** Add and remove items to a personal wishlist.
- 🧠 **AI-Powered Q&A:** Ask general questions like "How can I track my order?" or "How do I sign up?" — the bot will respond just like a customer support agent.
- 🧾 **Recent Searches:** Saves and displays your most recent queries for quick access.

## 🛠️ Tech Stack
- **Python**
- **Streamlit** – For building a clean, interactive web UI
- **LangChain** – To structure prompts and manage responses
- **Pandas** – For handling and filtering CSV datasets

## 📊 Dataset
Used a cleaned and structured CSV dataset simulating Walmart product listings to simulate product searches and filters.

## 📌 Why This Project?
Walmart receives millions of queries every day. Traditional search bars lack personalization and instant help. This chatbot bridges the gap between users and product discovery by offering:
- Instant query resolution
- Product recommendations
- Filter-based personalization
- Automation of general customer service queries

## 🎥 Demo
live link: https://walmart-shopping-assistant-ipqwfbfelclnvfsbcfbaau.streamlit.app/
Youtube Link: https://www.youtube.com/watch?v=8KmgYjCvMjs

## 📂 How to Run Locally
```bash
git clone https://github.com/Kritiiguptaa/walmart-shopping-assistant.git
cd walmart-shopping-assistant
pip install -r requirements.txt
streamlit run app.py
