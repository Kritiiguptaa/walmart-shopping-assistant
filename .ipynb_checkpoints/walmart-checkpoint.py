import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import re
import requests

# 🌱 Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LangChain_API_KEY"] = os.getenv("LangChain_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# 📦 Load datasets
df3 = pd.read_csv("useful3.csv")  # ✅ Use cleaned useful3 as priority
df1 = pd.read_csv("useful1.csv")
df2 = pd.read_csv("useful2.csv")
product_df = pd.concat([df3, df1, df2], ignore_index=True)
product_df.drop_duplicates(subset="product_url", keep="first", inplace=True)
product_df.reset_index(drop=True, inplace=True)

# 🔍 Sidebar filters
st.sidebar.header("🧃 Filter Products")

# 🎨 Clean & extract unique colors
def extract_clean_colors(color_string):
    if pd.isna(color_string):
        return []
    return [color.strip().title() for color in color_string.split(",") if color.strip()]

color_set = set()
product_df["colors"].dropna().apply(lambda val: color_set.update(extract_clean_colors(val)))
unique_colors = sorted(color_set)
selected_color = st.sidebar.selectbox("🎨 Select Color", ["All"] + unique_colors)

# 📏 Size Filter
all_sizes = set()
product_df["sizes"].dropna().apply(lambda val: all_sizes.update(map(str.strip, str(val).split(","))))
selected_sizes = st.sidebar.multiselect("📏 Select Sizes", sorted(all_sizes))

# 🔁 Free Returns Filter
return_options = ["All", "Free 90-day returns", "Free 30-day returns"]
selected_return = st.sidebar.selectbox("🔁 Free Return Policy", return_options)

# ⭐ Rating Filter
min_rating = st.sidebar.slider("⭐ Minimum Rating", 0.0, 5.0, 0.0, step=0.5)

# 🛍️ Main UI
st.title("🛒 Langchain Walmart Chatbot")
input_text = st.text_input("Ask about Walmart products (e.g., 'curtains under 20'): ")

# 🔎 Extract price
def extract_price(query):
    match = re.search(r'under\s*(\d+)', query.lower())
    return float(match.group(1)) if match else None

# 💬 Rating/review detection
def is_rating_query(query):
    return any(word in query.lower() for word in ["rating", "ratings", "review", "reviews"])

# 🌐 SerpApi review fetch
def fetch_reviews_from_serpapi(product_name, serpapi_key):
    params = {
        "engine": "walmart_reviews",
        "q": product_name,
        "api_key": serpapi_key
    }
    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code == 200:
        return response.json().get("reviews", [])[:5]
    return []

# 🔍 Product search logic
def search_products(query):
    query = query.lower()
    max_price = extract_price(query)

    mask = (
        product_df["product_name"].str.lower().str.contains(query, na=False) |
        product_df["description"].str.lower().str.contains(query, na=False) |
        product_df["category_url"].str.lower().str.contains(query, na=False) |
        product_df["brand"].str.lower().str.contains(query, na=False)
    )

    results = product_df[mask]

    if max_price:
        results = results[results["final_price"] <= max_price]

    if selected_color != "All":
        results = results[results["colors"].str.contains(selected_color, case=False, na=False)]

    if selected_sizes:
        size_pattern = "|".join([re.escape(s) for s in selected_sizes])
        results = results[results["sizes"].str.contains(size_pattern, na=False)]

    if selected_return != "All":
        results = results[results["free_returns"] == selected_return]

    if min_rating > 0:
        results = results[results["rating"].fillna(0) >= min_rating]

    return results.head(10)

# 🧠 LangChain LLM Setup
prompt = ChatPromptTemplate.from_template(
    "You're a helpful assistant for Walmart product info. Use this product data:\n\n{product_data}\n\nQuestion: {question}"
)
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="mistralai/mixtral-8x7b-instruct",
    temperature=0.2
)
chain = prompt | llm | StrOutputParser()

# 🤖 Main logic
if input_text:
    lowered = input_text.lower()

    # 🌟 Show reviews if asked
    if is_rating_query(lowered):
        reviews = fetch_reviews_from_serpapi(input_text, os.getenv("SERPAPI_KEY"))
        if reviews:
            st.subheader("⭐ Customer Reviews:")
            for review in reviews:
                st.markdown(f"""- **{review.get('rating', '?')}★** by *{review.get('reviewer_name', 'Anonymous')}*  
{review.get('snippet') or review.get('body', '')}""")
            st.stop()

    # 🔗 Show product URL if asked
    if any(kw in lowered for kw in ["product link", "product url", "where to buy", "link", "url"]):
        cleaned_query = re.sub(r"(product link|product url|where to buy|link|url)", "", lowered, flags=re.IGNORECASE).strip()
        found = product_df[product_df["product_name"].str.lower().str.contains(cleaned_query, na=False)].head(6)
        if not found.empty:
            st.write("🔗 **Product Link(s):**")
            for _, row in found.iterrows():
                st.markdown(f"- **{row['product_name']}** → [View Product]({row['product_url']})")
        else:
            st.warning("❌ Sorry, no matching product.")
        st.stop()

    # 🛍️ Product search
    matched = search_products(input_text)
    if not matched.empty:
        st.write("🛍️ Products matching your query:")
        for _, row in matched.iterrows():
            st.image(row["image_url"], width=120)
            st.markdown(f"""
                **{row['product_name']}**  
                {row['description'][:300]}...  
                💰 **Price:** ${row['final_price']}  
                🏷️ **Brand:** {row['brand']}  
                🔁 **Returns:** {row['free_returns']}  
                🎨 **Colors:** {row['colors']}  
                📏 **Sizes:** {row['sizes']}  
                ⭐ **Rating:** {row['rating']} ({row['review_count']} reviews)  
                🔗 [View Product]({row['product_url']})
                """)
    else:
        # Fallback to LLM response
        try:
            response = chain.invoke({
                "question": input_text,
                "product_data": product_df.sample(10).to_string(index=False)
            })
            st.markdown(response)
        except Exception as e:
            st.error(f"🤖 LLM Error: {str(e)}")
