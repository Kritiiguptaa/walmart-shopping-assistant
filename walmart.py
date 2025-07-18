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

# 🧠 Initialize session state
if "recent_searches" not in st.session_state:
    st.session_state.recent_searches = []

if "wishlist" not in st.session_state:
    st.session_state.wishlist = []

# 📦 Load datasets
df3 = pd.read_csv("useful3.csv")  # ✅ Use cleaned useful3 as priority
df1 = pd.read_csv("useful1.csv")
df2 = pd.read_csv("useful2.csv")
product_df = pd.concat([df3, df1, df2], ignore_index=True)
product_df.drop_duplicates(subset="product_url", keep="first", inplace=True)
product_df.reset_index(drop=True, inplace=True)

faq_df = pd.read_csv("train_expanded.csv")
faq_df.dropna(subset=["question", "answer"], inplace=True)  # Clean nulls

from difflib import get_close_matches

def get_faq_answer(user_query, threshold=0.6):
    # Convert questions to lowercase for comparison
    questions = faq_df["question"].str.lower().tolist()
    matches = get_close_matches(user_query.lower(), questions, n=1, cutoff=threshold)

    if matches:
        matched_question = matches[0]
        matched_row = faq_df[faq_df["question"].str.lower() == matched_question]
        if not matched_row.empty:
            return matched_row["answer"].values[0]
    return None


# 🎨 Define standard colors
standard_colors = [
    "Black", "White", "Red", "Green", "Blue", "Yellow", "Purple", "Orange",
    "Gray", "Grey", "Pink", "Brown", "Beige", "Teal", "Navy", "Maroon",
    "Olive", "Turquoise", "Lavender", "Coral", "Gold", "Silver", "Ivory",
    "Cyan", "Magenta", "Indigo", "Mint", "Peach", "Tan", "Chocolate",
    "Copper", "Burgundy", "Plum", "Rose", "Lilac", "Mauve", "Rust", 
    "Charcoal", "Mustard", "Denim", "Khaki", "Cream", "Sky Blue", 
    "Dark Blue", "Light Blue", "Light Green", "Dark Green", 
    "Dark Gray", "Light Gray", "Hot Pink", "Slate", "Lime", 
    "Aqua", "Sand", "Wine", "Amber"
]
# standard_sizes = set([
#     # Clothing sizes
#     "XS", "S", "M", "L", "XL", "XXL", "XXXL",

#     # Kids / baby sizes
#     "0-3 Months", "3-6 Months", "6-12 Months", "12 Months", "18 Months",
#     "2T", "3T", "4T", "5T", "6T",

#     # Shoe or numeric sizes
#     "5", "6", "7", "8", "9", "10", "11", "12",

#     # Dimensional sizes
#     "50 x 54", "50 x 63", "50 x 84", "50 x 95", "52 x 84", "42 x 84", "84 x 95", "84 x 96", "108 x 84", "63 x 95", "84 x 84",

#     # Mattress / bed sizes
#     "Twin", "Twin XL", "Full", "Queen", "King", "California King",

#     # Misc
#     "Standard", "One Size", "Plus Size"
# ])
# 🎯 Categorized Standard Sizes
grouped_standard_sizes = {
    "Clothing": ["XS", "S", "M", "L", "XL", "XXL", "XXXL", "Plus Size", "Standard", "One Size"],
    "Kids": ["0-3 Months", "3-6 Months", "6-12 Months", "12 Months", "18 Months", "2T", "3T", "4T", "5T", "6T"],
    "Shoes": ["5", "6", "7", "8", "9", "10", "11", "12"],
    "Curtains": ["50 x 54", "50 x 63", "50 x 84", "50 x 95", "52 x 84", "42 x 84", "84 x 95", "84 x 96", "108 x 84", "63 x 95", "84 x 84"],
    "Bedding": ["Twin", "Twin XL", "Full", "Queen", "King", "California King"]
}



# 🎨 Fuzzy match function
def fuzzy_color_match(color_list, selected_color):
    if not selected_color or selected_color == "All":
        return True
    selected_color = selected_color.lower()
    return any(selected_color in c.lower() for c in color_list)

# 📦 Extract clean color values (as list)
def extract_clean_colors(color_string):
    if pd.isna(color_string):
        return []
    if isinstance(color_string, list):
        return color_string
    return [color.strip().title() for color in str(color_string).split(",") if color.strip()]

# 🧃 Sidebar filters
st.sidebar.header("🧃 Filter Products")

# 🎨 Color Dropdown
all_product_colors = set()
product_df["colors"].dropna().apply(lambda val: all_product_colors.update(extract_clean_colors(val)))
dropdown_colors = sorted(set(standard_colors).intersection(all_product_colors))
selected_color = st.sidebar.selectbox("🎨 Select Color", ["All"] + dropdown_colors)

# 📏 Size Filter
# 📦 Flat list of valid sizes
flattened_labeled_sizes = []
for group, sizes in grouped_standard_sizes.items():
    for size in sizes:
        flattened_labeled_sizes.append(f"{group}: {size}")
selected_labeled_sizes = st.sidebar.multiselect("📏 Select Sizes", flattened_labeled_sizes)
selected_sizes = [label.split(": ")[1] for label in selected_labeled_sizes]


# all_valid_sizes = set()
# for sizes in grouped_standard_sizes.values():
#     all_valid_sizes.update(sizes)

# # 🧹 Extract only valid sizes from dataset
# def extract_valid_standard_sizes(size_string):
#     if pd.isna(size_string):
#         return []
#     parts = [s.strip().title() for s in str(size_string).split(",")]
#     return [s for s in parts if s in all_valid_sizes]

# product_df["sizes"] = product_df["sizes"].apply(lambda x: ", ".join(extract_valid_standard_sizes(x)))


# all_sizes = set()
# product_df["sizes"].dropna().apply(lambda val: all_sizes.update(extract_valid_standard_sizes(val)))
# selected_sizes = st.sidebar.multiselect("📏 Select Sizes", sorted(all_sizes))
# st.sidebar.markdown("### 📏 Select Sizes (Grouped)")

# selected_sizes = []
# for group, size_list in grouped_standard_sizes.items():
#     st.sidebar.markdown(f"**{group}**")
#     selected = st.sidebar.multiselect("", options=size_list, key=group)
#     selected_sizes.extend(selected)




# 🔁 Free Returns Filter
return_options = ["All", "Free 90-day returns", "Free 30-day returns"]
selected_return = st.sidebar.selectbox("🔁 Free Return Policy", return_options)

# ⭐ Rating Filter
min_rating = st.sidebar.slider("⭐ Minimum Rating", 0.0, 5.0, 0.0, step=0.5)

# 💰 Price Range Filter
min_price, max_price = st.sidebar.slider(
    "💰 Price Range ($)", 
    0, 
    int(product_df["final_price"].max()) if not product_df.empty else 1000, 
    (0, int(product_df["final_price"].max()) if not product_df.empty else 1000),
    step=1
)


# 🕵️ Recent Searches
st.sidebar.markdown("🕵️ **Recent Searches:**")
for q in reversed(st.session_state.recent_searches[-5:]):
    st.sidebar.markdown(f"- {q}")

# 💛 Wishlist
st.sidebar.markdown("💛 **Wishlist:**")

if st.session_state.wishlist:
    if st.sidebar.button("🧹 Clear Wishlist"):
        st.session_state.wishlist.clear()
        st.rerun()

    for i, item in enumerate(st.session_state.wishlist):
        with st.sidebar.container():
            st.markdown(f"- [{item['product_name']}]({item['product_url']})")
            if st.button("🗑 Remove", key=f"remove_{i}"):
                del st.session_state.wishlist[i]
                st.rerun()
else:
    st.sidebar.info("Your wishlist is empty.")


# 🛍️ Main UI
st.title("🛒 Langchain Walmart Chatbot")
# input_text = st.text_input("💬 Ask me anything! From Walmart products 🛍️ to general help 🤖 — try 'curtains', 'return policy', or 'how to sign up'")
st.markdown("<div style='font-size:18px; font-weight:500;'>💬 Ask me anything! From Walmart products 🛍️ to general help 🤖 — try 'curtains', 'return policy', or 'how to sign up':</div>", unsafe_allow_html=True)
input_text = st.text_input("")



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
    query_price = extract_price(query)  # Price from query like "under 20"
    effective_max_price = min(query_price, max_price) if query_price else max_price

    mask = (
        product_df["product_name"].str.lower().str.contains(query, na=False) |
        product_df["description"].str.lower().str.contains(query, na=False) |
        product_df["category_url"].str.lower().str.contains(query, na=False) |
        product_df["brand"].str.lower().str.contains(query, na=False)
    )

    results = product_df[mask]

    # if max_price:
    #     results = results[results["final_price"] <= max_price]
    # Apply price range filter
    results = results[
        (results["final_price"] >= min_price) &
        (results["final_price"] <= effective_max_price)
    ]


    if selected_color != "All":
        results = results[results["colors"].apply(lambda c: fuzzy_color_match(extract_clean_colors(c), selected_color))]

    if selected_sizes:
        size_pattern = "|".join([re.escape(size) for size in selected_sizes])
        results = results[results["sizes"].str.contains(size_pattern, case=False, na=False)]


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
    if input_text not in st.session_state.recent_searches:
        st.session_state.recent_searches.append(input_text)
    lowered = input_text.lower()

    # 1️⃣ First try answering from FAQ dataset
    faq_answer = get_faq_answer(lowered)
    if faq_answer:
        st.markdown(f"💡 **FAQ Answer:** {faq_answer}")
        st.stop()

    # 2️⃣ Then try reviews / links / product search / LLM...


    # 🌟 Show reviews if asked
    if is_rating_query(lowered):
        reviews = fetch_reviews_from_serpapi(input_text, os.getenv("SERPAPI_KEY"))
        if reviews:
            st.subheader("⭐ Customer Reviews:")
            for review in reviews:
                st.markdown(f"""- **{review.get('rating', '?')}★** by *{review.get('reviewer_name', 'Anonymous')}*  \
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
        for i, (_, row) in enumerate(matched.iterrows()):
            with st.container():
                st.image(row["image_url"], width=120)

                with st.form(f"wishlist_form_{i}"):
                    st.markdown(f"""
                        **{row['product_name']}**  
                        {row['description'][:300]}...  
                        💰 **Price:** ${row['final_price']}  
                        🏷️ **Brand:** {row['brand']}  
                        🔁 **Returns:** {row['free_returns']}  
                        🎨 **Colors:** {', '.join(row['colors']) if isinstance(row['colors'], list) else row['colors'] or "N/A"}  
                        📏 **Sizes:** {row['sizes']}  
                        ⭐ **Rating:** {row['rating']} ({row['review_count']} reviews)  
                        🔗 [View Product]({row['product_url']})
                    """)

                    submitted = st.form_submit_button("❤️ Add to Wishlist")
                    if submitted:
                        product = row.to_dict()
                        if not any(item["product_url"] == product["product_url"] for item in st.session_state.wishlist):
                            st.session_state.wishlist.append(product)
                            st.rerun()  # 🔁 Force immediate UI update
                        else:
                            st.info("ℹ️ Already in wishlist.")






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
