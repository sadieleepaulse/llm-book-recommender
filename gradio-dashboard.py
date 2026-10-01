import numpy as np
import pandas as pd
import gradio as gr

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"].fillna("") + "&fife=w800"
books.loc[books["thumbnail"].isna(), "large_thumbnail"] = "cover-not-found.jpg"

# Keep one book description per document; chunk_size=0 is invalid.
raw_text = TextLoader("tagged_description.txt", encoding="utf-8").load()[0].page_content
documents = [
    Document(page_content=line)
    for line in raw_text.splitlines()
    if line.strip()
]

db_books = Chroma.from_documents(
    documents,
    embedding=HuggingFaceEmbeddings(),
)


def retrieve_semantic_recommendations(
    query: str,
    category: str = "All",
    tone: str = "All",
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    docs = db_books.similarity_search(query, k=initial_top_k)

    # Each document starts with an ISBN; remove possible surrounding quotes.
    isbn_to_rank = {
        int(doc.page_content.split()[0].strip('" .')): rank
        for rank, doc in enumerate(docs)
    }

    book_recs = books[books["isbn13"].isin(isbn_to_rank)].copy()
    book_recs["_rank"] = book_recs["isbn13"].map(isbn_to_rank)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    tone_columns = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness",
    }

    if tone in tone_columns:
        book_recs = book_recs.sort_values(
            by=[tone_columns[tone], "_rank"],
            ascending=[False, True],
        )
    else:
        book_recs = book_recs.sort_values("_rank")

    return book_recs.head(final_top_k).drop(columns="_rank")


def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = str(row["description"])
        truncated_description = " ".join(description.split()[:30]) + "..."

        authors = str(row["authors"])
        author_names = authors.split(";")
        authors_str = (
            f"{author_names[0]} and {author_names[1]}"
            if len(author_names) == 2
            else authors
        )

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results


categories = ["All"] + sorted(
    books["simple_categories"].dropna().unique().tolist()
)
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness",
        )
        category_dropdown = gr.Dropdown(
            choices=categories, label="Select a category:", value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones, label="Select an emotional tone:", value="All"
        )

    submit_button = gr.Button("Find recommendations")
    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=4, rows=4)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
    )

if __name__ == "__main__":
    dashboard.launch(inbrowser=True)