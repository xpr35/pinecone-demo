from sentence_transformers import SentenceTransformer
from PIL import Image
from pinecone import Pinecone

INDEX_NAME = "animals"
PINECONE_API_KEY = ""


def main():
    model = SentenceTransformer('clip-ViT-B-32')  # 'clip-ViT-B-32-multilingual-v1'

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    image_to_search = Image.open('embedding_test_cat.jpeg')
    image_embedding = model.encode(sentences=image_to_search, normalize_embeddings=False)
    image_result = index.query(
        namespace="ns1",
        vector=image_embedding.tolist(),
        top_k=5,
        include_values=True,
        include_metadata=True
    )
    print(f"Found {[x['id'] for x in image_result['matches']]}")

    text_to_search = "熊"
    text_embedding = model.encode(sentences=text_to_search, normalize_embeddings=False)
    text_result = index.query(
        namespace="ns1",
        vector=text_embedding.tolist(),
        top_k=5,
        include_values=True,
        include_metadata=True
    )
    print(f"Found {[x['id'] for x in text_result['matches']]} for text {text_to_search}")


if __name__ == '__main__':
    main()
