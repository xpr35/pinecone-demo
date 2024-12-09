import os
import glob
import time

from PIL import Image

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

CHUNK_SIZE = 100
INDEX_NAME = "animals"
PATH_TO_IMAGES = "archive/animals/animals/**/*.jp*g"
PINECONE_API_KEY = ""


def chunk_list(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def main():
    start_time = time.perf_counter()

    model = SentenceTransformer('clip-ViT-B-32')

    model_load_duration = time.perf_counter() - start_time
    print(f'Duration load model = {model_load_duration}')

    normalize_emb = False
    filenames = glob.glob(PATH_TO_IMAGES, recursive=True)

    animals_embeddings = []
    animals_names = set()

    embedding_generation_start = time.perf_counter()
    image_count = 1
    for image_name in filenames:
        wrapped_image = Image.open(image_name)
        embedding = model.encode(sentences=wrapped_image, normalize_embeddings=normalize_emb)
        animals_names.add(image_name.split('/')[3])
        animals_embeddings.append(
            {
                "id": os.path.splitext(os.path.basename(image_name))[0],
                "values": embedding
            }
        )
        print(f'{image_count}/{len(filenames)} done')
        image_count+=1

    embedding_generation_duration = time.perf_counter() - embedding_generation_start
    print(f'Total embedding duration = {embedding_generation_duration}, average embedding duration = {embedding_generation_duration / len(filenames)}')

    pc = Pinecone(api_key=PINECONE_API_KEY)

    try:
        pc.create_index(
            name=INDEX_NAME,
            dimension=512,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    except:
        pass

    index = pc.Index(INDEX_NAME)
    for chunk in chunk_list(animals_embeddings, CHUNK_SIZE):
        index.upsert(
            vectors=chunk,
            namespace="ns1"
        )
    print(animals_names)


if __name__ == '__main__':
    main()
