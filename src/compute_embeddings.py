import pandas as pd
import ast
from sentence_transformers import SentenceTransformer
import logging
import gc
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from pinecone_text.sparse import BM25Encoder
import numpy as np
import asyncio
from typing import List
from src.env_vars import COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT, ENVIRONMENT, ENCODER_ID
import time
import psutil
import pickle
import base64

DATA_FILE = "full_dataset.csv"
CHUNK_SIZE = 5000
BATCH_SIZE = 250
MAX_CONCURRENT_REQUESTS = 10
DEV_SAMPLE_FRACTION = 0.25

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    logger.info(f"Memory usage: {memory_mb:.2f} MB")

def parse_field(field):
    try:
        if isinstance(field, str) and field.strip().startswith("[") and field.strip().endswith("]"):
            return ast.literal_eval(field)
    except Exception as e:
        logger.error(f"Error converting field: {field}\nError: {e}")
    return field

def aggregate_fields(row):
    parts = [f"{str(row.get('title', ''))}"]
    
    ingredients = row.get('ingredients', [])
    if ingredients:
        parts.append("ingredients:")
        parts.extend([f"- {ingredient}" for ingredient in ingredients])
    
    directions = row.get('directions', [])
    if directions:
        parts.append("directions:")
        parts.extend([f"- {direction}" for direction in directions])
    
    return "\n".join(parts)

def chunk_upserts(points: List[PointStruct], chunk_size: int):
    return [points[i:i + chunk_size] for i in range(0, len(points), chunk_size)]

def process_chunk(df_chunk):
    chunk = df_chunk.copy()
    chunk.loc[:, 'ingredients'] = chunk['ingredients'].apply(parse_field)
    chunk.loc[:, 'directions'] = chunk['directions'].apply(parse_field)
    chunk.loc[:, 'NER'] = chunk['NER'].apply(parse_field)
    return chunk

def convert_sparse_vector(sparse_dict, size=10000):
    vector = np.zeros(size)
    if isinstance(sparse_dict, dict) and 'indices' in sparse_dict and 'values' in sparse_dict:
        indices = sparse_dict['indices']
        values = sparse_dict['values']
        
        for idx, value in zip(indices, values):
            hashed_idx = idx % size
            vector[hashed_idx] = value
            
    else:
        logger.warning(f"Invalid sparse dict format: {sparse_dict}")
    return vector

async def upsert_batch(client: QdrantClient, collection_name: str, batch: List[PointStruct]) -> bool:
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, 
            lambda: client.upsert(collection_name=collection_name, points=batch)
        )
        return True
    except Exception as e:
        logger.error(f"Failed to upsert batch: {str(e)}")
        return False

async def parallel_upload_points(client: QdrantClient, points: List[PointStruct], collection_name: str) -> List[int]:
    if not points:
        return []
        
    batches = chunk_upserts(points, BATCH_SIZE)
    logger.info(f"Uploading {len(points)} points in {len(batches)} batches with {MAX_CONCURRENT_REQUESTS} concurrent requests")
    
    failed_ids = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def upload_batch_with_semaphore(batch):
        async with semaphore:
            batch_ids = [p.id for p in batch]
            success = await upsert_batch(client, collection_name, batch)
            
            if not success:
                return batch_ids
            return []
    
    for i in range(0, len(batches), MAX_CONCURRENT_REQUESTS):
        current_batches = batches[i:i + MAX_CONCURRENT_REQUESTS]
        tasks = [upload_batch_with_semaphore(batch) for batch in current_batches]
        results = await asyncio.gather(*tasks)
        
        for result in results:
            failed_ids.extend(result)
        
        gc.collect()
    
    return failed_ids

async def process_and_upload_chunk(
    model: SentenceTransformer,
    bm25_encoder: BM25Encoder,
    client: QdrantClient,
    chunk: pd.DataFrame,
    start_idx: int
) -> int:
    try:
        chunk = process_chunk(chunk)
        meta_batch = chunk[['title', 'ingredients', 'directions', 'NER']].to_dict(orient="records")
        
        texts = [aggregate_fields(row) for _, row in chunk.iterrows()]
        
        dense_embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
        
        sparse_embeddings = bm25_encoder.encode_documents(texts)
        
        points = []
        for idx, (dense, sparse, meta) in enumerate(zip(dense_embeddings, sparse_embeddings, meta_batch)):
            point_id = start_idx + idx
            sparse_vector = convert_sparse_vector(sparse)
            
            points.append(PointStruct(
                id=point_id,
                vector={
                    "dense": dense.tolist(),
                    "sparse": sparse_vector.tolist()
                },
                payload=meta
            ))
            del sparse_vector
        
        del dense_embeddings, sparse_embeddings
        gc.collect()
        
        failed_ids = await parallel_upload_points(client, points, COLLECTION_NAME)
        
        del points
        gc.collect()
        
        return len(chunk) - len(failed_ids)
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}", exc_info=True)
        return 0

def save_bm25_encoder_to_qdrant(client: QdrantClient, encoder: BM25Encoder):
    try:
        encoder_bytes = pickle.dumps(encoder)
        
        encoder_b64 = base64.b64encode(encoder_bytes).decode('utf-8')
        
        dummy_dense = np.zeros(384).tolist()
        dummy_sparse = np.zeros(10000).tolist()
        
        point = PointStruct(
            id=ENCODER_ID,
            vector={
                "dense": dummy_dense,
                "sparse": dummy_sparse
            },
            payload={
                "type": "bm25_encoder",
                "encoder_data": encoder_b64,
                "timestamp": time.time()
            }
        )
        
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
        
        logger.info(f"Saved BM25Encoder to Qdrant (ID: {ENCODER_ID})")
        return True
    except Exception as e:
        logger.error(f"Error saving BM25Encoder to Qdrant: {e}")
        return False

async def main():
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        try:
            client.delete_collection(COLLECTION_NAME)
        except:
            pass
        
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=384, distance=Distance.COSINE),
                "sparse": VectorParams(size=10000, distance=Distance.COSINE)
            }
        )

        model = SentenceTransformer('all-MiniLM-L6-v2')
        bm25_encoder = BM25Encoder()

        logger.info(f"Loading data from {DATA_FILE}")
        
        total_rows = 0
        logger.info("Counting total records in the dataset...")
        for chunk_df in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):
            total_rows += len(chunk_df)
        
        logger.info(f"Total records in dataset: {total_rows}")
            
        if ENVIRONMENT == "dev":
            target_rows = int(total_rows * DEV_SAMPLE_FRACTION)
            logger.info(f"Development mode: using ~{target_rows} samples ({DEV_SAMPLE_FRACTION*100}% of dataset)")
        else:
            target_rows = total_rows
            logger.info(f"Production mode: using all {total_rows} samples (100% of dataset)")
        
        logger.info("Collecting texts for BM25Encoder training...")
        all_training_texts = []
        processed_rows = 0
        
        for chunk_df in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):
            if ENVIRONMENT == "dev":
                chunk_df = chunk_df.sample(frac=DEV_SAMPLE_FRACTION, random_state=42)

            processed_chunk = process_chunk(chunk_df)
            batch_texts = [aggregate_fields(row) for _, row in processed_chunk.iterrows()]
            all_training_texts.extend(batch_texts)
            
            processed_rows += len(batch_texts)

            if ENVIRONMENT == "dev" and processed_rows >= target_rows:
                break

            if processed_rows >= target_rows:
                break
        
        if all_training_texts:
            logger.info(f"Training BM25Encoder with {len(all_training_texts)} texts...")
            start_time = time.time()
            bm25_encoder.fit(all_training_texts)
            training_time = time.time() - start_time
            logger.info(f"BM25Encoder training completed in {training_time:.2f} seconds")
            
            test_encode = bm25_encoder.encode_queries(["test query"])
            logger.info(f"BM25Encoder test encoding: {test_encode}")
            
            save_bm25_encoder_to_qdrant(client, bm25_encoder)
            
            del all_training_texts
            gc.collect()
            log_memory_usage()
        else:
            logger.warning("No texts collected for BM25Encoder training")

        total_processed = 0
        total_start_time = time.time()
        
        chunk_id = 0
        for chunk_df in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):
            chunk_id += 1
            
            if ENVIRONMENT == "dev" and chunk_id % 10 != 0:
                continue
                
            if ENVIRONMENT == "dev":
                chunk_df = chunk_df.sample(frac=DEV_SAMPLE_FRACTION, random_state=42)
                
            logger.info(f"Processing chunk {chunk_id} with {len(chunk_df)} rows")
            
            processed = await process_and_upload_chunk(
                model,
                bm25_encoder,
                client,
                chunk_df,
                total_processed
            )
            
            total_processed += processed
            
            if total_processed >= target_rows:
                logger.info(f"Reached target of {target_rows} rows. Stopping processing.")
                break
                
            gc.collect()

        total_time = time.time() - total_start_time
        logger.info(f"Processing completed: {total_processed} recipes in {total_time:.2f}s")
        logger.info(f"Average processing rate: {total_processed/total_time:.2f} points/second")
        
        logger.info("Collection info:")
        logger.info(client.get_collection(COLLECTION_NAME))

        logger.info("BM25Encoder atualizado com sucesso sem alterar o índice de receitas.")

    except Exception as e:
        logger.error(f"Critical error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())