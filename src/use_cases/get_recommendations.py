from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import NamedVector
import numpy as np
import logging
import pickle
import base64
import asyncio
import concurrent.futures
from src.services.llm_client import LLMClient
from ..types.recipe import PreprocessedQuery
from ..env_vars import COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT, ENCODER_ID
from ..utils.prompts import build_recipe_selection_prompt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')

class GetRecommendations:
    def __init__(self, llm_client: LLMClient):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3000)
        self.bm25_encoder = self._load_encoder_from_qdrant()
        self.llm_client = llm_client
        
    def _load_encoder_from_qdrant(self):
        try:
            logger.info(f"Attempting to load BM25Encoder from Qdrant (ID: {ENCODER_ID})...")
            
            search_result = self.client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[ENCODER_ID],
                with_payload=True,
                with_vectors=False
            )
            
            if search_result and len(search_result) > 0:
                point = search_result[0]
                if (hasattr(point, 'payload') and 
                    point.payload and 
                    'type' in point.payload and 
                    point.payload['type'] == 'bm25_encoder' and
                    'encoder_data' in point.payload):
                    
                    encoder_b64 = point.payload['encoder_data']
                    encoder_bytes = base64.b64decode(encoder_b64)
                    encoder = pickle.loads(encoder_bytes)

                    return encoder
            
            logger.warning("BM25Encoder not found in Qdrant or corrupted.")
        except Exception as e:
            logger.error(f"Error loading BM25Encoder from Qdrant: {e}")
            
        return encoder
    
    def get_recommendation(self, query: str, preprocessed_query: PreprocessedQuery, alpha=0.7) -> tuple:
        query_text = preprocessed_query["simplified_request"]
        keywords = preprocessed_query["keywords"]
        restrictions = preprocessed_query.get("restrictions", [])
        
        logger.info(f"Performing hybrid search for query: '{query_text}'")
        
        try:
            dense_vector = model.encode(query_text, convert_to_tensor=False).tolist()
            
            combined_sparse_vector = np.zeros(10000)
            if keywords:
                for keyword in keywords:
                    try:
                        sparse_vector_dict = self.bm25_encoder.encode_queries([keyword])[0]
                        sparse_vector = self._convert_sparse_dict_to_vector(sparse_vector_dict)
                        combined_sparse_vector = np.maximum(combined_sparse_vector, sparse_vector)
                    except Exception as e:
                        logger.error(f"Error processing keyword '{keyword}': {e}")
            
            non_zero_elements = np.count_nonzero(combined_sparse_vector)
            logger.info(f"Combined sparse vector has {non_zero_elements} non-zero elements")
            
            query_vectors = [
                (NamedVector(name="dense", vector=dense_vector), alpha),
                (NamedVector(name="sparse", vector=combined_sparse_vector.tolist()), 1.0 - alpha)
            ]
            
            search_batch_results = self.client.search_batch(
                collection_name=COLLECTION_NAME,
                requests=[
                    {
                        "vector": named_vector,
                        "with_payload": True,
                        "limit": 5
                    }
                    for named_vector, _ in query_vectors
                ]
            )
            
            results_map = {}
            
            if search_batch_results and len(search_batch_results) == 2:
                dense_results = search_batch_results[0]
                sparse_results = search_batch_results[1]
                
                for hit in dense_results:
                    results_map[hit.id] = {
                        "id": hit.id,
                        "dense_score": hit.score * alpha,
                        "sparse_score": 0.0,
                        "payload": hit.payload,
                        "matched_restrictions": [],
                        "restriction_penalty": 0.0
                    }
                
                for hit in sparse_results:
                    if hit.id in results_map:
                        results_map[hit.id]["sparse_score"] = hit.score * (1.0 - alpha)
                    else:
                        results_map[hit.id] = {
                            "id": hit.id,
                            "dense_score": 0.0,
                            "sparse_score": hit.score * (1.0 - alpha),
                            "payload": hit.payload,
                            "matched_restrictions": [],
                            "restriction_penalty": 0.0
                        }
                
                for result in results_map.values():
                    result["score"] = result["dense_score"] + result["sparse_score"]
          
            if restrictions and results_map:
                combined_restriction_vector = np.zeros(10000)
                restriction_names = []
                
                for restriction in restrictions:
                    try:
                        sparse_vector_dict = self.bm25_encoder.encode_queries([restriction])[0]
                        sparse_vector = self._convert_sparse_dict_to_vector(sparse_vector_dict)
                        
                        if np.count_nonzero(sparse_vector) > 0:
                            combined_restriction_vector = np.maximum(combined_restriction_vector, sparse_vector)
                            restriction_names.append(restriction)
                    except Exception as e:
                        logger.error(f"Error processing restriction '{restriction}': {e}")
                
                if restriction_names:
                    restriction_results = self.client.search(
                        collection_name=COLLECTION_NAME,
                        query_vector=NamedVector(name="sparse", vector=combined_restriction_vector.tolist()),
                        limit=100,
                        with_payload=True
                    )
                    
                    for hit in restriction_results:
                        if hit.id in results_map:
                            results_map[hit.id]["matched_restrictions"] = restriction_names
                            results_map[hit.id]["has_restrictions"] = True
            
            sorted_results = sorted(
                results_map.values(),
                key=lambda x: x["score"],
                reverse=True
            )[:5 * 2]
            
            if not sorted_results:
                return (0.0, {})
            
            prompt = build_recipe_selection_prompt(query, preprocessed_query, sorted_results)
            
            try:
                llm_response = self._sync_call_llm(prompt)
                logger.info(f"LLM response: {llm_response}")
                
                chosen_recipe_id = self._extract_recipe_id_from_llm_response(llm_response, sorted_results)
                
                if chosen_recipe_id and chosen_recipe_id in results_map:
                    chosen_recipe = results_map[chosen_recipe_id]
                    return (1.0, chosen_recipe["payload"])
                else:
                    return (sorted_results[0]["score"], sorted_results[0]["payload"])
                
            except Exception as llm_error:
                logger.error(f"Error when calling LLM: {llm_error}")
                return (sorted_results[0]["score"], sorted_results[0]["payload"])
            
        except Exception as e:
            logger.error(f"Error during hybrid search, falling back to dense search: {e}")
            
            try:
                dense_vector = model.encode(query_text, convert_to_tensor=False).tolist()
                dense_results = self.client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=NamedVector(name="dense", vector=dense_vector),
                    limit=5,
                    with_payload=True
                )
                if dense_results:
                    return (dense_results[0].score, dense_results[0].payload)
                return (0.0, {})
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                return (0.0, {})
    
    def _convert_sparse_dict_to_vector(self, sparse_dict, size=10000):
        vector = np.zeros(size)
        if isinstance(sparse_dict, dict) and 'indices' in sparse_dict and 'values' in sparse_dict:
            indices = sparse_dict['indices']
            values = sparse_dict['values']
            
            for idx, value in zip(indices, values):
                hashed_idx = idx % size
                vector[hashed_idx] = value
                    
        return vector
    
    def _extract_recipe_id_from_llm_response(self, llm_response, candidate_recipes):
        try:
            if "ID:" in llm_response:
                id_part = llm_response.split("ID:")[1].strip()
                possible_id = id_part.split()[0].strip()
                
                for recipe in candidate_recipes:
                    if str(recipe["id"]) == possible_id:
                        logger.info(f"Found recipe ID in expected format: {possible_id}")
                        return recipe["id"]
            
            for recipe in candidate_recipes:
                str_id = str(recipe["id"])
                if str_id in llm_response:
                    logger.info(f"Found recipe ID in response: {str_id}")
                    return recipe["id"]
            
            logger.warning(f"Could not extract recipe ID from LLM response: {llm_response}")
            return None
        except Exception as e:
            logger.error(f"Error extracting recipe ID from LLM response: {e}")
            return None
    
    def _sync_call_llm(self, prompt):
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                def run_async_in_thread():
                    temp_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(temp_loop)
                    try:
                        return temp_loop.run_until_complete(self.llm_client.get_response(prompt))
                    finally:
                        temp_loop.close()
                
                future = executor.submit(run_async_in_thread)
                return future.result(timeout=30)
        except concurrent.futures.TimeoutError:
            logger.error("LLM request timed out after 30 seconds")
            raise TimeoutError("LLM request timed out")
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise