#=====================================================================================
#                          RETRIEVAL EXPERIMENTS                                      
#             This module tests different retrieval methods for the knowledge base                   
#=====================================================================================

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from ingest import ingest_data
import numpy as np
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
import pickle
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

#vector_store, chunks = ingest_data()

raw_documents = ingest_data()

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)

#Initialize Embeddings and an Empty Baseline Vector Index
#sentence-transformers/all-MiniLM-L6-v2
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
# Start empty
vector_store = FAISS.from_documents(
    documents=[Document(page_content="init_anchor", metadata={"source": "init"})], 
    embedding=embeddings
)
store = InMemoryStore()

# Build the ParentDocumentRetriever and index the documents
parent_child_retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
print("✂️ Splitting into parent-child fragments and embedding vectors...")
parent_child_retriever.add_documents(raw_documents, ids=None)

# Save cache to disk for runtime stability
vector_store.save_local("./faiss_index")
with open("./faiss_index/parent_store.pkl", "wb") as f:
    pickle.dump(store, f)

# 5. Extract parent chunks for the legacy BM25 retriever leg to look at
all_parent_keys = list(store.yield_keys())
chunks = []
for k in all_parent_keys:
    doc = store.mget([k])[0]
    if doc:
        # doc is already a LangChain Document object!
        # It already has its original web URL inside doc.metadata["source"]
        chunks.append(doc)

# Initialize BM25 leg using the rich parent text blocks
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 12  

# Set the parent-child retriever search parameter to look for child vectors
parent_child_retriever.search_kwargs = {"k": 15}

eval_queries = [
    "What causes deforestation?",
    "Why are tropical forests important?",
    "How does agriculture contribute to forest loss?",
    "What are microplastics?",
    "Why is plastic pollution dangerous for oceans?",
    "Can all plastics be recycled?",
    "What are the benefits of recycling?",
    "How can people recycle common household materials?",
    "Why do data centers need water?",
    "How can people reduce their environmental impact?",
    "What is a circular economy?",
    "Why is biodiversity important?",
    "How does climate change affect water resources?",
    "What foods have the highest environmental impact?",
    "How can renewable energy help climate change?",
    "What are greenhouse gases?",
    "What is sustainable agriculture?",
    "How can consumers help reduce deforestation?",
    "Why is water scarcity increasing?",
    "What are sustainable living habits?"
    ]

ground_truth_docs = [
    [
    "https://ourworldindata.org/deforestation",
    "https://earth.org/deforestation-facts/",
    "https://fsc.org/en/blog/deforestation-facts"
    ],

    [
        "https://ourworldindata.org/deforestation",
        "https://www.worldwildlife.org/initiatives/forests",
        "https://www.cbd.int/forest/"
    ],

    [
        "https://ourworldindata.org/deforestation",
        "https://ourworldindata.org/environmental-impacts-of-food",
        "https://www.worldwildlife.org/industries/sustainable-agriculture"
    ],

    [
        "https://ourworldindata.org/plastic-pollution",
        "https://www.unep.org/plastic-pollution",
        "https://www.iucn.org/resources/issues-brief/marine-plastic-pollution"
    ],

    [
        "https://ourworldindata.org/plastic-pollution",
        "https://education.nationalgeographic.org/resource/marine-pollution/",
        "https://www.unep.org/plastic-pollution"
    ],

    [
        "https://www.epa.gov/recycle/how-do-i-recycle-common-recyclables",
        "https://www.epa.gov/recycle/recycling-basics-and-benefits",
        "https://ellenmacarthurfoundation.org/topics/plastics/overview"
    ],

    [
        "https://www.epa.gov/recycle/recycling-basics-and-benefits",
        "https://environment.ec.europa.eu/topics/circular-economy_en",
        "https://www.unep.org/explore-topics/resource-efficiency/what-we-do/circular-economy"
    ],

    [
        "https://www.epa.gov/recycle/how-do-i-recycle-common-recyclables",
        "https://www.epa.gov/recycle"
    ],

    [
        "https://electronics.howstuffworks.com/everyday-tech/why-do-data-centers-need-water.htm",
        "https://www.iea.org/energy-system/buildings/data-centres-and-data-transmission-networks"
    ],

    [
        "https://www.un.org/en/actnow",
        "https://www.epa.gov/greenliving",
        "https://www.unep.org/explore-topics/resource-efficiency"
    ],

    [
        "https://environment.ec.europa.eu/topics/circular-economy_en",
        "https://ellenmacarthurfoundation.org/topics/circular-economy-introduction/overview",
        "https://www.unep.org/explore-topics/resource-efficiency/what-we-do/circular-economy"
    ],

    [
        "https://ourworldindata.org/biodiversity",
        "https://www.worldwildlife.org/threats/biodiversity-loss",
        "https://www.cbd.int/"
    ],

    [
        "https://www.un.org/en/climatechange/science/climate-issues/water",
        "https://ourworldindata.org/water-use-stress",
        "https://www.unwater.org/"
    ],

    [
        "https://ourworldindata.org/environmental-impacts-of-food",
        "https://ourworldindata.org/food-ghg-emissions",
        "https://www.fao.org/sustainability/en/"
    ],

    [
        "https://ourworldindata.org/renewable-energy",
        "https://www.iea.org/topics/renewables",
        "https://www.energy.gov/clean-energy"
    ],

    [
        "https://ourworldindata.org/co2-and-greenhouse-gas-emissions",
        "https://climate.nasa.gov/",
        "https://www.un.org/en/climatechange"
    ],

    [
        "https://www.worldwildlife.org/industries/sustainable-agriculture",
        "https://www.fao.org/sustainability/en/",
        "https://www.wri.org/food"
    ],

    [
        "https://fsc.org/en/blog/deforestation-facts",
        "https://ourworldindata.org/deforestation",
        "https://www.worldwildlife.org/initiatives/forests"
    ],

    [
        "https://ourworldindata.org/water-use-stress",
        "https://www.worldwildlife.org/threats/water-scarcity",
        "https://www.unwater.org/"
    ],

    [
        "https://www.un.org/en/actnow",
        "https://www.epa.gov/greenliving",
        "https://www.unep.org/explore-topics/resource-efficiency"
    ]

]

def eval_weight(weight, eval_queries, ground_truth_docs):
    results = []
    ensemble_retriever = EnsembleRetriever(
      retrievers=[bm25_retriever, parent_child_retriever],
      weights=weight  # Weight semantic search slightly higher
    )
    for query, expected_src_ids in zip(eval_queries, ground_truth_docs):
      retrieved_docs = ensemble_retriever.invoke(query)
      retrieved_ids = [ doc.metadata.get("source") for doc in retrieved_docs]

      hit = any(src in retrieved_ids for src in expected_src_ids)
      rank = next((i + 1 for i, src in enumerate(retrieved_ids) if src in expected_src_ids),None)
      results.append({"hit": hit, "rank": rank})

    hit_rate = np.mean([r["hit"] for r in results])
    mrr = np.mean([1 / r["rank"] if r["rank"] else 0 for r in results])
    return {"weights": weight, "hit_rate": hit_rate, "mrr": mrr}


#for r in results:
#    print(f"Weights: {r['weights']} | Hit Rate: {r['hit_rate']:.3f} | MRR: {r['mrr']:.3f}")

def best_weights(weights):
    results = [eval_weight(w, eval_queries, ground_truth_docs) for w in weights]
    eligible = [r for r in results if r["hit_rate"] > 0.6]
    if eligible:
        #best = max(
        #    eligible, 
        #    key=lambda r: (2 * r["hit_rate"] * r["mrr"]) / (r["hit_rate"] + r["mrr"]) if (r["hit_rate"] + r["mrr"]) > 0 else 0
        #)
        #best = max(eligible, key=lambda r: (round(r["hit_rate"], 2), r["mrr"]))
        #best = max(eligible, key=lambda r: (r["hit_rate"], r["mrr"]))
        #best = max(eligible, key=lambda r: r["hit_rate"])
        best = max(eligible, key=lambda r: (r["hit_rate"], r["mrr"]))
        return best["weights"], best["hit_rate"], best["mrr"]
    return None, 0, 0


def found_retriever(weights):

    #these can be modified
    #weights = [
    #    [0.0, 1.0],
    #    [0.25, 0.75],
    #    [0.5, 0.5],
    #    [0.75, 0.25],
    #    [1.0, 0.0],
    #    [0.3, 0.7],
    #    [0.4, 0.6]
    #]
    b_weights, hit_rate, mrr = best_weights(weights)
    if b_weights:
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, parent_child_retriever],
            weights=b_weights
        )
        cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=retriever,
        )
        return compression_retriever
        #return retriever
    return None
