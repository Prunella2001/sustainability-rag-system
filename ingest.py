#=====================================================================================
#                          KNOWLEDGE BASE ENGINE                                      
#             This module ingests the data and creates the database                   
#=====================================================================================

import os
import aiohttp
import asyncio
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_classic.retrievers import ParentDocumentRetriever
import pickle


# Setting up OpenAI API key

os.environ["OPENAI_API_KEY"] = ""
# Load content from web pages
links = [
    # =========================
    # DEFORESTATION & FORESTS
    # =========================

    "https://ourworldindata.org/deforestation",
    "https://fsc.org/en/blog/deforestation-facts",
    "https://earth.org/deforestation-facts/",
    "https://www.worldwildlife.org/initiatives/forests",
    "https://www.globalforestwatch.org/",
    "https://www.cbd.int/forest/",
    "https://ourworldindata.org/forest-area",
    "https://ourworldindata.org/afforestation",
    "https://ourworldindata.org/biodiversity",

    # =========================
    # CLIMATE CHANGE
    # =========================

    "https://climate.nasa.gov/",
    "https://www.ipcc.ch/reports/",
    "https://ourworldindata.org/co2-and-greenhouse-gas-emissions",
    "https://ourworldindata.org/climate-change",
    "https://www.un.org/en/climatechange",
    "https://www.unep.org/explore-topics/climate-action",
    "https://ghgprotocol.org/",
    "https://climate.ec.europa.eu/eu-action/european-green-deal_en",
    "https://www.noaa.gov/education/resource-collections/climate/climate-change-impacts",

    # =========================
    # WATER & WATER CONSERVATION
    # =========================

    "https://ourworldindata.org/water-use-stress",
    "https://www.epa.gov/watersense/start-saving",
    "https://www.epa.gov/watersense",
    "https://www.unwater.org/",
    "https://www.un.org/en/climatechange/science/climate-issues/water",
    "https://www.wri.org/topics/water",
    "https://education.nationalgeographic.org/resource/freshwater-crisis/",
    "https://www.worldwildlife.org/threats/water-scarcity",

    # =========================
    # PLASTIC POLLUTION
    # =========================

    "https://ourworldindata.org/plastic-pollution",
    "https://www.unep.org/plastic-pollution",
    "https://www.oecd.org/environment/plastics/",
    "https://ourworldindata.org/ocean-plastics",
    "https://education.nationalgeographic.org/resource/marine-pollution/",
    "https://www.iucn.org/resources/issues-brief/marine-plastic-pollution",
    "https://www.pewtrusts.org/en/projects/breaking-the-plastic-wave",
    "https://www.visualcapitalist.com/sp/where-the-worlds-ocean-plastic-waste-comes-from/",
    "https://www.undp.org/popping-the-bottle",
    "https://www.unep.org/news-and-stories/story/everything-you-should-know-about-microplastics"

    # =========================
    # RECYCLING & CIRCULAR ECONOMY
    # =========================

    "https://www.epa.gov/recycle/recycling-basics-and-benefits",
    "https://www.epa.gov/recycle/how-do-i-recycle-common-recyclables",
    "https://www.epa.gov/recycle",
    "https://environment.ec.europa.eu/topics/circular-economy_en",
    "https://ellenmacarthurfoundation.org/topics/circular-economy-introduction/overview",
    "https://ellenmacarthurfoundation.org/topics/plastics/overview",
    "https://www.unep.org/explore-topics/resource-efficiency/what-we-do/circular-economy",

    # =========================
    # ENERGY & RENEWABLE ENERGY
    # =========================

    "https://ourworldindata.org/energy",
    "https://ourworldindata.org/renewable-energy",
    "https://www.iea.org/topics/renewables",
    "https://www.energy.gov/clean-energy",
    "https://www.irena.org/",
    "https://www.un.org/en/climatechange/raising-ambition/renewable-energy",
    "https://ourworldindata.org/fossil-fuels",

    # =========================
    # TRANSPORTATION & EMISSIONS
    # =========================

    "https://ourworldindata.org/co2-emissions-from-transport",
    "https://www.iea.org/topics/transport",
    "https://www.epa.gov/greenvehicles",
    "https://www.unep.org/explore-topics/transport",

    # =========================
    # SUSTAINABLE FOOD & AGRICULTURE
    # =========================

    "https://ourworldindata.org/environmental-impacts-of-food",
    "https://ourworldindata.org/food-choice-vs-eating-local",
    "https://www.fao.org/sustainability/en/",
    "https://www.wri.org/food",
    "https://ourworldindata.org/food-ghg-emissions",
    "https://www.unep.org/explore-topics/food",
    "https://www.worldwildlife.org/industries/sustainable-agriculture",

    # =========================
    # BIODIVERSITY & ECOSYSTEMS
    # =========================

    "https://ourworldindata.org/biodiversity",
    "https://www.unep.org/explore-topics/biodiversity",
    "https://www.worldwildlife.org/threats/biodiversity-loss",
    "https://www.cbd.int/",
    "https://www.iucn.org/",

    # =========================
    # SUSTAINABLE LIVING
    # =========================

    "https://www.un.org/en/actnow",
    "https://www.epa.gov/greenliving",
    "https://www.unep.org/explore-topics/resource-efficiency",
    "https://www.nationalgeographic.com/environment/article/green-living",

    # =========================
    # ESG & CORPORATE SUSTAINABILITY
    # =========================

    "https://ghgprotocol.org/",
    "https://www.cdp.net/en",
    "https://www.weforum.org/agenda/archive/esg/",
    "https://www.globalreporting.org/",
    "https://www.sasb.org/",
    "https://www.unpri.org/",

    # =========================
    # DATA CENTERS & DIGITAL SUSTAINABILITY
    # =========================

    "https://electronics.howstuffworks.com/everyday-tech/why-do-data-centers-need-water.htm",
    "https://www.iea.org/energy-system/buildings/data-centres-and-data-transmission-networks",
    "https://sustainability.google/reports/",
    "https://www.microsoft.com/en-us/sustainability",
    "https://aws.amazon.com/sustainability/",

    # =========================
    # GLOBAL POLICY & SDGs
    # =========================

    "https://sdgs.un.org/goals",
    "https://unfccc.int/process-and-meetings/the-paris-agreement",
    "https://www.un.org/sustainabledevelopment/",
    "https://ec.europa.eu/commission/presscorner/detail/fr/MEMO_08_632",

    # =========================
    # EDUCATIONAL / BEGINNER-FRIENDLY
    # =========================

    "https://ourworldindata.org/",
    "https://education.nationalgeographic.org/",
    "https://climatekids.nasa.gov/",
    "https://www.bbc.com/future/tags/sustainability",

    # =========================
    # MYTHS / FACT CHECKING
    # =========================

    "https://climate.nasa.gov/evidence/",
    "https://skepticalscience.com/",
    "https://www.unep.org/news-and-stories/story/5-climate-change-myths-debunked",
    "https://www.iea.org/commentaries/clean-energy-progress-after-the-global-energy-crisis"
]


def ingest_data():
    print("📥 Initiating raw web scraping engine...")
    documents = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "text/html"
    }

    for idx, url in enumerate(links):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                    element.decompose()
                
                clean_text = soup.get_text(separator="\n")
                lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
                final_content = "\n".join(lines)
                
                if final_content:
                    doc = Document(page_content=final_content, metadata={"source": url})
                    documents.append(doc)
        except Exception as e:
            print(f"❌ Failed to reach {url}: {str(e)}")

    print(f"✅ Scraping complete. Total raw documents: {len(documents)}")
    return documents  # 🚀 Just returns the raw, un-split documents!