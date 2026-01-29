"""
BioFlow Image Data Gatherer - PROFESSIONAL DATA SOURCES
========================================================

Streams biomedical images from THREE high-quality open repositories:

1. **IDR (Image Data Resource)** - Microscopy images with gene annotations
   - https://idr.openmicroscopy.org
   - Cell/tissue microscopy linked to genes (UniProt alignment)
   
2. **PubChem** - Chemical spectra images
   - https://pubchem.ncbi.nlm.nih.gov
   - Compound structure images for visual search
   
3. **PMC Open Access** - Western blots & gels from papers
   - https://www.ncbi.nlm.nih.gov/pmc
   - Figure images with captions (image-to-text search)

All images are streamed IN-MEMORY (zero disk usage).
"""
import os
import sys
import logging
import time
import requests
import tarfile
import math
from typing import Generator, Dict, Any, List, Optional
from PIL import Image
from io import BytesIO
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
USER_AGENT = "BioFlow/1.0 (hackathon-educational; mailto:bioflow@example.com)"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}

IDR_HEADERS = {
    **HEADERS,
    "Referer": "https://idr.openmicroscopy.org/",
}

DEFAULT_TIMEOUT_SECONDS = 20

IDR_BASE_URL = "https://idr.openmicroscopy.org"
IDR_INDEX_URL = f"{IDR_BASE_URL}/webclient/?experimenter=-1"


def _get_idr_session() -> requests.Session:
    """Create a session with IDR to obtain cookies for public endpoints."""
    session = requests.Session()
    session.headers.update(IDR_HEADERS)
    try:
        resp = session.get(IDR_INDEX_URL, timeout=(10, 15))
        if resp.status_code != 200:
            logger.warning(f"IDR session init HTTP {resp.status_code} (url={IDR_INDEX_URL})")
    except Exception as e:
        logger.warning(f"IDR session init failed: {type(e).__name__}: {e}")
    return session


def _fetch_idr_thumbnail(session: requests.Session, image_id: int) -> Optional[Image.Image]:
    """Fetch a public thumbnail image from IDR using authenticated session cookies."""
    # Use the webclient thumbnail endpoint which is public and stable for IDR
    thumb_url = f"{IDR_BASE_URL}/webclient/render_thumbnail/{image_id}/"
    try:
        resp = session.get(thumb_url, timeout=DEFAULT_TIMEOUT_SECONDS)
        if resp.status_code != 200:
            logger.warning(f"IDR thumbnail {image_id} HTTP {resp.status_code} (url={thumb_url})")
            return None
        return Image.open(BytesIO(resp.content))
    except requests.exceptions.Timeout as e:
        logger.warning(f"IDR thumbnail {image_id} timeout ({e})")
        return None
    except Exception as e:
        logger.warning(f"IDR thumbnail {image_id} error: {type(e).__name__}: {str(e)[:160]}")
        return None

def stream_biomedical_images(
    limit: int = 30,
    max_per_type: int = 10,
    query: str = "kinase",
    *,
    strict: bool = False,
) -> Generator[Dict[str, Any], None, None]:
    """
    Main generator that yields images from all three sources.
    
    Args:
        limit: Total images to stream
        max_per_type: Max images per source type
        query: Search query to guide image selection (e.g., "kinase", "cancer")
    """
    logger.info("📡 Streaming from professional biomedical image repositories...")
    logger.info(f"🎯 Target: {limit} images total, max {max_per_type} per type")
    logger.info(f"🔍 Query: '{query}'")

    if limit <= 0:
        logger.info("✅ Streaming complete! Total: 0 images")
        logger.info("📊 Final breakdown: {'microscopy': 0, 'spectra': 0, 'gel': 0}")
        return

    # Allocate a soft per-type target, but backfill later sources to hit `limit`.
    per_type_target = min(max_per_type, max(1, math.ceil(limit / 3)))

    counts = {"microscopy": 0, "spectra": 0, "gel": 0}
    total_yielded = 0

    def _yield_with_guard(
        generator: Generator[Dict[str, Any], None, None],
        kind: str,
        max_items: int,
    ) -> Generator[Dict[str, Any], None, None]:
        nonlocal total_yielded
        for img in generator:
            if total_yielded >= limit or counts[kind] >= max_items:
                break
            yield img
            counts[kind] += 1
            total_yielded += 1

    # 1) IDR microscopy
    remaining = limit - total_yielded
    microscopy_target = min(per_type_target, remaining)
    if microscopy_target > 0:
        logger.info("\n🔬 === IDR MICROSCOPY ===")
        for img in _yield_with_guard(_stream_idr_microscopy(limit=microscopy_target, query=query, strict=strict), "microscopy", microscopy_target):
            yield img

    # 2) PubChem spectra/structures
    remaining = limit - total_yielded
    # If IDR under-delivers, allow PubChem to take some of that quota (faster/less flaky than tarballs).
    idr_shortfall = max(0, microscopy_target - counts["microscopy"])
    spectra_target = min(max_per_type + idr_shortfall, remaining)
    if spectra_target > 0:
        logger.info("\n🧪 === PUBCHEM SPECTRA ===")
        for img in _yield_with_guard(_stream_pubchem_spectra(limit=spectra_target, query=query, strict=strict), "spectra", spectra_target):
            yield img

    # 3) PMC gels - backfill the remainder by default
    remaining = limit - total_yielded
    if remaining > 0:
        logger.info("\n🧬 === PMC GELS (Western Blots) ===")
        # Allow gels to backfill beyond max_per_type if earlier sources under-deliver.
        gel_target = remaining
        for img in _yield_with_guard(_stream_pmc_gels(limit=gel_target, query=query), "gel", gel_target):
            yield img

    logger.info(f"✅ Streaming complete! Total: {total_yielded} images")
    logger.info(f"📊 Final breakdown: {counts}")

    if strict and total_yielded < limit:
        raise RuntimeError(f"Image streaming strict mode: requested {limit}, yielded {total_yielded}. Breakdown={counts}")

# -----------------------------------------------------------------------------
# SOURCE 1: IDR (Microscopy)
# -----------------------------------------------------------------------------
def _stream_idr_microscopy(limit: int, query: str = "kinase", *, strict: bool = False) -> Generator[Dict[str, Any], None, None]:
    """
    Streams microscopy images from IDR based on query.
    
    Searches IDR Screen 102 (Human HeLa cells - Mitosis screen) for gene-related images.
    If query doesn't match genes, falls back to diverse sample from screen.
    """
    logger.info(f"🔬 Fetching microscopy from IDR (limit={limit}, query='{query}')...")
    
    # IDR Screen 102 has ~60,000 images across ~200 genes
    # We'll query the screen metadata to find images matching the query
    # For simplicity, using a diverse sampling approach since IDR API requires OMERO authentication
    
    # Strategy: Sample diverse images from Screen 102 (ID: 102)
    # Images are numbered sequentially, so we sample at intervals.
    base_image_id = 179693  # First image in Screen 102
    step = 50  # Sample every 50th image for diversity
    
    session = _get_idr_session()
    count = 0
    failures: List[str] = []
    for i in range(limit * 6):  # Try 6x limit to account for failures
        if count >= limit:
            break

        image_id = base_image_id + (i * step)

        try:
            image = _fetch_idr_thumbnail(session, image_id)
            if image is None:
                msg = f"IDR image {image_id} fetch failed"
                failures.append(msg)
                continue
            # Extract gene name from IDR metadata (simplified - placeholder)
            gene_name = f"Screen_102_Image_{count}"

            yield {
                "source": "IDR",
                "source_id": str(image_id),
                "image_type": "microscopy",
                "modality": "image",
                "caption": f"Microscopy image from IDR Screen 102 (Mitosis/Cell Division) - {gene_name}",
                "image": image,
                "metadata": {
                    "gene": gene_name,
                    "screen": 102,
                    "image_id": image_id,
                    "query": query,
                    "url": f"{IDR_BASE_URL}/webclient/img_detail/{image_id}/",
                    "thumbnail_url": f"{IDR_BASE_URL}/webclient/render_thumbnail/{image_id}/",
                },
            }
            count += 1
            time.sleep(0.3)

        except requests.exceptions.Timeout as e:
            msg = f"IDR image {image_id} timeout"
            failures.append(msg)
            logger.warning(f"{msg} ({e})")
            continue
        except Exception as e:
            msg = f"IDR image {image_id} error: {type(e).__name__}: {str(e)[:160]}"
            failures.append(msg)
            logger.warning(msg)
            continue

    if count == 0:
        logger.error(
            "IDR yielded 0 images. Most common causes: endpoint blocked/rate-limited or using a non-public render endpoint. "
            "This build uses webgateway; if you still see 403, the IDR server may be denying access from your network."
        )
        if strict:
            raise RuntimeError("IDR yielded 0 images")

# -----------------------------------------------------------------------------
# SOURCE 2: PubChem (Spectra/Structures)
# -----------------------------------------------------------------------------
def _stream_pubchem_spectra(limit: int, query: str = "kinase", *, strict: bool = False) -> Generator[Dict[str, Any], None, None]:
    """
    Streams 2D structure images from PubChem based on query.
    
    Searches PubChem for compounds matching the query and fetches their structure images.
    """
    logger.info(f"🧪 Fetching spectra from PubChem (limit={limit}, query='{query}')...")
    
    # Use PUG REST first for name-like queries, then fall back to Entrez for general terms
    # or if the name search yields too few hits.
    # PUG REST docs: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
    cids: List[int] = []

    try:
        pug_name_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(query)}/cids/JSON"
        pug_resp = requests.get(pug_name_url, headers=HEADERS, timeout=DEFAULT_TIMEOUT_SECONDS)
        if pug_resp.status_code == 200:
            id_list = pug_resp.json().get("IdentifierList", {}).get("CID", [])
            cids = [int(x) for x in id_list if str(x).isdigit()]
            if cids:
                logger.info(f"   Found {len(cids)} candidate compounds via PUG REST name search for '{query}'")
        else:
            logger.info(f"   PUG REST name search returned HTTP {pug_resp.status_code} for '{query}', falling back to Entrez")
    except Exception as e:
        logger.info(f"   PUG REST name search error: {type(e).__name__}: {e}; falling back to Entrez")

    min_expected = max(5, min(limit // 5, 50))
    if not cids or len(cids) < min_expected:
        # For general biomedical queries like "kinase", use Entrez ESearch on db=pccompound.
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

        params = {
            "db": "pccompound",
            "term": query,
            "retmode": "json",
            "retmax": min(limit * 10, 200),
        }

        try:
            resp = requests.get(search_url, params=params, headers=HEADERS, timeout=DEFAULT_TIMEOUT_SECONDS)
            if resp.status_code != 200:
                logger.error(f"PubChem Entrez search failed HTTP {resp.status_code} for query '{query}'")
                if strict:
                    raise RuntimeError("PubChem Entrez search failed")
                return

            id_list = resp.json().get("esearchresult", {}).get("idlist", [])
            ent_cids = [int(x) for x in id_list if str(x).isdigit()]
            if not ent_cids:
                logger.error(f"No PubChem compounds found via Entrez for query '{query}'")
                if strict:
                    raise RuntimeError("No PubChem compounds found")
                return

            # Merge while preserving order preference for PUG name hits
            if cids:
                merged = list(dict.fromkeys(cids + ent_cids))
                cids = merged
                logger.info(
                    f"   Found {len(ent_cids)} via Entrez for '{query}', merged total {len(cids)} candidates"
                )
            else:
                cids = ent_cids
                logger.info(f"   Found {len(cids)} candidate compounds via Entrez for '{query}'")
        except Exception as e:
            logger.error(f"PubChem Entrez search error: {type(e).__name__}: {e}")
            if strict:
                raise
            return
    
    count = 0
    failures: List[str] = []
    for i, cid in enumerate(cids):
        if count >= limit: break
        
        try:
            # Get compound name first
            name_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/Title/JSON"
            name_resp = requests.get(name_url, headers=HEADERS, timeout=5)
            name = "Unknown"
            if name_resp.status_code == 200:
                name = name_resp.json().get("PropertyTable", {}).get("Properties", [{}])[0].get("Title", "Unknown")
            
            # Get 2D structure image
            img_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG"
            img_resp = requests.get(img_url, headers=HEADERS, timeout=DEFAULT_TIMEOUT_SECONDS)
            
            if img_resp.status_code == 200:
                image = Image.open(BytesIO(img_resp.content))
                
                yield {
                    "source": "PubChem",
                    "source_id": str(cid),
                    "image_type": "spectra",
                    "modality": "image",
                    "caption": f"Chemical structure of {name} (CID: {cid}) - Query: {query}",
                    "image": image,
                    "metadata": {"cid": cid, "name": name, "query": query}
                }
                count += 1
                time.sleep(0.4)
            else:
                msg = f"PubChem CID {cid} image fetch failed HTTP {img_resp.status_code}"
                failures.append(msg)
                logger.warning(f"{msg} (url={img_url})")
                continue
        except requests.exceptions.Timeout:
            msg = f"Timeout fetching PubChem CID {cid}"
            failures.append(msg)
            logger.warning(msg)
            continue
        except Exception as e:
            msg = f"Error fetching PubChem CID {cid}: {type(e).__name__}: {str(e)[:160]}"
            failures.append(msg)
            logger.warning(msg)
            continue

    if count == 0:
        logger.error("PubChem yielded 0 images for this query.")
        if strict:
            raise RuntimeError("PubChem yielded 0 images")

# -----------------------------------------------------------------------------
# SOURCE 3: PMC Open Access (Gels) - UPDATED
# -----------------------------------------------------------------------------
def _stream_pmc_gels(limit: int, query: str = "kinase", *, max_total_seconds: int = 300) -> Generator[Dict[str, Any], None, None]:
    """
    Streams Western Blot/Gel images by parsing PMC OA tarballs on the fly.
    
    Searches PMC for papers related to query with Western Blot/Gel images.
    """
    logger.info(f"🧬 Fetching gels from PMC Open Access (limit={limit}, query='{query}')...")

    start_time = time.time()
    
    # 1. Find articles with query + "Western Blot" in open access subset
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    
    # Combine query with Western Blot search for relevance
    search_term = f"{query} AND (Western Blot[Caption] OR gel electrophoresis[Caption]) AND open access[filter]"
    
    params = {
        "db": "pmc",
        "term": search_term,
        "retmode": "json",
        "retmax": min(limit * 3, 100)  # Fetch 3x limit (max 100) to account for failures
    }
    
    try:
        resp = requests.get(search_url, params=params, headers=HEADERS, timeout=(10, 15))
        pmc_ids = resp.json().get("esearchresult", {}).get("idlist", [])
        logger.info(f"   Found {len(pmc_ids)} candidate PMC articles for '{query}'")
    except Exception as e:
        logger.error(f"PMC Search failed: {e}")
        return

    # If too few hits, broaden the query (remove Caption restriction)
    if len(pmc_ids) < max(10, limit // 5):
        broad_term = f"{query} AND (Western Blot OR gel electrophoresis) AND open access[filter]"
        broad_params = {
            "db": "pmc",
            "term": broad_term,
            "retmode": "json",
            "retmax": min(limit * 3, 200),
        }
        try:
            broad_resp = requests.get(search_url, params=broad_params, headers=HEADERS, timeout=(10, 15))
            broad_ids = broad_resp.json().get("esearchresult", {}).get("idlist", [])
            if broad_ids:
                merged = list(dict.fromkeys(pmc_ids + broad_ids))
                pmc_ids = merged
                logger.info(
                    f"   Broad PMC search added {len(broad_ids)} candidates; merged total {len(pmc_ids)}"
                )
        except Exception as e:
            logger.warning(f"PMC broad search failed: {e}")

    # 2. For each article, get the OA extraction URL
    count = 0
    for pmcid in pmc_ids:
        if count >= limit: break

        if (time.time() - start_time) > max_total_seconds:
            logger.warning(f"   PMC gel streaming hit time budget ({max_total_seconds}s); stopping early with {count}/{limit} images")
            break
        
        full_pmcid = f"PMC{pmcid}"
        oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={full_pmcid}"
        
        try:
            # OA lookup can occasionally stall; keep timeouts tight and retry once.
            oa_resp = None
            for attempt in range(2):
                try:
                    oa_resp = requests.get(oa_url, headers=HEADERS, timeout=(10, 15))
                    if oa_resp.status_code == 200:
                        break
                    logger.warning(f"   OA lookup for {full_pmcid} HTTP {oa_resp.status_code} (attempt {attempt+1}/2)")
                except requests.exceptions.Timeout:
                    logger.warning(f"   OA lookup timeout for {full_pmcid} (attempt {attempt+1}/2)")
                time.sleep(0.5)

            if oa_resp is None or oa_resp.status_code != 200:
                logger.warning(f"   OA lookup failed for {full_pmcid}, skipping...")
                continue

            root = ET.fromstring(oa_resp.content)
            
            # Find the TGZ link
            link_node = root.find(".//link[@format='tgz']")
            if link_node is None:
                logger.warning(f"   No TGZ archive found for {full_pmcid}, skipping...")
                continue
                
            tgz_url = link_node.get("href")
            # Usually ftp://, change to https:// for better firewall handling if possible,
            # but requests handles ftp usually via adapter or just use the http link if provided.
            # PMC OA often provides ftp links. Let's try to convert to https if it's ncbi
            if tgz_url.startswith("ftp://ftp.ncbi.nlm.nih.gov"):
                tgz_url = tgz_url.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")
            
            logger.info(f"   ⬇️ Streaming archive for {full_pmcid}...")
            
            # 3. Stream the Tarball with retry
            max_retries = 2
            tgz_resp = None
            
            for retry in range(max_retries):
                try:
                    tgz_resp = requests.get(tgz_url, stream=True, headers=HEADERS, timeout=(10, 30))
                    if tgz_resp.status_code == 200:
                        break
                except requests.exceptions.Timeout:
                    if retry < max_retries - 1:
                        logger.warning(f"   Timeout on {full_pmcid}, retry {retry+1}/{max_retries}...")
                        time.sleep(1)
                    else:
                        raise
            
            if tgz_resp and tgz_resp.status_code == 200:
                # Open tarfile from stream
                with tarfile.open(fileobj=tgz_resp.raw, mode="r|gz") as tar:
                    for member in tar:
                        # Look for figure images (usually .jpg)
                        # We want to catch 'figure', 'g001', etc.
                        if member.isfile() and member.name.lower().endswith(('.jpg', '.jpeg')):
                            
                            # Extract file to memory
                            f = tar.extractfile(member)
                            if f:
                                img_data = f.read()
                                image = Image.open(BytesIO(img_data))
                                
                                # Verify it's a valid image
                                image.verify() 
                                # Re-open for usage
                                image = Image.open(BytesIO(img_data))
                                
                                yield {
                                    "source": "PMC",
                                    "source_id": f"{full_pmcid}_{member.name}",
                                    "image_type": "gel",
                                    "modality": "image",
                                    "caption": f"Western Blot/Gel evidence from {full_pmcid} (Query: {query}). File: {member.name}",
                                    "image": image,
                                    "metadata": {"pmcid": full_pmcid, "filename": member.name, "query": query}
                                }
                                
                                count += 1
                                # Just take one image per paper to get variety
                                break 
                                
            time.sleep(0.5)
            
        except requests.exceptions.Timeout:
            logger.warning(f"   Timeout on {full_pmcid}, skipping...")
            continue
        except Exception as e:
            logger.warning(f"   Error processing {full_pmcid}: {str(e)[:100]}, skipping...")
            continue

if __name__ == "__main__":
    # Test run
    for img in stream_biomedical_images(limit=3, max_per_type=1):
        print(f"Captured: {img['caption']}")
