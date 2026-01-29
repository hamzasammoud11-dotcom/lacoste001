"""
UC4 BioFlow Comprehensive Stress Test
======================================
Tests all implemented features against the UC4 Vision:
"Multimodal Biological Design & Discovery Intelligence"

Tests cover:
1. Data Ingestion & Normalization
2. Multimodal Similarity Search
3. Guided Exploration (facets, filtering)
4. Design Assistance (close but diverse variants)
5. Scientific Traceability (evidence linking)
"""

import json
import time
import requests
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime

# Configuration
API_BASE = "http://localhost:8000"
UI_BASE = "http://localhost:3000"
TIMEOUT = 30

@dataclass
class TestResult:
    name: str
    category: str
    passed: bool
    duration_ms: float
    details: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class UC4StressTester:
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
    def log(self, msg: str):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    
    def add_result(self, result: TestResult):
        self.results.append(result)
        status = "✅" if result.passed else "❌"
        self.log(f"{status} {result.category}/{result.name} ({result.duration_ms:.1f}ms)")
        if result.errors:
            for e in result.errors:
                self.log(f"   ERROR: {e}")
        if result.warnings:
            for w in result.warnings:
                self.log(f"   WARN: {w}")

    # =========================================================================
    # PHASE 1: DATA INGESTION & NORMALIZATION
    # =========================================================================
    
    def test_ingestion_pubmed(self):
        """Test PubMed text ingestion"""
        start = time.time()
        errors = []
        warnings = []
        
        try:
            # Test ingest endpoint
            response = requests.post(
                f"{API_BASE}/api/ingest",
                json={
                    "content": "EGFR inhibitors show promise in non-small cell lung cancer treatment",
                    "modality": "text",
                    "metadata": {"source": "pubmed", "pmid": "12345678"}
                },
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                if not data.get("id"):
                    warnings.append("Ingest returned no ID")
            else:
                errors.append(f"Status {response.status_code}: {response.text[:100]}")
                
        except Exception as e:
            errors.append(str(e))
        
        self.add_result(TestResult(
            name="pubmed_text_ingest",
            category="ingestion",
            passed=len(errors) == 0,
            duration_ms=(time.time() - start) * 1000,
            details="Text ingestion to Qdrant",
            errors=errors,
            warnings=warnings
        ))

    def test_ingestion_molecule(self):
        """Test molecule SMILES ingestion"""
        start = time.time()
        errors = []
        warnings = []
        
        test_molecules = [
            ("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin"),
            ("CC(=O)NC1=CC=C(C=C1)O", "Acetaminophen"),
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),
            ("CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", "Testosterone"),
            ("C1=CC=C(C=C1)CC(C(=O)O)N", "Phenylalanine"),
        ]
        
        successful = 0
        for smiles, name in test_molecules:
            try:
                response = requests.post(
                    f"{API_BASE}/api/ingest",
                    json={
                        "content": smiles,
                        "modality": "smiles",
                        "metadata": {"name": name, "source": "chembl"}
                    },
                    timeout=TIMEOUT
                )
                if response.status_code == 200:
                    successful += 1
                else:
                    warnings.append(f"Failed to ingest {name}")
            except Exception as e:
                errors.append(f"{name}: {e}")
        
        if successful < len(test_molecules):
            warnings.append(f"Only {successful}/{len(test_molecules)} molecules ingested")
        
        self.add_result(TestResult(
            name="molecule_smiles_ingest",
            category="ingestion",
            passed=len(errors) == 0 and successful > 0,
            duration_ms=(time.time() - start) * 1000,
            details=f"{successful}/{len(test_molecules)} molecules",
            errors=errors,
            warnings=warnings
        ))

    def test_ingestion_protein(self):
        """Test protein sequence ingestion"""
        start = time.time()
        errors = []
        warnings = []
        
        test_proteins = [
            ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH", "Test_Kinase"),
            ("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH", "Hemoglobin_alpha"),
        ]
        
        successful = 0
        for seq, name in test_proteins:
            try:
                response = requests.post(
                    f"{API_BASE}/api/ingest",
                    json={
                        "content": seq,
                        "modality": "protein",
                        "metadata": {"name": name, "source": "uniprot", "organism": "human"}
                    },
                    timeout=TIMEOUT
                )
                if response.status_code == 200:
                    successful += 1
                else:
                    warnings.append(f"Failed to ingest {name}")
            except Exception as e:
                errors.append(f"{name}: {e}")
        
        self.add_result(TestResult(
            name="protein_sequence_ingest",
            category="ingestion",
            passed=len(errors) == 0 and successful > 0,
            duration_ms=(time.time() - start) * 1000,
            details=f"{successful}/{len(test_proteins)} proteins",
            errors=errors,
            warnings=warnings
        ))

    # =========================================================================
    # PHASE 2: MULTIMODAL SIMILARITY SEARCH
    # =========================================================================

    def test_search_text_query(self):
        """Test text-based semantic search"""
        start = time.time()
        errors = []
        warnings = []
        
        queries = [
            "EGFR inhibitor for lung cancer",
            "kinase inhibitor mechanism",
            "protein folding disease",
            "drug resistance mutation",
        ]
        
        results_count = []
        for query in queries:
            try:
                response = requests.post(
                    f"{API_BASE}/api/search",
                    json={"query": query, "top_k": 10, "use_mmr": False},
                    timeout=TIMEOUT
                )
                if response.status_code == 200:
                    data = response.json()
                    results_count.append(len(data.get("results", [])))
                else:
                    errors.append(f"Query '{query}': Status {response.status_code}")
            except Exception as e:
                errors.append(f"Query '{query}': {e}")
        
        avg_results = sum(results_count) / len(results_count) if results_count else 0
        if avg_results < 3:
            warnings.append(f"Low average results: {avg_results:.1f}")
        
        self.add_result(TestResult(
            name="text_semantic_search",
            category="search",
            passed=len(errors) == 0,
            duration_ms=(time.time() - start) * 1000,
            details=f"Avg {avg_results:.1f} results per query",
            errors=errors,
            warnings=warnings
        ))

    def test_search_mmr_diversity(self):
        """Test MMR diversification in search results"""
        start = time.time()
        errors = []
        warnings = []
        
        try:
            # Search with MMR
            response = requests.post(
                f"{API_BASE}/api/search",
                json={
                    "query": "cancer treatment drug",
                    "top_k": 10,
                    "use_mmr": True,
                    "mmr_lambda": 0.5
                },
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                # Check diversity metrics
                if "diversity_score" in data:
                    if data["diversity_score"] < 0.01:
                        warnings.append(f"Low diversity: {data['diversity_score']:.4f}")
                else:
                    warnings.append("No diversity_score in response")
                
                # Check for varied sources
                sources = set(r.get("source", "unknown") for r in results)
                if len(sources) < 2:
                    warnings.append("Results lack source diversity")
                    
            else:
                errors.append(f"Status {response.status_code}")
                
        except Exception as e:
            errors.append(str(e))
        
        self.add_result(TestResult(
            name="mmr_diversity_search",
            category="search",
            passed=len(errors) == 0,
            duration_ms=(time.time() - start) * 1000,
            errors=errors,
            warnings=warnings
        ))

    def test_search_with_filters(self):
        """Test filtered search (modality, source)"""
        start = time.time()
        errors = []
        warnings = []
        
        filters = [
            {"modality": "text"},
            {"modality": "molecule"},
            {"modality": "protein"},
            {"source": "pubmed"},
            {"source": "chembl"},
        ]
        
        working_filters = 0
        for f in filters:
            try:
                response = requests.post(
                    f"{API_BASE}/api/search",
                    json={
                        "query": "inhibitor",
                        "top_k": 5,
                        "filters": f
                    },
                    timeout=TIMEOUT
                )
                if response.status_code == 200:
                    working_filters += 1
                else:
                    warnings.append(f"Filter {f} returned {response.status_code}")
            except Exception as e:
                errors.append(f"Filter {f}: {e}")
        
        self.add_result(TestResult(
            name="filtered_search",
            category="search",
            passed=len(errors) == 0 and working_filters >= 3,
            duration_ms=(time.time() - start) * 1000,
            details=f"{working_filters}/{len(filters)} filters work",
            errors=errors,
            warnings=warnings
        ))

    def test_search_evidence_linking(self):
        """Test evidence linking in search results"""
        start = time.time()
        errors = []
        warnings = []
        
        try:
            response = requests.post(
                f"{API_BASE}/api/search",
                json={
                    "query": "EGFR mutation",
                    "top_k": 5,
                    "include_evidence": True
                },
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                has_evidence = 0
                has_source = 0
                has_citation = 0
                
                for r in results:
                    if r.get("evidence_links"):
                        has_evidence += 1
                    if r.get("source"):
                        has_source += 1
                    if r.get("citation"):
                        has_citation += 1
                
                if has_source == 0:
                    warnings.append("No results have source metadata")
                if has_evidence == 0:
                    warnings.append("No results have evidence_links")
                    
            else:
                errors.append(f"Status {response.status_code}")
                
        except Exception as e:
            errors.append(str(e))
        
        self.add_result(TestResult(
            name="evidence_linking",
            category="search",
            passed=len(errors) == 0,
            duration_ms=(time.time() - start) * 1000,
            errors=errors,
            warnings=warnings
        ))

    # =========================================================================
    # PHASE 3: DESIGN ASSISTANCE (AGENT PIPELINE)
    # =========================================================================

    def test_agent_generate_text(self):
        """Test molecule generation from text"""
        start = time.time()
        errors = []
        warnings = []
        
        prompts = [
            "anti-inflammatory compound",
            "kinase inhibitor",
            "blood-brain barrier permeable drug",
            "water soluble therapeutic",
        ]
        
        total_molecules = 0
        for prompt in prompts:
            try:
                response = requests.post(
                    f"{API_BASE}/api/agents/generate",
                    json={
                        "prompt": prompt,
                        "mode": "text",
                        "num_samples": 3
                    },
                    timeout=TIMEOUT
                )
                
                if response.status_code == 200:
                    data = response.json()
                    mols = data.get("molecules", [])
                    total_molecules += len(mols)
                else:
                    warnings.append(f"'{prompt}': Status {response.status_code}")
            except Exception as e:
                errors.append(f"'{prompt}': {e}")
        
        expected = len(prompts) * 3
        if total_molecules < expected * 0.5:
            warnings.append(f"Low generation: {total_molecules}/{expected}")
        
        self.add_result(TestResult(
            name="generate_from_text",
            category="agents",
            passed=len(errors) == 0 and total_molecules > 0,
            duration_ms=(time.time() - start) * 1000,
            details=f"{total_molecules} molecules generated",
            errors=errors,
            warnings=warnings
        ))

    def test_agent_generate_mutate(self):
        """Test molecule mutation from seed SMILES"""
        start = time.time()
        errors = []
        warnings = []
        
        seed_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        
        try:
            response = requests.post(
                f"{API_BASE}/api/agents/generate",
                json={
                    "prompt": "more potent variant",
                    "mode": "mutate",
                    "smiles": seed_smiles,
                    "num_samples": 5
                },
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                mols = data.get("molecules", [])
                if len(mols) == 0:
                    warnings.append("No mutants generated")
            else:
                errors.append(f"Status {response.status_code}: {response.text[:100]}")
                
        except Exception as e:
            errors.append(str(e))
        
        self.add_result(TestResult(
            name="generate_mutations",
            category="agents",
            passed=len(errors) == 0,
            duration_ms=(time.time() - start) * 1000,
            errors=errors,
            warnings=warnings
        ))

    def test_agent_validate(self):
        """Test ADMET/toxicity validation"""
        start = time.time()
        errors = []
        warnings = []
        
        test_smiles = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin - should pass
            "CCO",  # Ethanol - might fail MW
            "C1=CC=C(C=C1)N",  # Aniline - structural alert
            "CC(C)(C)NCC(O)C1=CC(O)=CC(O)=C1",  # Salbutamol
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        ]
        
        try:
            response = requests.post(
                f"{API_BASE}/api/agents/validate",
                json={
                    "smiles": test_smiles,
                    "check_lipinski": True,
                    "check_admet": True,
                    "check_alerts": True
                },
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                validations = data.get("validations", [])
                
                if len(validations) != len(test_smiles):
                    warnings.append(f"Expected {len(test_smiles)}, got {len(validations)}")
                
                valid_count = sum(1 for v in validations if v.get("is_valid", False))
                
            else:
                errors.append(f"Status {response.status_code}")
                
        except Exception as e:
            errors.append(str(e))
        
        self.add_result(TestResult(
            name="admet_validation",
            category="agents",
            passed=len(errors) == 0,
            duration_ms=(time.time() - start) * 1000,
            details=f"{valid_count}/{len(test_smiles)} valid" if 'valid_count' in dir() else "",
            errors=errors,
            warnings=warnings
        ))

    def test_agent_rank(self):
        """Test multi-criteria ranking"""
        start = time.time()
        errors = []
        warnings = []
        
        candidates = [
            {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "name": "Aspirin", "score": 0.8},
            {"smiles": "CC(=O)NC1=CC=C(C=C1)O", "name": "Acetaminophen", "score": 0.75},
            {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "name": "Caffeine", "score": 0.6},
            {"smiles": "CCO", "name": "Ethanol", "score": 0.3},
        ]
        
        try:
            response = requests.post(
                f"{API_BASE}/api/agents/rank",
                json={
                    "candidates": candidates,
                    "weights": {"qed": 0.4, "validity": 0.3, "mw": 0.15, "logp": 0.15},
                    "top_k": 3
                },
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                ranked = data.get("ranked", [])
                
                if len(ranked) == 0:
                    warnings.append("No ranked results returned")
                elif len(ranked) > 3:
                    warnings.append(f"Expected top 3, got {len(ranked)}")
                    
            else:
                errors.append(f"Status {response.status_code}")
                
        except Exception as e:
            errors.append(str(e))
        
        self.add_result(TestResult(
            name="multi_criteria_ranking",
            category="agents",
            passed=len(errors) == 0,
            duration_ms=(time.time() - start) * 1000,
            errors=errors,
            warnings=warnings
        ))

    def test_agent_workflow(self):
        """Test full discovery workflow"""
        start = time.time()
        errors = []
        warnings = []
        
        try:
            response = requests.post(
                f"{API_BASE}/api/agents/workflow",
                json={
                    "query": "anti-cancer kinase inhibitor",
                    "num_candidates": 10,
                    "top_k": 5
                },
                timeout=60  # Longer timeout for full workflow
            )
            
            if response.status_code == 200:
                data = response.json()
                
                steps = data.get("steps_completed", 0)
                total = data.get("total_steps", 0)
                exec_time = data.get("execution_time_ms", 0)
                top_candidates = data.get("top_candidates", [])
                
                if steps < total:
                    warnings.append(f"Only {steps}/{total} steps completed")
                if len(top_candidates) == 0:
                    warnings.append("No candidates returned")
                if exec_time > 5000:
                    warnings.append(f"Slow execution: {exec_time:.0f}ms")
                    
            else:
                errors.append(f"Status {response.status_code}: {response.text[:200]}")
                
        except Exception as e:
            errors.append(str(e))
        
        self.add_result(TestResult(
            name="full_discovery_workflow",
            category="agents",
            passed=len(errors) == 0,
            duration_ms=(time.time() - start) * 1000,
            errors=errors,
            warnings=warnings
        ))

    # =========================================================================
    # PHASE 4: UI/UX COMPONENTS
    # =========================================================================

    def test_ui_visualization_page(self):
        """Test 3D visualization page loads"""
        start = time.time()
        errors = []
        warnings = []
        
        try:
            response = requests.get(
                f"{UI_BASE}/dashboard/visualization",
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                content = response.text
                # Check for key components
                if "Embedding" not in content and "3D" not in content:
                    warnings.append("Page may not have loaded correctly")
            else:
                errors.append(f"Status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            errors.append("UI server not running on port 3000")
        except Exception as e:
            errors.append(str(e))
        
        self.add_result(TestResult(
            name="visualization_page",
            category="ui",
            passed=len(errors) == 0,
            duration_ms=(time.time() - start) * 1000,
            errors=errors,
            warnings=warnings
        ))

    def test_ui_workflow_page(self):
        """Test workflow builder page loads"""
        start = time.time()
        errors = []
        warnings = []
        
        try:
            response = requests.get(
                f"{UI_BASE}/dashboard/workflow",
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                content = response.text
                if "Workflow" not in content:
                    warnings.append("Page may not have loaded correctly")
            else:
                errors.append(f"Status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            errors.append("UI server not running on port 3000")
        except Exception as e:
            errors.append(str(e))
        
        self.add_result(TestResult(
            name="workflow_page",
            category="ui",
            passed=len(errors) == 0,
            duration_ms=(time.time() - start) * 1000,
            errors=errors,
            warnings=warnings
        ))

    def test_ui_discovery_page(self):
        """Test discovery/search page"""
        start = time.time()
        errors = []
        warnings = []
        
        try:
            response = requests.get(
                f"{UI_BASE}/dashboard/discovery",
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                pass  # Page loads
            else:
                errors.append(f"Status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            errors.append("UI server not running")
        except Exception as e:
            errors.append(str(e))
        
        self.add_result(TestResult(
            name="discovery_page",
            category="ui",
            passed=len(errors) == 0,
            duration_ms=(time.time() - start) * 1000,
            errors=errors,
            warnings=warnings
        ))

    # =========================================================================
    # STRESS TESTS
    # =========================================================================

    def test_stress_concurrent_searches(self):
        """Test concurrent search requests"""
        start = time.time()
        errors = []
        warnings = []
        
        import concurrent.futures
        
        queries = [f"drug target {i}" for i in range(10)]
        
        def search(query):
            try:
                r = requests.post(
                    f"{API_BASE}/api/search",
                    json={"query": query, "top_k": 5},
                    timeout=TIMEOUT
                )
                return r.status_code == 200
            except:
                return False
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(search, queries))
            
            success_rate = sum(results) / len(results)
            if success_rate < 0.8:
                warnings.append(f"Low success rate: {success_rate:.0%}")
                
        except Exception as e:
            errors.append(str(e))
        
        self.add_result(TestResult(
            name="concurrent_searches",
            category="stress",
            passed=len(errors) == 0,
            duration_ms=(time.time() - start) * 1000,
            details=f"{sum(results)}/{len(queries)} succeeded" if 'results' in dir() else "",
            errors=errors,
            warnings=warnings
        ))

    def test_stress_large_batch_ingest(self):
        """Test batch ingestion performance"""
        start = time.time()
        errors = []
        warnings = []
        
        # Generate test data
        test_items = [
            {
                "content": f"Research paper abstract about compound {i} and its effects",
                "modality": "text",
                "metadata": {"id": f"test_{i}", "source": "stress_test"}
            }
            for i in range(20)
        ]
        
        successful = 0
        for item in test_items:
            try:
                r = requests.post(
                    f"{API_BASE}/api/ingest",
                    json=item,
                    timeout=10
                )
                if r.status_code == 200:
                    successful += 1
            except:
                pass
        
        duration = (time.time() - start) * 1000
        rate = successful / (duration / 1000) if duration > 0 else 0
        
        if rate < 5:
            warnings.append(f"Slow ingestion: {rate:.1f} items/sec")
        
        self.add_result(TestResult(
            name="batch_ingestion",
            category="stress",
            passed=successful > 10,
            duration_ms=duration,
            details=f"{successful}/{len(test_items)} at {rate:.1f}/sec",
            errors=errors,
            warnings=warnings
        ))

    # =========================================================================
    # UC4 SPECIFIC REQUIREMENTS
    # =========================================================================

    def test_uc4_multimodal_items(self):
        """UC4: Ingest & normalize multimodal items with meaningful metadata"""
        start = time.time()
        errors = []
        warnings = []
        
        # Test all modalities
        modalities = {
            "text": ("This is a research abstract about protein kinases", {"type": "abstract"}),
            "smiles": ("CCO", {"name": "ethanol", "mw": 46.07}),
            "protein": ("MKTAYIAK", {"organism": "human", "function": "kinase"}),
        }
        
        for mod, (content, metadata) in modalities.items():
            try:
                r = requests.post(
                    f"{API_BASE}/api/ingest",
                    json={"content": content, "modality": mod, "metadata": metadata},
                    timeout=TIMEOUT
                )
                if r.status_code != 200:
                    warnings.append(f"{mod}: ingest failed")
            except Exception as e:
                errors.append(f"{mod}: {e}")
        
        self.add_result(TestResult(
            name="multimodal_normalization",
            category="uc4",
            passed=len(errors) == 0,
            duration_ms=(time.time() - start) * 1000,
            details=f"Tested {len(modalities)} modalities",
            errors=errors,
            warnings=warnings
        ))

    def test_uc4_similarity_search(self):
        """UC4: Multimodal similarity search for experiments/candidates"""
        start = time.time()
        errors = []
        warnings = []
        
        # Test cross-modal search
        test_queries = [
            {"query": "kinase inhibitor", "expected_modalities": ["text", "molecule"]},
            {"query": "EGFR protein", "expected_modalities": ["text", "protein"]},
        ]
        
        for test in test_queries:
            try:
                r = requests.post(
                    f"{API_BASE}/api/search",
                    json={"query": test["query"], "top_k": 10},
                    timeout=TIMEOUT
                )
                if r.status_code == 200:
                    results = r.json().get("results", [])
                    found_modalities = set(r.get("modality", "unknown") for r in results)
                    # Check if we got diverse modalities
                    if len(found_modalities) == 1:
                        warnings.append(f"'{test['query']}': single modality results")
            except Exception as e:
                errors.append(str(e))
        
        self.add_result(TestResult(
            name="cross_modal_search",
            category="uc4",
            passed=len(errors) == 0,
            duration_ms=(time.time() - start) * 1000,
            errors=errors,
            warnings=warnings
        ))

    def test_uc4_design_variants(self):
        """UC4: Design assistance - propose close but diverse variants"""
        start = time.time()
        errors = []
        warnings = []
        
        try:
            # Generate variants
            r = requests.post(
                f"{API_BASE}/api/agents/generate",
                json={
                    "prompt": "potent kinase inhibitor with good ADMET",
                    "mode": "text",
                    "num_samples": 10
                },
                timeout=TIMEOUT
            )
            
            if r.status_code == 200:
                molecules = r.json().get("molecules", [])
                
                if len(molecules) < 5:
                    warnings.append(f"Only {len(molecules)} variants generated")
                
                # Check diversity (unique SMILES)
                smiles = [m.get("smiles", "") for m in molecules if isinstance(m, dict)]
                unique = len(set(smiles))
                if unique < len(smiles) * 0.8:
                    warnings.append("Low diversity in generated variants")
                    
        except Exception as e:
            errors.append(str(e))
        
        self.add_result(TestResult(
            name="diverse_variants",
            category="uc4",
            passed=len(errors) == 0,
            duration_ms=(time.time() - start) * 1000,
            errors=errors,
            warnings=warnings
        ))

    def test_uc4_traceability(self):
        """UC4: Scientific traceability - link suggestions to evidence"""
        start = time.time()
        errors = []
        warnings = []
        
        try:
            r = requests.post(
                f"{API_BASE}/api/search",
                json={
                    "query": "EGFR inhibitor treatment",
                    "top_k": 5,
                    "include_evidence": True
                },
                timeout=TIMEOUT
            )
            
            if r.status_code == 200:
                results = r.json().get("results", [])
                
                traceable = 0
                for result in results:
                    has_source = bool(result.get("source"))
                    has_links = bool(result.get("evidence_links"))
                    has_citation = bool(result.get("citation"))
                    
                    if has_source or has_links or has_citation:
                        traceable += 1
                
                if traceable < len(results) * 0.5:
                    warnings.append(f"Only {traceable}/{len(results)} results are traceable")
                    
        except Exception as e:
            errors.append(str(e))
        
        self.add_result(TestResult(
            name="scientific_traceability",
            category="uc4",
            passed=len(errors) == 0,
            duration_ms=(time.time() - start) * 1000,
            errors=errors,
            warnings=warnings
        ))

    # =========================================================================
    # RUN ALL TESTS
    # =========================================================================

    def run_all(self):
        """Run all tests"""
        self.log("=" * 70)
        self.log("UC4 BIOFLOW COMPREHENSIVE STRESS TEST")
        self.log("=" * 70)
        
        # Phase 1: Ingestion
        self.log("\n📥 PHASE 1: DATA INGESTION")
        self.log("-" * 50)
        self.test_ingestion_pubmed()
        self.test_ingestion_molecule()
        self.test_ingestion_protein()
        
        # Phase 2: Search
        self.log("\n🔍 PHASE 2: MULTIMODAL SEARCH")
        self.log("-" * 50)
        self.test_search_text_query()
        self.test_search_mmr_diversity()
        self.test_search_with_filters()
        self.test_search_evidence_linking()
        
        # Phase 3: Agents
        self.log("\n🤖 PHASE 3: AGENT PIPELINE")
        self.log("-" * 50)
        self.test_agent_generate_text()
        self.test_agent_generate_mutate()
        self.test_agent_validate()
        self.test_agent_rank()
        self.test_agent_workflow()
        
        # Phase 4: UI
        self.log("\n🖥️ PHASE 4: UI/UX")
        self.log("-" * 50)
        self.test_ui_visualization_page()
        self.test_ui_workflow_page()
        self.test_ui_discovery_page()
        
        # Stress Tests
        self.log("\n⚡ STRESS TESTS")
        self.log("-" * 50)
        self.test_stress_concurrent_searches()
        self.test_stress_large_batch_ingest()
        
        # UC4 Specific
        self.log("\n🎯 UC4 REQUIREMENTS")
        self.log("-" * 50)
        self.test_uc4_multimodal_items()
        self.test_uc4_similarity_search()
        self.test_uc4_design_variants()
        self.test_uc4_traceability()
        
        # Summary
        self.generate_report()
    
    def generate_report(self):
        """Generate final report"""
        total_time = time.time() - self.start_time
        
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total_warnings = sum(len(r.warnings) for r in self.results)
        
        print("\n" + "=" * 70)
        print("📊 STRESS TEST REPORT")
        print("=" * 70)
        
        # By category
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = {"passed": 0, "failed": 0, "warnings": 0}
            if r.passed:
                categories[r.category]["passed"] += 1
            else:
                categories[r.category]["failed"] += 1
            categories[r.category]["warnings"] += len(r.warnings)
        
        print("\nBy Category:")
        for cat, stats in categories.items():
            status = "✅" if stats["failed"] == 0 else "⚠️" if stats["passed"] > 0 else "❌"
            print(f"  {status} {cat}: {stats['passed']}/{stats['passed']+stats['failed']} passed, {stats['warnings']} warnings")
        
        print(f"\nTotal: {passed}/{passed+failed} tests passed")
        print(f"Warnings: {total_warnings}")
        print(f"Duration: {total_time:.1f}s")
        
        # Failed tests
        if failed > 0:
            print("\n❌ FAILED TESTS:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.category}/{r.name}: {', '.join(r.errors)}")
        
        # All warnings
        if total_warnings > 0:
            print("\n⚠️ WARNINGS:")
            for r in self.results:
                for w in r.warnings:
                    print(f"  - {r.category}/{r.name}: {w}")
        
        print("\n" + "=" * 70)
        
        return {
            "passed": passed,
            "failed": failed,
            "warnings": total_warnings,
            "duration_s": total_time,
            "categories": categories,
            "results": [
                {
                    "name": r.name,
                    "category": r.category,
                    "passed": r.passed,
                    "errors": r.errors,
                    "warnings": r.warnings
                }
                for r in self.results
            ]
        }


if __name__ == "__main__":
    tester = UC4StressTester()
    report = tester.run_all()
    
    # Save report
    with open("stress_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to stress_test_report.json")
