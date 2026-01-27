
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from bioflow.core import (
    BioFlowOrchestrator, 
    ToolRegistry, 
    WorkflowConfig, 
    Modality,
    BioEncoder,
    BioPredictor,
    BioRetriever,
    EmbeddingResult,
    PredictionResult,
    RetrievalResult
)

# 1. Create Mock Tools for Verification
class MockEncoder(BioEncoder):
    def encode(self, content, modality):
        print(f"  [Mock] Encoding {modality.value}: {content[:20]}...")
        return EmbeddingResult(vector=[0.1]*768, modality=modality, dimension=768)
    @property
    def dimension(self): return 768

class MockPredictor(BioPredictor):
    def predict(self, drug, target):
        print(f"  [Mock] Predicting interaction for drug candidate...")
        return PredictionResult(score=0.85, confidence=0.9)

class MockRetriever(BioRetriever):
    def search(self, query, limit=10, filters=None, **kwargs):
        print(f"  [Mock] Searching Vector DB for neighbors...")
        return [RetrievalResult(id="mol_1", score=0.95, content="CCO", modality=Modality.SMILES)]
    def ingest(self, content, modality, payload=None): return "1"

# 2. Setup Registry
print("--- 🛠️ Step 1: Tool Registry Setup ---")
ToolRegistry.register_encoder("obm", MockEncoder())
ToolRegistry.register_predictor("deeppurpose", MockPredictor())
print(ToolRegistry.summary())
print()

# 3. Load and Visualize Workflow
print("--- 🗺️ Step 2: Workflow DAG Resolution ---")
workflow_path = Path("bioflow/workflows/drug_discovery.yaml")
import yaml
with open(workflow_path, 'r') as f:
    config_dict = yaml.safe_load(f)

orchestrator = BioFlowOrchestrator()
orchestrator.set_retriever(MockRetriever()) # Configured the retriever
config = WorkflowConfig.from_dict(config_dict)
orchestrator.register_workflow(config)

# Show topological order
order = orchestrator._build_execution_order(config)
print(f"Workflow: {config.name}")
print("Execution Sequence (Topological Sort):")
for i, node in enumerate(order):
    deps = f" (depends on: {', '.join(node.inputs)})" if node.inputs else ""
    print(f"  {i+1}. [{node.id}] Type: {node.type.value}{deps}")
print()

# 4. Perform Dry Run
print("--- 🚀 Step 3: Pipeline Execution Dry Run ---")
input_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" # Caffeine
result = orchestrator.run("drug_discovery_basic", input_smiles)

print("\n--- 📊 Final Report ---")
if result.success:
    print(f"Status: ✅ SUCCESS")
    print(f"Final Output Score: {result.output[0].score if isinstance(result.output, list) else result.output.score}")
    print(f"Duration: {result.duration_ms:.2f} ms")
else:
    print(f"Status: ❌ FAILED")
    print(f"Errors: {result.context.errors}")
