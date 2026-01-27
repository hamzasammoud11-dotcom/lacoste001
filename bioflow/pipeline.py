"""
BioFlow Pipeline - Workflow Orchestration
==========================================

This module provides the pipeline orchestration for BioFlow,
connecting agents, memory (Qdrant), and OBM encoders.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from bioflow.obm_wrapper import OBMWrapper
from bioflow.qdrant_manager import QdrantManager, SearchResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents in the BioFlow system."""
    GENERATOR = "generator"      # Generates new molecules/variants
    VALIDATOR = "validator"      # Validates properties (toxicity, etc.)
    MINER = "miner"              # Mines literature for evidence
    RANKER = "ranker"            # Ranks candidates
    CUSTOM = "custom"


@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""
    success: bool
    outputs: List[Any]
    messages: List[AgentMessage]
    stats: Dict[str, Any]


class BaseAgent:
    """Base class for all BioFlow agents."""
    
    def __init__(
        self,
        name: str,
        agent_type: AgentType,
        obm: OBMWrapper,
        qdrant: QdrantManager
    ):
        self.name = name
        self.agent_type = agent_type
        self.obm = obm
        self.qdrant = qdrant
    
    def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> AgentMessage:
        """Process input and return output message."""
        raise NotImplementedError


class MinerAgent(BaseAgent):
    """
    Literature mining agent.
    
    Retrieves relevant scientific articles/abstracts based on query.
    """
    
    def __init__(self, obm: OBMWrapper, qdrant: QdrantManager, collection: Optional[str] = None):
        super().__init__("LiteratureMiner", AgentType.MINER, obm, qdrant)
        self.collection = collection
    
    def process(
        self, 
        input_data: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """
        Search for relevant literature.
        
        Args:
            input_data: Query text, SMILES, or protein sequence.
            context: Optional context with 'modality', 'limit'.
        """
        context = context or {}
        modality = context.get("modality", "text")
        limit = context.get("limit", 5)
        
        results = self.qdrant.search(
            query=input_data,
            query_modality=modality,
            collection=self.collection,
            limit=limit,
            filter_modality="text"
        )
        
        return AgentMessage(
            sender=self.name,
            content=[r.payload for r in results],
            metadata={
                "query": input_data,
                "modality": modality,
                "result_count": len(results),
                "top_score": results[0].score if results else 0
            }
        )


class ValidatorAgent(BaseAgent):
    """
    Validation agent.
    
    Checks molecules against known toxicity, drug-likeness, etc.
    """
    
    def __init__(self, obm: OBMWrapper, qdrant: QdrantManager, collection: Optional[str] = None):
        super().__init__("Validator", AgentType.VALIDATOR, obm, qdrant)
        self.collection = collection
    
    def process(
        self,
        input_data: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """
        Validate a molecule.
        
        Args:
            input_data: SMILES string to validate.
            context: Optional context.
        """
        context = context or {}
        
        # Search for similar known molecules
        similar = self.qdrant.search(
            query=input_data,
            query_modality="smiles",
            collection=self.collection,
            limit=10,
            filter_modality="smiles"
        )
        
        # Basic validation flags
        validation = {
            "has_similar_known": len(similar) > 0,
            "max_similarity": similar[0].score if similar else 0,
            "similar_molecules": [
                {
                    "smiles": r.content,
                    "score": r.score,
                    "tags": r.payload.get("tags", [])
                }
                for r in similar[:3]
            ]
        }
        
        # Flag potential issues based on tags of similar molecules
        risk_tags = ["toxic", "mutagenic", "carcinogenic"]
        flagged_risks = []
        for r in similar:
            tags = r.payload.get("tags", [])
            for tag in tags:
                if any(risk in tag.lower() for risk in risk_tags):
                    flagged_risks.append({"molecule": r.content, "tag": tag})
        
        validation["flagged_risks"] = flagged_risks
        validation["passed"] = len(flagged_risks) == 0
        
        return AgentMessage(
            sender=self.name,
            content=validation,
            metadata={"input_smiles": input_data}
        )


class RankerAgent(BaseAgent):
    """
    Ranking agent.
    
    Ranks candidates based on multiple criteria.
    """
    
    def __init__(self, obm: OBMWrapper, qdrant: QdrantManager):
        super().__init__("Ranker", AgentType.RANKER, obm, qdrant)
    
    def process(
        self,
        input_data: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """
        Rank a list of candidates.
        
        Args:
            input_data: List of candidate dicts with 'content', 'scores'.
        """
        # Simple weighted ranking
        ranked = sorted(
            input_data,
            key=lambda x: sum(x.get("scores", {}).values()),
            reverse=True
        )
        
        return AgentMessage(
            sender=self.name,
            content=ranked,
            metadata={"original_count": len(input_data)}
        )


class BioFlowPipeline:
    """
    Main pipeline orchestrator for BioFlow.
    
    Connects multiple agents in a workflow.
    """
    
    def __init__(
        self,
        obm: OBMWrapper,
        qdrant: QdrantManager
    ):
        self.obm = obm
        self.qdrant = qdrant
        self.agents: Dict[str, BaseAgent] = {}
        self.workflow: List[str] = []
        self.messages: List[AgentMessage] = []
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the pipeline."""
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name} ({agent.agent_type.value})")
    
    def set_workflow(self, agent_names: List[str]) -> None:
        """
        Set the workflow order.
        
        Args:
            agent_names: List of agent names in execution order.
        """
        for name in agent_names:
            if name not in self.agents:
                raise ValueError(f"Unknown agent: {name}")
        self.workflow = agent_names
    
    def run(
        self,
        initial_input: Any,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Execute the pipeline.
        
        Args:
            initial_input: Starting input data.
            initial_context: Initial context for first agent.
            
        Returns:
            PipelineResult with all outputs and messages.
        """
        self.messages = []
        current_input = initial_input
        current_context = initial_context or {}
        outputs = []
        
        for agent_name in self.workflow:
            agent = self.agents[agent_name]
            logger.info(f"Executing agent: {agent_name}")
            
            try:
                message = agent.process(current_input, current_context)
                self.messages.append(message)
                outputs.append(message.content)
                
                # Pass output to next agent
                current_input = message.content
                current_context.update(message.metadata)
                
            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}")
                return PipelineResult(
                    success=False,
                    outputs=outputs,
                    messages=self.messages,
                    stats={"failed_at": agent_name, "error": str(e)}
                )
        
        return PipelineResult(
            success=True,
            outputs=outputs,
            messages=self.messages,
            stats={
                "agents_executed": len(self.workflow),
                "total_messages": len(self.messages)
            }
        )
    
    def run_discovery_workflow(
        self,
        query: str,
        query_modality: str = "text",
        target_modality: str = "smiles"
    ) -> Dict[str, Any]:
        """
        Run a complete discovery workflow.
        
        1. Search for related literature
        2. Find similar molecules
        3. Validate candidates
        4. Return ranked results
        """
        results = {
            "query": query,
            "query_modality": query_modality,
            "target_modality": target_modality,
            "stages": {}
        }
        
        # Stage 1: Literature search
        literature = self.qdrant.search(
            query=query,
            query_modality=query_modality,
            limit=5,
            filter_modality="text"
        )
        results["stages"]["literature"] = [
            {"content": r.content, "score": r.score}
            for r in literature
        ]
        
        # Stage 2: Cross-modal molecule search
        molecules = self.qdrant.cross_modal_search(
            query=query,
            query_modality=query_modality,
            target_modality=target_modality,
            limit=10
        )
        results["stages"]["molecules"] = [
            {"content": r.content, "score": r.score, "payload": r.payload}
            for r in molecules
        ]
        
        # Stage 3: Validate top candidates
        if "Validator" in self.agents and molecules:
            validated = []
            for mol in molecules[:3]:
                val_msg = self.agents["Validator"].process(mol.content)
                validated.append({
                    "smiles": mol.content,
                    "validation": val_msg.content
                })
            results["stages"]["validation"] = validated
        
        # Stage 4: Diversity analysis
        diversity = self.qdrant.get_neighbors_diversity(
            query=query,
            query_modality=query_modality,
            k=10
        )
        results["stages"]["diversity"] = diversity
        
        return results
