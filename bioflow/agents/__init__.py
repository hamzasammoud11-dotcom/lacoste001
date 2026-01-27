"""
BioFlow Agents Module
=====================

Agent implementations for discovery workflows:
- GeneratorAgent: Molecule generation from text prompts
- ValidatorAgent: ADMET/toxicity validation
- RankerAgent: Score-based ranking with feedback
- WorkflowEngine: Multi-step workflow orchestration
- DiscoveryWorkflow: Pre-configured drug discovery pipeline
"""

from bioflow.agents.base import (
    BaseAgent,
    AgentMessage,
    AgentContext,
    AgentType,
)
from bioflow.agents.generator import GeneratorAgent, GeneratedMolecule
from bioflow.agents.validator import ValidatorAgent, ValidationResult, ValidationStatus
from bioflow.agents.ranker import RankerAgent, RankedCandidate, FeedbackLoop
from bioflow.agents.workflow import WorkflowEngine, DiscoveryWorkflow, WorkflowResult

__all__ = [
    # Base
    "BaseAgent",
    "AgentMessage",
    "AgentContext",
    "AgentType",
    # Generator
    "GeneratorAgent",
    "GeneratedMolecule",
    # Validator
    "ValidatorAgent",
    "ValidationResult",
    "ValidationStatus",
    # Ranker
    "RankerAgent",
    "RankedCandidate",
    "FeedbackLoop",
    # Workflow
    "WorkflowEngine",
    "DiscoveryWorkflow",
    "WorkflowResult",
]
