"""
Base Agent Classes
==================

Foundation classes for BioFlow agents.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod

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
    """Message passed between agents in a workflow."""
    sender: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sender": self.sender,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class AgentContext:
    """
    Shared context passed through a workflow.
    
    Carries state between agents for multi-step processing.
    """
    query: str
    modality: str = "text"
    target_properties: Dict[str, Any] = field(default_factory=dict)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    history: List[AgentMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: AgentMessage) -> None:
        """Add a message to history."""
        self.history.append(message)
    
    def get_last_output(self) -> Optional[Any]:
        """Get the output from the last agent."""
        if self.history:
            return self.history[-1].content
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "modality": self.modality,
            "target_properties": self.target_properties,
            "candidates": self.candidates,
            "evidence": self.evidence,
            "history": [m.to_dict() for m in self.history],
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all BioFlow agents.
    
    Each agent processes input and produces output that can be passed
    to the next agent in a workflow.
    """
    
    def __init__(
        self,
        name: str,
        agent_type: AgentType,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize agent.
        
        Args:
            name: Unique agent name
            agent_type: Type of agent (generator, validator, etc.)
            config: Optional configuration dictionary
        """
        self.name = name
        self.agent_type = agent_type
        self.config = config or {}
        self._initialized = False
        logger.info(f"Agent {name} ({agent_type.value}) created")
    
    def initialize(self) -> None:
        """
        Initialize agent resources (models, connections, etc.).
        
        Override in subclasses for lazy initialization.
        """
        self._initialized = True
        logger.info(f"Agent {self.name} initialized")
    
    @abstractmethod
    def process(
        self,
        input_data: Any,
        context: Optional[AgentContext] = None,
    ) -> AgentMessage:
        """
        Process input and return output message.
        
        Args:
            input_data: Input to process (query, SMILES, etc.)
            context: Optional shared context
            
        Returns:
            AgentMessage with results
        """
        pass
    
    def __call__(
        self,
        input_data: Any,
        context: Optional[AgentContext] = None,
    ) -> AgentMessage:
        """Allow calling agent as a function."""
        if not self._initialized:
            self.initialize()
        return self.process(input_data, context)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, type={self.agent_type.value})"
