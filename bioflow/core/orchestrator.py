"""
BioFlow Orchestrator
=====================

Stateful pipeline engine that manages the flow of data through
registered tools, forming a Directed Acyclic Graph (DAG) of operations.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from typing import Optional as OptionalType
from bioflow.core.base import BioEncoder, BioPredictor, BioGenerator, Modality
from bioflow.core.config import NodeConfig, WorkflowConfig, NodeType
from bioflow.core.registry import ToolRegistry

# Re-import Optional with a different name to avoid conflicts
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Context passed through pipeline execution."""
    workflow_id: str
    start_time: datetime = field(default_factory=datetime.now)
    node_outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def get_input(self, node_id: str) -> Any:
        """Get output from a previous node."""
        return self.node_outputs.get(node_id)
    
    def set_output(self, node_id: str, value: Any):
        """Store output from a node."""
        self.node_outputs[node_id] = value


@dataclass
class PipelineResult:
    """Final result of workflow execution."""
    success: bool
    output: Any
    context: ExecutionContext
    duration_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "duration_ms": self.duration_ms,
            "errors": self.context.errors,
            "node_outputs": {k: str(v)[:100] for k, v in self.context.node_outputs.items()}
        }


class BioFlowOrchestrator:
    """
    Main orchestration engine for BioFlow pipelines.
    
    Responsibilities:
    - Parse workflow configurations
    - Build execution DAG from node dependencies
    - Execute nodes in topological order
    - Manage state between nodes
    - Handle errors and retries
    
    Example:
        >>> orchestrator = BioFlowOrchestrator()
        >>> orchestrator.register_workflow(workflow_config)
        >>> result = orchestrator.run("my_workflow", input_data)
    """
    
    def __init__(self, registry: Optional[ToolRegistry] = None):
        """
        Initialize orchestrator.
        
        Args:
            registry: Tool registry instance. Uses global if None.
        """
        self.registry = registry if registry is not None else ToolRegistry
        self.workflows: Dict[str, WorkflowConfig] = {}
        self.custom_handlers: Dict[str, Callable] = {}
        self._retriever = None  # Qdrant manager reference
        
    def set_retriever(self, retriever):
        """Set the vector DB retriever (QdrantManager)."""
        self._retriever = retriever
    
    def register_workflow(self, config: WorkflowConfig) -> None:
        """Register a workflow configuration."""
        self.workflows[config.name] = config
        logger.info(f"Registered workflow: {config.name} ({len(config.nodes)} nodes)")
    
    def register_custom_handler(self, name: str, handler: Callable) -> None:
        """Register a custom node handler function."""
        self.custom_handlers[name] = handler
        logger.info(f"Registered custom handler: {name}")
    
    def _build_execution_order(self, config: WorkflowConfig) -> List[NodeConfig]:
        """
        Build topological execution order from node dependencies.
        
        Returns nodes sorted so dependencies are executed first.
        """
        # Build adjacency list
        in_degree = defaultdict(int)
        dependents = defaultdict(list)
        node_map = {node.id: node for node in config.nodes}
        
        for node in config.nodes:
            for dep in node.inputs:
                if dep != "input" and dep in node_map:
                    dependents[dep].append(node.id)
                    in_degree[node.id] += 1
        
        # Kahn's algorithm for topological sort
        queue = [n.id for n in config.nodes if in_degree[n.id] == 0]
        order = []
        
        while queue:
            node_id = queue.pop(0)
            order.append(node_map[node_id])
            for dependent in dependents[node_id]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(order) != len(config.nodes):
            raise ValueError("Cycle detected in workflow DAG")
        
        return order
    
    def _execute_node(
        self,
        node: NodeConfig,
        context: ExecutionContext,
        initial_input: Any
    ) -> Any:
        """Execute a single node and return its output."""
        
        # Gather inputs
        inputs = []
        for inp in node.inputs:
            if inp == "input":
                inputs.append(initial_input)
            else:
                inputs.append(context.get_input(inp))
        
        # Single input case
        node_input = inputs[0] if len(inputs) == 1 else inputs
        
        logger.debug(f"Executing node: {node.id} (type={node.type.value})")
        
        try:
            if node.type == NodeType.ENCODE:
                encoder = self.registry.get_encoder(node.tool)
                modality = Modality(node.params.get("modality", "text"))
                return encoder.encode(node_input, modality)
            
            elif node.type == NodeType.PREDICT:
                predictor = self.registry.get_predictor(node.tool)
                drug: str = str(node.params.get("drug") or node_input)
                target: str = str(node.params.get("target") or node.params.get("target_input") or "")
                return predictor.predict(drug, target)
            
            elif node.type == NodeType.RETRIEVE:
                if self._retriever is None:
                    raise ValueError("No retriever configured. Call set_retriever() first.")
                limit = node.params.get("limit", 5)
                modality = node.params.get("modality", "text")
                return self._retriever.search(
                    query=node_input,
                    query_modality=modality,
                    limit=limit
                )
            
            elif node.type == NodeType.GENERATE:
                generator = self.registry.get_generator(node.tool)
                constraints = node.params.get("constraints", {})
                return generator.generate(node_input, constraints)
            
            elif node.type == NodeType.FILTER:
                # Built-in filter: expects list, applies threshold
                threshold = node.params.get("threshold", 0.5)
                key = node.params.get("key", "score")
                if isinstance(node_input, list):
                    return [x for x in node_input if getattr(x, key, x.get(key, 0)) >= threshold]
                return node_input
            
            elif node.type == NodeType.CUSTOM:
                if node.tool not in self.custom_handlers:
                    raise ValueError(f"Custom handler '{node.tool}' not registered")
                handler = self.custom_handlers[node.tool]
                return handler(node_input, **node.params)
            
            else:
                raise ValueError(f"Unknown node type: {node.type}")
                
        except Exception as e:
            context.errors.append(f"Node {node.id}: {str(e)}")
            logger.error(f"Error in node {node.id}: {e}")
            raise
    
    def run(
        self,
        workflow_name: str,
        input_data: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Execute a registered workflow.
        
        Args:
            workflow_name: Name of registered workflow
            input_data: Initial input to the pipeline
            metadata: Optional metadata to include in context
            
        Returns:
            PipelineResult with output and execution details
        """
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")
        
        config = self.workflows[workflow_name]
        context = ExecutionContext(
            workflow_id=workflow_name,
            metadata=metadata or {}
        )
        
        start = datetime.now()
        
        try:
            # Get execution order
            execution_order = self._build_execution_order(config)
            
            # Execute each node
            for node in execution_order:
                output = self._execute_node(node, context, input_data)
                context.set_output(node.id, output)
            
            # Get final output
            final_output = context.get_input(config.output_node) if config.output_node else output
            
            duration = (datetime.now() - start).total_seconds() * 1000
            
            return PipelineResult(
                success=True,
                output=final_output,
                context=context,
                duration_ms=duration
            )
            
        except Exception as e:
            duration = (datetime.now() - start).total_seconds() * 1000
            logger.error(f"Workflow {workflow_name} failed: {e}")
            
            return PipelineResult(
                success=False,
                output=None,
                context=context,
                duration_ms=duration
            )
    
    def run_from_dict(
        self,
        workflow_dict: Dict[str, Any],
        input_data: Any
    ) -> PipelineResult:
        """
        Execute a workflow from a dictionary (e.g., loaded YAML).
        
        Useful for ad-hoc workflows without pre-registration.
        """
        config = WorkflowConfig.from_dict(workflow_dict)
        self.register_workflow(config)
        return self.run(config.name, input_data)
    
    def list_workflows(self) -> List[str]:
        """List all registered workflows."""
        return list(self.workflows.keys())
    
    def describe_workflow(self, name: str) -> Dict[str, Any]:
        """Get details about a workflow."""
        if name not in self.workflows:
            raise ValueError(f"Workflow '{name}' not found")
        
        config = self.workflows[name]
        return {
            "name": config.name,
            "description": config.description,
            "nodes": [
                {"id": n.id, "type": n.type.value, "tool": n.tool}
                for n in config.nodes
            ],
            "output_node": config.output_node
        }
