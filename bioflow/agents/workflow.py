"""
Workflow Engine
===============

Chains agents together for multi-step discovery workflows.

Features:
- Pipeline-based workflow execution
- Context passing between agents
- Parallel and sequential execution modes
- Error handling and recovery
"""

import logging
import time
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from bioflow.agents.base import (
    BaseAgent, 
    AgentMessage, 
    AgentContext, 
    AgentType,
)
from bioflow.agents.generator import GeneratorAgent
from bioflow.agents.validator import ValidatorAgent
from bioflow.agents.ranker import RankerAgent

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    name: str
    agent: BaseAgent
    input_transform: Optional[Callable[[Any, AgentContext], Any]] = None
    output_transform: Optional[Callable[[AgentMessage, AgentContext], Any]] = None
    condition: Optional[Callable[[AgentContext], bool]] = None
    on_error: str = "stop"  # "stop", "continue", "retry"
    max_retries: int = 3


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""
    status: WorkflowStatus
    steps_completed: int
    total_steps: int
    outputs: Dict[str, Any]
    context: AgentContext
    errors: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "steps_completed": self.steps_completed,
            "total_steps": self.total_steps,
            "outputs": self.outputs,
            "context": self.context.to_dict(),
            "errors": self.errors,
            "execution_time_ms": self.execution_time_ms,
        }


class WorkflowEngine:
    """
    Engine for executing multi-step discovery workflows.
    
    A workflow consists of steps that are executed in sequence,
    with each step's output available to subsequent steps via context.
    
    Example:
        >>> engine = WorkflowEngine()
        >>> engine.add_step("generate", GeneratorAgent())
        >>> engine.add_step("validate", ValidatorAgent())
        >>> engine.add_step("rank", RankerAgent())
        >>> result = engine.run("Design a kinase inhibitor")
    """
    
    def __init__(
        self,
        name: str = "DiscoveryWorkflow",
        max_parallel_workers: int = 4,
    ):
        """
        Initialize workflow engine.
        
        Args:
            name: Workflow name
            max_parallel_workers: Max workers for parallel steps
        """
        self.name = name
        self.steps: List[WorkflowStep] = []
        self.max_parallel_workers = max_parallel_workers
        self._status = WorkflowStatus.PENDING
        self._hooks: Dict[str, List[Callable]] = {
            "before_step": [],
            "after_step": [],
            "on_error": [],
            "on_complete": [],
        }
    
    def add_step(
        self,
        name: str,
        agent: BaseAgent,
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
        condition: Optional[Callable] = None,
        on_error: str = "stop",
    ) -> "WorkflowEngine":
        """
        Add a step to the workflow.
        
        Args:
            name: Step name (used for output keys)
            agent: Agent to execute
            input_transform: Transform context to agent input
            output_transform: Transform agent output for context
            condition: Condition to check before running step
            on_error: Error handling strategy
            
        Returns:
            Self for chaining
        """
        self.steps.append(WorkflowStep(
            name=name,
            agent=agent,
            input_transform=input_transform,
            output_transform=output_transform,
            condition=condition,
            on_error=on_error,
        ))
        logger.info(f"Added step '{name}' ({agent.agent_type.value}) to workflow")
        return self
    
    def add_hook(
        self,
        event: str,
        callback: Callable,
    ) -> "WorkflowEngine":
        """
        Add a callback hook for workflow events.
        
        Events:
        - before_step: (step_name, context) -> None
        - after_step: (step_name, result, context) -> None
        - on_error: (step_name, error, context) -> None
        - on_complete: (result) -> None
        """
        if event in self._hooks:
            self._hooks[event].append(callback)
        return self
    
    def run(
        self,
        query: str,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        """
        Execute the workflow.
        
        Args:
            query: Initial query/prompt
            initial_context: Optional initial context values
            
        Returns:
            WorkflowResult with all outputs
        """
        start_time = time.time()
        self._status = WorkflowStatus.RUNNING
        
        # Initialize context
        context = AgentContext(
            query=query,
            metadata=initial_context or {},
        )
        
        outputs = {}
        errors = []
        steps_completed = 0
        
        for step in self.steps:
            # Check condition
            if step.condition is not None:
                try:
                    if not step.condition(context):
                        logger.info(f"Step '{step.name}' skipped (condition not met)")
                        continue
                except Exception as e:
                    logger.warning(f"Condition check failed for '{step.name}': {e}")
            
            # Run before hooks
            for hook in self._hooks["before_step"]:
                try:
                    hook(step.name, context)
                except Exception as e:
                    logger.warning(f"Before hook failed: {e}")
            
            # Prepare input
            if step.input_transform is not None:
                try:
                    input_data = step.input_transform(context.get_last_output(), context)
                except Exception as e:
                    logger.error(f"Input transform failed for '{step.name}': {e}")
                    input_data = context.get_last_output() or query
            else:
                input_data = context.get_last_output() or query
            
            # Execute step
            try:
                logger.info(f"Executing step '{step.name}'...")
                result = step.agent(input_data, context)
                
                # Transform output
                if step.output_transform is not None:
                    output = step.output_transform(result, context)
                else:
                    output = result.content
                
                # Store output
                outputs[step.name] = output
                context.add_message(result)
                
                # Update context based on agent type
                self._update_context(step.agent.agent_type, result, context)
                
                steps_completed += 1
                
                # Run after hooks
                for hook in self._hooks["after_step"]:
                    try:
                        hook(step.name, result, context)
                    except Exception as e:
                        logger.warning(f"After hook failed: {e}")
                
            except Exception as e:
                error_msg = f"Step '{step.name}' failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                
                # Run error hooks
                for hook in self._hooks["on_error"]:
                    try:
                        hook(step.name, e, context)
                    except:
                        pass
                
                # Handle error based on strategy
                if step.on_error == "stop":
                    self._status = WorkflowStatus.FAILED
                    break
                elif step.on_error == "continue":
                    outputs[step.name] = {"error": str(e)}
                    steps_completed += 1
                # "retry" would need additional logic
        
        # Finalize
        execution_time = (time.time() - start_time) * 1000
        
        if self._status != WorkflowStatus.FAILED:
            self._status = WorkflowStatus.COMPLETED
        
        result = WorkflowResult(
            status=self._status,
            steps_completed=steps_completed,
            total_steps=len(self.steps),
            outputs=outputs,
            context=context,
            errors=errors,
            execution_time_ms=execution_time,
        )
        
        # Run completion hooks
        for hook in self._hooks["on_complete"]:
            try:
                hook(result)
            except Exception as e:
                logger.warning(f"Completion hook failed: {e}")
        
        logger.info(f"Workflow completed: {steps_completed}/{len(self.steps)} steps, {execution_time:.1f}ms")
        return result
    
    def _update_context(
        self,
        agent_type: AgentType,
        result: AgentMessage,
        context: AgentContext,
    ) -> None:
        """Update context based on agent output."""
        if agent_type == AgentType.GENERATOR:
            # Add generated molecules to candidates
            if isinstance(result.content, list):
                for mol in result.content:
                    if isinstance(mol, dict) and "smiles" in mol:
                        context.candidates.append(mol)
        
        elif agent_type == AgentType.VALIDATOR:
            # Merge validation results into candidates
            if isinstance(result.content, list):
                validation_by_smiles = {
                    v["smiles"]: v 
                    for v in result.content 
                    if isinstance(v, dict) and "smiles" in v
                }
                for cand in context.candidates:
                    smiles = cand.get("smiles")
                    if smiles in validation_by_smiles:
                        cand["validation"] = validation_by_smiles[smiles]
                        cand["validation_score"] = validation_by_smiles[smiles].get("score", 0)
        
        elif agent_type == AgentType.RANKER:
            # Update candidates with ranking info
            if isinstance(result.content, list):
                ranked_by_smiles = {
                    r["smiles"]: r
                    for r in result.content
                    if isinstance(r, dict) and "smiles" in r
                }
                for cand in context.candidates:
                    smiles = cand.get("smiles")
                    if smiles in ranked_by_smiles:
                        cand["rank"] = ranked_by_smiles[smiles].get("rank")
                        cand["final_score"] = ranked_by_smiles[smiles].get("final_score")
    
    def clear(self) -> None:
        """Clear all steps."""
        self.steps = []
        self._status = WorkflowStatus.PENDING


class DiscoveryWorkflow:
    """
    Pre-configured workflow for drug discovery.
    
    Pipeline:
    1. Generate molecules from text prompt
    2. Validate ADMET properties
    3. Rank candidates
    
    Example:
        >>> workflow = DiscoveryWorkflow()
        >>> result = workflow.run("Design an EGFR inhibitor with good oral bioavailability")
        >>> print(result.outputs["rank"][0])  # Top candidate
    """
    
    def __init__(
        self,
        num_candidates: int = 10,
        top_k: int = 5,
    ):
        """
        Initialize discovery workflow.
        
        Args:
            num_candidates: Number of molecules to generate
            top_k: Number of top candidates to return
        """
        self.num_candidates = num_candidates
        self.top_k = top_k
        
        # Create agents
        self.generator = GeneratorAgent(num_samples=num_candidates)
        self.validator = ValidatorAgent()
        self.ranker = RankerAgent()
        
        # Build engine
        self.engine = WorkflowEngine(name="DrugDiscovery")
        
        # Step 1: Generate
        self.engine.add_step(
            name="generate",
            agent=self.generator,
        )
        
        # Step 2: Validate
        self.engine.add_step(
            name="validate",
            agent=self.validator,
            input_transform=lambda output, ctx: [
                m["smiles"] for m in output if isinstance(m, dict) and "smiles" in m
            ],
        )
        
        # Step 3: Rank
        self.engine.add_step(
            name="rank",
            agent=self.ranker,
            input_transform=self._prepare_ranking_input,
        )
    
    def _prepare_ranking_input(
        self,
        output: Any,
        context: AgentContext,
    ) -> Dict[str, Any]:
        """Prepare input for ranking from context."""
        # Merge generation and validation data
        candidates = []
        
        # Get generation data
        gen_output = context.history[0].content if context.history else []
        gen_by_smiles = {
            m["smiles"]: m
            for m in gen_output
            if isinstance(m, dict) and "smiles" in m
        }
        
        # Get validation data
        val_output = output if isinstance(output, list) else []
        
        for val in val_output:
            if not isinstance(val, dict):
                continue
            smiles = val.get("smiles")
            if smiles:
                candidate = {
                    "smiles": smiles,
                    "validation_score": val.get("score", 0),
                    "qed": next(
                        (p["value"] for p in val.get("properties", []) 
                         if p["name"] == "QED"),
                        0.5
                    ),
                }
                # Add generation confidence
                if smiles in gen_by_smiles:
                    candidate["confidence"] = gen_by_smiles[smiles].get("confidence", 0.5)
                candidates.append(candidate)
        
        return {
            "candidates": candidates,
            "top_k": self.top_k,
        }
    
    def run(
        self,
        query: str,
        **kwargs,
    ) -> WorkflowResult:
        """
        Run the discovery workflow.
        
        Args:
            query: Text description of desired molecule
            **kwargs: Additional context
            
        Returns:
            WorkflowResult with generated, validated, and ranked candidates
        """
        return self.engine.run(query, initial_context=kwargs)
    
    def get_top_candidates(
        self,
        result: WorkflowResult,
    ) -> List[Dict[str, Any]]:
        """
        Extract top candidates from workflow result.
        
        Args:
            result: Workflow result
            
        Returns:
            List of top-ranked candidates with all metadata
        """
        ranked = result.outputs.get("rank", [])
        
        # Enrich with full context
        enriched = []
        for r in ranked[:self.top_k]:
            smiles = r.get("smiles")
            candidate_ctx = next(
                (c for c in result.context.candidates if c.get("smiles") == smiles),
                {}
            )

            validation_ui = self._validation_to_ui(candidate_ctx.get("validation", {}))
            name = (
                candidate_ctx.get("name")
                or candidate_ctx.get("title")
                or candidate_ctx.get("label")
                or r.get("name")
                or f"Candidate {r.get('rank') or 0}"
            )
            score = r.get("final_score", r.get("score", 0.0))
            enriched.append({
                **r,
                # UI-friendly fields
                "name": name,
                "score": score,
                "validation": validation_ui,
                # Keep full context for debugging / extended UI panels
                "generation": candidate_ctx,
            })
        
        return enriched

    def _validation_to_ui(self, validation: Any) -> Dict[str, Any]:
        """
        Normalize validator output to the UI-friendly shape:
        { is_valid: bool, checks: {name: bool}, properties: {name: number} }.
        """
        if not isinstance(validation, dict):
            return {"is_valid": True, "checks": {}, "properties": {}}

        # Determine validity
        if "is_valid" in validation:
            is_valid = bool(validation.get("is_valid"))
        else:
            status = str(validation.get("status", "passed")).lower()
            is_valid = status in ("passed", "ok", "success", "true")

        checks: Dict[str, bool] = {}
        properties: Dict[str, float] = {}

        props = validation.get("properties", [])
        if isinstance(props, list):
            for p in props:
                if not isinstance(p, dict):
                    continue
                name = str(p.get("name") or "").strip()
                if not name:
                    continue
                checks[name] = bool(p.get("passed", True))
                value = p.get("value")
                if isinstance(value, (int, float)):
                    properties[name] = float(value)

        alerts = validation.get("alerts", [])
        if isinstance(alerts, list):
            checks["no_alerts"] = len(alerts) == 0

        return {"is_valid": is_valid, "checks": checks, "properties": properties}
