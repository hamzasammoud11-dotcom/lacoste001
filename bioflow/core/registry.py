"""
BioFlow Tool Registry
======================

Central registry for all biological tools in the BioFlow platform.
Supports encoders, predictors, generators, and misc tools.
"""

from typing import Dict, Type, Any, Optional, List
import logging

from bioflow.core.base import BioEncoder, BioPredictor, BioGenerator, BioTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry for all biological tools in the BioFlow platform.
    
    Features:
    - Register/unregister tools by name
    - Get tools with fallback to default
    - List all registered tools
    - Auto-discovery of tools from plugins directory
    
    Usage:
        >>> ToolRegistry.register_encoder("esm2", ESM2Encoder())
        >>> encoder = ToolRegistry.get_encoder("esm2")
    """
    
    _encoders: Dict[str, BioEncoder] = {}
    _predictors: Dict[str, BioPredictor] = {}
    _generators: Dict[str, BioGenerator] = {}
    _misc_tools: Dict[str, BioTool] = {}
    _default_encoder: Optional[str] = None
    _default_predictor: Optional[str] = None

    # ==================== ENCODERS ====================
    
    @classmethod
    def register_encoder(cls, name: str, encoder: BioEncoder, set_default: bool = False):
        """Register an encoder with optional default flag."""
        cls._encoders[name] = encoder
        if set_default or cls._default_encoder is None:
            cls._default_encoder = name
        logger.info(f"Registered encoder: {name} (dim={encoder.dimension})")

    @classmethod
    def unregister_encoder(cls, name: str):
        """Remove an encoder from registry."""
        if name in cls._encoders:
            del cls._encoders[name]
            if cls._default_encoder == name:
                cls._default_encoder = next(iter(cls._encoders), None)

    @classmethod
    def get_encoder(cls, name: str = None) -> BioEncoder:
        """Get encoder by name, or return default."""
        name = name or cls._default_encoder
        if name not in cls._encoders:
            available = list(cls._encoders.keys())
            raise ValueError(f"Encoder '{name}' not found. Available: {available}")
        return cls._encoders[name]

    # ==================== PREDICTORS ====================
    
    @classmethod
    def register_predictor(cls, name: str, predictor: BioPredictor, set_default: bool = False):
        """Register a predictor with optional default flag."""
        cls._predictors[name] = predictor
        if set_default or cls._default_predictor is None:
            cls._default_predictor = name
        logger.info(f"Registered predictor: {name}")

    @classmethod
    def unregister_predictor(cls, name: str):
        """Remove a predictor from registry."""
        if name in cls._predictors:
            del cls._predictors[name]
            if cls._default_predictor == name:
                cls._default_predictor = next(iter(cls._predictors), None)

    @classmethod
    def get_predictor(cls, name: str = None) -> BioPredictor:
        """Get predictor by name, or return default."""
        name = name or cls._default_predictor
        if name not in cls._predictors:
            available = list(cls._predictors.keys())
            raise ValueError(f"Predictor '{name}' not found. Available: {available}")
        return cls._predictors[name]

    # ==================== GENERATORS ====================
    
    @classmethod
    def register_generator(cls, name: str, generator: BioGenerator):
        """Register a generator."""
        cls._generators[name] = generator
        logger.info(f"Registered generator: {name}")

    @classmethod
    def get_generator(cls, name: str) -> BioGenerator:
        """Get generator by name."""
        if name not in cls._generators:
            available = list(cls._generators.keys())
            raise ValueError(f"Generator '{name}' not found. Available: {available}")
        return cls._generators[name]

    # ==================== MISC TOOLS ====================
    
    @classmethod
    def register_tool(cls, name: str, tool: BioTool):
        """Register a miscellaneous tool."""
        cls._misc_tools[name] = tool
        logger.info(f"Registered tool: {name}")

    @classmethod
    def get_tool(cls, name: str) -> BioTool:
        """Get misc tool by name."""
        if name not in cls._misc_tools:
            available = list(cls._misc_tools.keys())
            raise ValueError(f"Tool '{name}' not found. Available: {available}")
        return cls._misc_tools[name]

    # ==================== UTILITIES ====================
    
    @classmethod
    def list_tools(cls) -> Dict[str, List[str]]:
        """List all registered tools by category."""
        return {
            "encoders": list(cls._encoders.keys()),
            "predictors": list(cls._predictors.keys()),
            "generators": list(cls._generators.keys()),
            "tools": list(cls._misc_tools.keys())
        }
    
    @classmethod
    def clear(cls):
        """Clear all registered tools (useful for testing)."""
        cls._encoders.clear()
        cls._predictors.clear()
        cls._generators.clear()
        cls._misc_tools.clear()
        cls._default_encoder = None
        cls._default_predictor = None
    
    @classmethod
    def summary(cls) -> str:
        """Get human-readable summary of registered tools."""
        tools = cls.list_tools()
        lines = ["BioFlow Tool Registry:"]
        for category, names in tools.items():
            lines.append(f"  {category}: {', '.join(names) if names else '(none)'}")
        return "\n".join(lines)
