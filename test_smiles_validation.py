#!/usr/bin/env python3
"""
Quick test to verify SMILES validation behavior:
1. Invalid query "aaa" should return HTTP 400 (not 500)
2. Valid SMILES like caffeine should pass validation
"""
import sys

# Test the validate_smiles_query function directly
sys.path.insert(0, '.')

from bioflow.api.server import validate_smiles_query

print("=" * 60)
print("SMILES VALIDATION TESTS")
print("=" * 60)

# Test cases
test_cases = [
    # (query, expected_valid, description)
    ("aaa", False, "Random text 'aaa'"),
    ("hello world", False, "Plain text query"),
    ("aspirin", False, "Drug name (not SMILES)"),
    ("", False, "Empty string"),
    ("   ", False, "Whitespace only"),
    
    # Valid SMILES
    ("CCO", True, "Ethanol (simple)"),
    ("C", True, "Methane (simplest)"),
    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", True, "Caffeine"),
    ("CC(=O)OC1=CC=CC=C1C(=O)O", True, "Aspirin (SMILES)"),
    ("c1ccccc1", True, "Benzene (aromatic)"),
    ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", True, "Ibuprofen"),
    ("O=C(C)Oc1ccccc1C(=O)O", True, "Aspirin (variant)"),
]

passed = 0
failed = 0

for query, expected_valid, description in test_cases:
    result = validate_smiles_query(query)
    is_valid = result.get("is_valid_smiles", False)  # Correct key
    
    if is_valid == expected_valid:
        status = "✓ PASS"
        passed += 1
    else:
        status = "✗ FAIL"
        failed += 1
    
    print(f"\n{status}: {description}")
    print(f"  Query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
    print(f"  Expected valid: {expected_valid}, Got: {is_valid}")
    print(f"  Query type: {result.get('query_type', 'N/A')}")
    if result.get('warning'):
        print(f"  Warning: {result.get('warning')}")
    if result.get('mol_info'):
        mol = result['mol_info']
        print(f"  MW: {mol.get('molecular_weight', 'N/A'):.2f}, LogP: {mol.get('logp', 'N/A'):.2f}")

print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

# Exit with error code if any failed
sys.exit(0 if failed == 0 else 1)
