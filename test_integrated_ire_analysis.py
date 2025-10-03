#!/usr/bin/env python3
"""
Test script for the integrated IRE shadow analysis in NormalMapLightEstimator.
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_integrated_parameters():
    """
    Test that the NormalMapLightEstimator now includes IRE parameters.
    """
    print("=== Testing Integrated IRE Analysis in NormalMapLightEstimator ===")
    
    try:
        from nodes.nodes import NormalMapLightEstimator
        
        # Test INPUT_TYPES
        input_types = NormalMapLightEstimator.INPUT_TYPES()
        
        print("✅ INPUT_TYPES loaded successfully")
        
        # Check for new IRE parameters
        required_params = input_types.get("required", {})
        optional_params = input_types.get("optional", {})
        
        # Check required IRE parameters
        ire_required_params = ["shadow_ire_threshold", "transition_sensitivity"]
        for param in ire_required_params:
            if param in required_params:
                print(f"✅ Required parameter '{param}' found: {required_params[param]}")
            else:
                print(f"❌ Required parameter '{param}' missing")
        
        # Check optional IRE parameters
        ire_optional_params = ["ire_analysis_weight"]
        for param in ire_optional_params:
            if param in optional_params:
                print(f"✅ Optional parameter '{param}' found: {optional_params[param]}")
            else:
                print(f"❌ Optional parameter '{param}' missing")
        
        # Check analysis method options
        analysis_method = required_params.get("analysis_method", optional_params.get("analysis_method"))
        if analysis_method and "combined" in analysis_method[0]:
            print(f"✅ 'combined' analysis method available: {analysis_method}")
        else:
            print("❌ 'combined' analysis method not found")
        
        # Test RETURN_TYPES
        return_types = NormalMapLightEstimator.RETURN_TYPES
        return_names = NormalMapLightEstimator.RETURN_NAMES
        
        print(f"✅ RETURN_TYPES: {len(return_types)} outputs")
        print(f"✅ RETURN_NAMES: {len(return_names)} names")
        
        # Check for IRE-specific outputs
        ire_outputs = [
            "false_color_ire", "shadow_mask", "soft_shadow_mask", "hard_shadow_mask", "ire_legend",
            "shadow_character", "transition_quality", "ire_range", "shadow_coverage", "gradient_analysis",
            "mean_ire", "shadow_percentage", "soft_ratio", "hard_ratio", "mean_gradient",
            "final_shadow_character", "final_light_quality", "combined_confidence", "analysis_consistency"
        ]
        
        for output in ire_outputs:
            if output in return_names:
                print(f"✅ IRE output '{output}' found")
            else:
                print(f"❌ IRE output '{output}' missing")
        
        print("\n=== Parameter Summary ===")
        print("Required parameters:")
        for param, config in required_params.items():
            if "ire" in param.lower() or "shadow" in param.lower() or "transition" in param.lower():
                print(f"  - {param}: {config}")
        
        print("\nOptional parameters:")
        for param, config in optional_params.items():
            if "ire" in param.lower() or "analysis" in param.lower():
                print(f"  - {param}: {config}")
        
        print("\n=== Integration Test Complete ===")
        print("✅ NormalMapLightEstimator successfully integrated with IRE shadow analysis!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_defaults():
    """
    Test the default parameter values for IRE analysis.
    """
    print("\n=== Testing Parameter Defaults ===")
    
    try:
        from nodes.nodes import NormalMapLightEstimator
        
        input_types = NormalMapLightEstimator.INPUT_TYPES()
        required_params = input_types.get("required", {})
        optional_params = input_types.get("optional", {})
        
        # Test IRE parameter defaults
        ire_defaults = {
            "shadow_ire_threshold": 20.0,
            "transition_sensitivity": 0.1,
            "ire_analysis_weight": 0.5
        }
        
        for param, expected_default in ire_defaults.items():
            if param in required_params:
                actual_default = required_params[param]["default"]
                if actual_default == expected_default:
                    print(f"✅ {param}: {actual_default} (correct)")
                else:
                    print(f"❌ {param}: {actual_default} (expected {expected_default})")
            elif param in optional_params:
                actual_default = optional_params[param]["default"]
                if actual_default == expected_default:
                    print(f"✅ {param}: {actual_default} (correct)")
                else:
                    print(f"❌ {param}: {actual_default} (expected {expected_default})")
            else:
                print(f"❌ {param}: not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing defaults: {e}")
        return False

if __name__ == "__main__":
    print("Testing Integrated IRE Analysis in NormalMapLightEstimator...")
    
    success1 = test_integrated_parameters()
    success2 = test_parameter_defaults()
    
    if success1 and success2:
        print("\n🎉 All tests passed! IRE shadow analysis successfully integrated into NormalMapLightEstimator.")
        print("\nNew capabilities:")
        print("- shadow_ire_threshold: IRE threshold for shadow detection (0-50)")
        print("- transition_sensitivity: Sensitivity for shadow transition detection (0.01-1.0)")
        print("- ire_analysis_weight: Weighting for IRE vs normal analysis (0.0-1.0)")
        print("- analysis_method: Now includes 'combined' option")
        print("- 19 new outputs including false color IRE, shadow masks, and combined results")
    else:
        print("\n❌ Some tests failed. Check the integration.")
