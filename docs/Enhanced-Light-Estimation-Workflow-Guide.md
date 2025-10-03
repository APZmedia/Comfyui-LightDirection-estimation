# Enhanced Light Estimation Workflow Guide

## Overview

This guide explains the new **Enhanced Light Estimation Workflow** that combines traditional normal map analysis with advanced IRE shadow analysis for comprehensive lighting assessment.

## Workflow Structure

The enhanced workflow (`TEST-Enhanced-Light-Estimation-v3.json`) provides:

### 🔄 **Dual Analysis Approach**
1. **Enhanced Light Estimator**: Combined normal map + IRE analysis
2. **Standalone IRE Shadow Analyzer**: Pure IRE shadow analysis for comparison

### 📊 **Comprehensive Outputs**
- Traditional light direction analysis
- IRE-based shadow characterization  
- False color visualizations
- Shadow transition analysis
- Combined confidence scoring

## Workflow Components

### 1. **Input Processing Section**
```
LoadImage → ImageResizeKJv2 → RMBG → LBM_DepthNormal + rgb2x
```

**Purpose**: Prepare input images and generate required data
- **LoadImage**: Load source image
- **ImageResizeKJv2**: Resize to standard dimensions (1280x1280)
- **RMBG**: Background removal for clean analysis
- **LBM_DepthNormal**: Generate normal map from depth
- **rgb2x**: Convert to irradiance for luma analysis

### 2. **Enhanced Light Estimator (Combined Analysis)**
```
EnhancedLightEstimator Node
```

**Inputs**:
- `normal_map`: From LBM_DepthNormal
- `luma_image`: From ImageResizeKJv2  
- `exclusion_mask`: From RMBG

**Key Parameters**:
- `luma_threshold`: 0.5 (standard)
- `curve_type`: "s_curve" (enhanced mid-tones)
- `shadow_ire_threshold`: 20.0 (IRE units)
- `transition_sensitivity`: 0.1 (balanced)
- `analysis_method`: "combined" (both normal + IRE)
- `ire_analysis_weight`: 0.5 (equal weighting)

**Outputs**:
- **Direction Analysis**: X/Y direction, combined direction
- **Quality Analysis**: Hard/soft index, confidence scores
- **IRE Analysis**: Shadow character, transition quality
- **Visualizations**: Debug masks, false color, threshold previews
- **Combined Results**: Final shadow character, light quality

### 3. **Standalone IRE Shadow Analyzer**
```
IREShadowAnalyzer Node
```

**Purpose**: Pure IRE-based shadow analysis for comparison

**Parameters**:
- `shadow_ire_threshold`: 20.0 IRE
- `transition_sensitivity`: 0.1
- `analysis_mode`: "full"

**Outputs**:
- False color IRE visualization
- Shadow masks (soft/hard)
- IRE legend
- Shadow characterizations
- Quantitative metrics

### 4. **Results & Visualizations Section**

#### **Text Outputs**
- X/Y Direction results
- Combined direction
- Hard/Soft index
- Shadow character (IRE-based)
- Transition quality
- Final combined results

#### **Image Outputs**
- Debug mask (lit areas)
- Lit normals visualization
- Cluster delta chart
- Threshold previews (X/Y)
- False color IRE visualization
- Shadow masks (soft/hard)
- IRE legend

## Key Features

### 🎯 **Enhanced Analysis Capabilities**

#### **1. Combined Normal Map + IRE Analysis**
- **Geometric Analysis**: Surface normal distribution
- **Luminance Analysis**: IRE-based shadow transitions
- **Weighted Combination**: User-controllable weighting
- **Consistency Checking**: Validates agreement between methods

#### **2. IRE Shadow Analysis**
- **False Color Visualization**: 7-zone IRE color mapping
- **Transition Analysis**: Gradient-based soft/hard detection
- **Quantitative Metrics**: Ratios, percentages, confidence scores
- **Professional Standards**: Broadcast/video industry IRE standards

#### **3. Advanced Visualizations**
- **Debug Masks**: Show which areas are analyzed
- **Threshold Previews**: Visualize classification zones
- **False Color IRE**: Professional exposure analysis
- **Shadow Masks**: Separate soft/hard shadow areas
- **IRE Legend**: Color mapping reference

### 📈 **Analysis Parameters**

#### **Enhanced Light Estimator Parameters**
```json
{
  "luma_threshold": 0.5,           // Luma masking threshold
  "curve_type": "s_curve",         // Luma processing curve
  "x_threshold": 0.4,              // X direction sensitivity
  "y_threshold_upper": 0.1,       // Y direction (upper)
  "y_threshold_lower": 0.1,       // Y direction (lower)
  "central_threshold": 0.3,       // Central lighting detection
  "hard_light_threshold": 0.15,   // Hard light detection
  "soft_light_threshold": 0.35,   // Soft light detection
  "shadow_ire_threshold": 20.0,   // IRE shadow threshold
  "transition_sensitivity": 0.1,   // IRE transition sensitivity
  "analysis_method": "combined",   // Analysis mode
  "ire_analysis_weight": 0.5      // IRE analysis weighting
}
```

#### **IRE Shadow Analyzer Parameters**
```json
{
  "shadow_ire_threshold": 20.0,   // IRE threshold for shadows
  "transition_sensitivity": 0.1,   // Transition detection sensitivity
  "analysis_mode": "full"         // Analysis scope
}
```

## Usage Instructions

### **Step 1: Load and Prepare Image**
1. Use `LoadImage` to load your source image
2. `ImageResizeKJv2` resizes to 1280x1280 for consistent analysis
3. `RMBG` removes background for clean analysis

### **Step 2: Generate Analysis Data**
1. `LBM_DepthNormal` creates normal map from depth information
2. `rgb2x` converts to irradiance for luma analysis

### **Step 3: Run Enhanced Analysis**
1. **Enhanced Light Estimator** provides combined analysis
2. **IRE Shadow Analyzer** provides standalone IRE analysis
3. Compare results for validation

### **Step 4: Interpret Results**

#### **Direction Analysis**
- **X Direction**: Left/Center/Right lighting
- **Y Direction**: Above/Center/Below lighting  
- **Combined**: Full directional classification

#### **Quality Analysis**
- **Hard/Soft Index**: 0.0 (hard) to 1.0 (soft)
- **Confidence Scores**: Analysis reliability
- **Spread Value**: Normal distribution spread

#### **IRE Shadow Analysis**
- **Shadow Character**: Very Soft/Soft/Medium/Hard
- **Transition Quality**: Very Gradual to Very Sharp
- **IRE Range**: Actual IRE values in image
- **Shadow Coverage**: Percentage of shadow areas

#### **Combined Results**
- **Final Shadow Character**: Weighted combination
- **Final Light Quality**: Integrated assessment
- **Combined Confidence**: Overall reliability
- **Analysis Consistency**: Agreement between methods

## Advanced Usage

### **Parameter Tuning**

#### **For Soft Shadow Detection**
```json
{
  "shadow_ire_threshold": 15.0,    // Lower threshold
  "transition_sensitivity": 0.05,  // More sensitive
  "ire_analysis_weight": 0.7       // Favor IRE analysis
}
```

#### **For Hard Shadow Detection**
```json
{
  "shadow_ire_threshold": 25.0,    // Higher threshold
  "transition_sensitivity": 0.2,  // Less sensitive
  "ire_analysis_weight": 0.3       // Favor normal analysis
}
```

#### **For Balanced Analysis**
```json
{
  "shadow_ire_threshold": 20.0,    // Standard threshold
  "transition_sensitivity": 0.1,    // Balanced sensitivity
  "ire_analysis_weight": 0.5       // Equal weighting
}
```

### **Workflow Customization**

#### **Add IRE Shadow Comparison**
- Use `IREShadowComparison` node to compare two images
- Analyze shadow differences between lighting setups
- Validate lighting changes

#### **Modular Analysis**
- Use individual nodes for specific analysis
- `LightImageProcessor` for image preparation
- `LightDistributionAnalyzer` for distribution analysis
- `IREShadowAnalyzer` for pure IRE analysis

## Troubleshooting

### **Common Issues**

#### **"No shadows detected"**
- Lower `shadow_ire_threshold` (try 15.0)
- Check image contrast and exposure
- Verify IRE range in image

#### **"Incorrect shadow classification"**
- Adjust `transition_sensitivity` (0.05-0.3)
- Check `ire_analysis_weight` balance
- Verify normal map quality

#### **"Poor analysis consistency"**
- Check image quality and lighting
- Adjust parameter sensitivity
- Use higher resolution images

### **Performance Optimization**

- Use appropriate image resolution (512x512 to 1024x1024)
- Process images in batches for efficiency
- Monitor GPU memory usage for large images
- Use exclusion masks to focus analysis areas

## Professional Applications

### **Cinematography**
- Lighting setup validation
- Shadow quality assessment
- Exposure analysis using IRE standards
- Professional workflow integration

### **Photography**
- Portrait lighting analysis
- Shadow correction workflows
- Exposure zone analysis
- Professional retouching

### **Computer Graphics**
- Lighting validation for 3D renders
- Shadow quality assessment
- Professional lighting workflows
- Industry-standard analysis

This enhanced workflow provides comprehensive lighting analysis suitable for professional applications while maintaining ease of use for general purposes.
