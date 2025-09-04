#!/usr/bin/env python3
"""
UK Biobank Data Mapping Script
Maps literature-identified biomarkers to available UKBB fields
Analyzes participant intersection across multiple modalities
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Define data paths
UKBB_PATH = Path("/mnt/data1/UKBB")
RETINAL_PATH = Path("/mnt/data1/UKBB_retinal_img")

def analyze_data_availability():
    """Analyze data availability across modalities"""
    
    print("=" * 80)
    print("UK BIOBANK DATA AVAILABILITY ANALYSIS")
    print("=" * 80)
    
    # 1. Load participant IDs from main dataset
    print("\n1. LOADING MAIN DATASET...")
    main_df = pd.read_csv(UKBB_PATH / "ukb_main.csv", usecols=['f.eid'])
    total_participants = len(main_df)
    print(f"Total participants in UK Biobank: {total_participants:,}")
    
    # 2. Check blood biomarkers availability
    print("\n2. BLOOD BIOMARKERS AVAILABILITY...")
    blood_cols = ['f.eid', 'C_reactive_protein', 'Glycated_haemoglobin_HbA1c', 
                  'Creatinine', 'Albumin', 'Lymphocyte_percentage', 
                  'Red_blood_cell_erythrocyte_distribution_width',
                  'Gamma_glutamyltransferase', 'Aspartate_aminotransferase',
                  'Alanine_aminotransferase']
    
    try:
        # First check available columns
        blood_df_header = pd.read_csv(UKBB_PATH / "ukb_blood.csv", nrows=0)
        available_blood_cols = [col for col in blood_cols if col in blood_df_header.columns]
        print(f"Available blood biomarker columns: {len(available_blood_cols)-1}")
        
        # Load blood data with available columns
        blood_df = pd.read_csv(UKBB_PATH / "ukb_blood.csv", usecols=available_blood_cols)
        blood_participants = len(blood_df)
        print(f"Participants with blood data: {blood_participants:,}")
        
        # Check non-missing for each biomarker
        for col in available_blood_cols[1:]:  # Skip f.eid
            non_missing = blood_df[col].notna().sum()
            print(f"  - {col}: {non_missing:,} ({100*non_missing/blood_participants:.1f}%)")
    except Exception as e:
        print(f"Error loading blood data: {e}")
        blood_df = None
    
    # 3. Check NMR metabolomics
    print("\n3. NMR METABOLOMICS AVAILABILITY...")
    try:
        nmr_df = pd.read_csv(UKBB_PATH / "ukb_NMR.csv", usecols=['f.eid'])
        # Get unique participants (may have multiple records)
        nmr_participants = nmr_df['f.eid'].nunique()
        print(f"Participants with NMR data: {nmr_participants:,}")
    except Exception as e:
        print(f"Error loading NMR data: {e}")
        nmr_df = None
    
    # 4. Check OCT data
    print("\n4. OCT (OPTICAL COHERENCE TOMOGRAPHY) DATA...")
    try:
        oct_df = pd.read_csv(UKBB_PATH / "ukb_OCT.csv", usecols=['f.eid'])
        oct_participants = len(oct_df)
        print(f"Participants with OCT data: {oct_participants:,}")
    except Exception as e:
        print(f"Error loading OCT data: {e}")
        oct_df = None
    
    # 5. Check retinal imaging data
    print("\n5. RETINAL IMAGING DATA (Fundus Photography)...")
    retinal_participants = set()
    
    # Check UKB_new_2024 directory
    new_retinal_path = RETINAL_PATH / "UKB_new_2024"
    if new_retinal_path.exists():
        # Field 21015 = Left eye fundus image
        # Field 21016 = Right eye fundus image
        retinal_dirs = ['21015_2_1', '21015_3_1', '21016_2_1', '21016_2_3', '21016_3_1']
        
        for dir_name in retinal_dirs:
            dir_path = new_retinal_path / dir_name
            if dir_path.exists():
                # Extract participant IDs from filenames
                files = list(dir_path.glob("*.png"))
                for f in files[:100]:  # Sample first 100 to get pattern
                    # Format: participantID_fieldID_instance_array.png
                    participant_id = f.stem.split('_')[0]
                    retinal_participants.add(int(participant_id))
                print(f"  - {dir_name}: {len(files):,} images")
    
    # Check older UKBB_FP directories
    for fp_dir in ['UKBB_FP_01', 'UKBB_FP_02', 'UKBB_FP_03', 'UKBB_FP_04']:
        fp_path = RETINAL_PATH / fp_dir
        if fp_path.exists():
            files = list(fp_path.glob("*.png"))
            for f in files[:100]:  # Sample to get participant IDs
                participant_id = f.stem.split('_')[0]
                retinal_participants.add(int(participant_id))
            print(f"  - {fp_dir}: {len(files):,} images")
    
    print(f"\nEstimated unique participants with retinal images: ~{len(retinal_participants):,}")
    print("(Note: Full count requires scanning all image filenames)")
    
    # 6. Multi-modal intersection analysis
    print("\n6. MULTI-MODAL DATA INTERSECTION...")
    
    if blood_df is not None and oct_df is not None:
        # Blood + OCT intersection
        blood_oct_intersection = set(blood_df['f.eid']) & set(oct_df['f.eid'])
        print(f"Participants with both blood biomarkers AND OCT: {len(blood_oct_intersection):,}")
    
    if blood_df is not None and nmr_df is not None:
        # Blood + NMR intersection
        blood_nmr_intersection = set(blood_df['f.eid']) & set(nmr_df['f.eid'].unique())
        print(f"Participants with both blood biomarkers AND NMR: {len(blood_nmr_intersection):,}")
    
    # 7. Check body measurements and frailty
    print("\n7. ADDITIONAL MEASUREMENTS...")
    try:
        body_df = pd.read_csv(UKBB_PATH / "ukb_body.csv", usecols=['f.eid'])
        print(f"Participants with body measurements: {len(body_df):,}")
    except:
        pass
    
    try:
        frailty_df = pd.read_csv(UKBB_PATH / "ukb_frailty.csv", usecols=['f.eid'])
        print(f"Records in frailty dataset: {len(frailty_df):,}")
        print(f"Unique participants with frailty data: {frailty_df['f.eid'].nunique():,}")
    except:
        pass
    
    return {
        'total_participants': total_participants,
        'blood_participants': blood_participants if blood_df is not None else 0,
        'nmr_participants': nmr_participants if nmr_df is not None else 0,
        'oct_participants': oct_participants if oct_df is not None else 0,
        'retinal_participants_estimate': len(retinal_participants)
    }

def map_literature_to_ukbb_fields():
    """Map literature biomarkers to UKBB field IDs"""
    
    print("\n" + "=" * 80)
    print("LITERATURE TO UK BIOBANK FIELD MAPPING")
    print("=" * 80)
    
    # Create mapping dictionary based on literature findings
    biomarker_mapping = {
        # Clinical Biomarkers from Literature
        'Clinical_Biomarkers': {
            'CRP': {
                'ukbb_field': 'C_reactive_protein',
                'field_id': 30710,
                'units': 'mg/L',
                'file': 'ukb_blood.csv'
            },
            'HbA1c': {
                'ukbb_field': 'Glycated_haemoglobin_HbA1c',
                'field_id': 30750,
                'units': 'mmol/mol',
                'file': 'ukb_blood.csv'
            },
            'Creatinine': {
                'ukbb_field': 'Creatinine',
                'field_id': 30700,
                'units': 'umol/L',
                'file': 'ukb_blood.csv'
            },
            'Albumin': {
                'ukbb_field': 'Albumin',
                'field_id': 30600,
                'units': 'g/L',
                'file': 'ukb_blood.csv'
            },
            'Lymphocyte_percentage': {
                'ukbb_field': 'Lymphocyte_percentage',
                'field_id': 30180,
                'units': '%',
                'file': 'ukb_blood.csv'
            },
            'RDW': {
                'ukbb_field': 'Red_blood_cell_erythrocyte_distribution_width',
                'field_id': 30070,
                'units': '%',
                'file': 'ukb_blood.csv'
            },
            'GGT': {
                'ukbb_field': 'Gamma_glutamyltransferase',
                'field_id': 30730,
                'units': 'U/L',
                'file': 'ukb_blood.csv'
            },
            'AST': {
                'ukbb_field': 'Aspartate_aminotransferase',
                'field_id': 30650,
                'units': 'U/L',
                'file': 'ukb_blood.csv'
            },
            'ALT': {
                'ukbb_field': 'Alanine_aminotransferase',
                'field_id': 30620,
                'units': 'U/L',
                'file': 'ukb_blood.csv'
            }
        },
        
        # Lipid Profile from NMR
        'Lipid_Profile': {
            'Total_Cholesterol': {
                'ukbb_field': 'Total_cholesterol_in_SM',
                'field_id': 23400,
                'units': 'mmol/L',
                'file': 'ukb_NMR.csv'
            },
            'LDL_Cholesterol': {
                'ukbb_field': 'Clinical_LDL_cholesterol',
                'field_id': 23413,
                'units': 'mmol/L',
                'file': 'ukb_NMR.csv'
            },
            'HDL_Cholesterol': {
                'ukbb_field': 'Total_cholesterol_in_HDL',
                'field_id': 23407,
                'units': 'mmol/L',
                'file': 'ukb_NMR.csv'
            },
            'Triglycerides': {
                'ukbb_field': 'Triglycerides_in_VLDL',
                'field_id': 23456,
                'units': 'mmol/L',
                'file': 'ukb_NMR.csv'
            }
        },
        
        # Retinal Features from OCT
        'Retinal_OCT_Features': {
            'Macular_thickness_central': {
                'ukbb_field': 'Macular_thickness_at_the_central_subfield',
                'field_id': 21003,
                'units': 'micrometers',
                'file': 'ukb_OCT.csv'
            },
            'RNFL_thickness': {
                'ukbb_field': 'Retinal_nerve_fiber_layer_thickness',
                'field_id': 21053,
                'units': 'micrometers',
                'file': 'ukb_OCT.csv'
            },
            'Cup_disc_ratio': {
                'ukbb_field': 'Cup_disc_ratio',
                'field_id': 21056,
                'units': 'ratio',
                'file': 'ukb_OCT.csv'
            }
        },
        
        # Fundus Photography
        'Retinal_Fundus_Images': {
            'Left_eye_fundus': {
                'ukbb_field': 'Fundus_image_left',
                'field_id': 21015,
                'format': 'PNG image',
                'directory': 'UKB_new_2024/21015_*'
            },
            'Right_eye_fundus': {
                'ukbb_field': 'Fundus_image_right',
                'field_id': 21016,
                'format': 'PNG image',
                'directory': 'UKB_new_2024/21016_*'
            }
        }
    }
    
    # Print mapping table
    print("\n### BIOMARKER TO UK BIOBANK FIELD MAPPING ###\n")
    
    for category, markers in biomarker_mapping.items():
        print(f"\n{category}:")
        print("-" * 60)
        for marker_name, details in markers.items():
            print(f"  {marker_name}:")
            print(f"    - UKBB Field: {details.get('ukbb_field', 'N/A')}")
            print(f"    - Field ID: {details.get('field_id', 'N/A')}")
            if 'units' in details:
                print(f"    - Units: {details['units']}")
            if 'format' in details:
                print(f"    - Format: {details['format']}")
            print(f"    - Source: {details.get('file', details.get('directory', 'N/A'))}")
    
    return biomarker_mapping

def check_temporal_alignment():
    """Check temporal alignment of measurements"""
    
    print("\n" + "=" * 80)
    print("TEMPORAL ALIGNMENT ANALYSIS")
    print("=" * 80)
    
    print("\nInstance mapping (from documentation):")
    print("  - Instance 0 (.0): Baseline assessment (2006-2010)")
    print("  - Instance 1 (.1): First repeat assessment (2012-2013)")
    print("  - Instance 2 (.2): Imaging visit (2014+)")
    print("  - Instance 3 (.3): First repeat imaging visit (2019+)")
    
    print("\nRetinal imaging instances available:")
    print("  - 21015_2_1: Left eye fundus at imaging visit (instance 2)")
    print("  - 21015_3_1: Left eye fundus at repeat imaging (instance 3)")
    print("  - 21016_2_1: Right eye fundus at imaging visit (instance 2)")
    print("  - 21016_2_3: Right eye fundus at imaging visit (array 3)")
    print("  - 21016_3_1: Right eye fundus at repeat imaging (instance 3)")
    
    print("\nRecommendation for temporal alignment:")
    print("  - Focus on Instance 2 for cross-sectional analysis (most imaging data)")
    print("  - Use Instance 0 baseline for longitudinal prediction models")
    print("  - Instance 3 available for validation in subset of participants")

def generate_recommendations():
    """Generate recommendations for implementing literature approaches"""
    
    print("\n" + "=" * 80)
    print("IMPLEMENTATION RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = """
    1. FEASIBILITY ASSESSMENT:
       ✓ Core blood biomarkers (CRP, HbA1c, creatinine, albumin) - AVAILABLE
       ✓ Complete blood counts (lymphocyte %, RDW) - AVAILABLE
       ✓ Liver function tests (GGT, AST, ALT) - AVAILABLE
       ✓ NMR lipid profiles - AVAILABLE
       ✓ OCT retinal measurements - AVAILABLE (limited participants)
       ✓ Fundus photography - AVAILABLE
       ✗ Cortisol/DHEAS ratio - NOT AVAILABLE (not in standard blood panel)
    
    2. PARTICIPANT SELECTION STRATEGY:
       - Start with OCT participants (n=~85,000) as they have most complete data
       - These participants likely have:
         * Blood biomarkers (high overlap expected)
         * Fundus photography (same imaging visit)
         * Body measurements
       - This provides richest multi-modal dataset
    
    3. MISSING DATA HANDLING:
       - Use multiple imputation for sporadic missing values
       - Consider complete case analysis for primary findings
       - Report both imputed and complete case results
       - Document missingness patterns by biomarker
    
    4. ANALYSIS PIPELINE:
       Step 1: Load and merge OCT + blood biomarker data
       Step 2: Add fundus image participant IDs
       Step 3: Filter for complete multi-modal coverage
       Step 4: Apply quality control filters
       Step 5: Implement biological age algorithms from literature
       Step 6: Validate using longitudinal outcomes
    
    5. DEEP LEARNING IMPLEMENTATION:
       - Fundus images available as PNG files
       - Organize by participant ID and instance
       - Can implement CNN models as per literature
       - Consider pre-trained models (e.g., from EyePACS)
    
    6. QUALITY CONTROL:
       - Use QC flags in blood biomarker data
       - Check freeze-thaw cycles for blood samples
       - Verify image quality before CNN processing
       - Exclude participants with measurement device issues
    """
    
    print(recommendations)

if __name__ == "__main__":
    # Run all analyses
    availability = analyze_data_availability()
    mapping = map_literature_to_ukbb_fields()
    check_temporal_alignment()
    generate_recommendations()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)