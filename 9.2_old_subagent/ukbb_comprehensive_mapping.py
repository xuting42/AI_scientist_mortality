#!/usr/bin/env python3
"""
UK Biobank Data Mapping and Participant Intersection Analysis
Comprehensive analysis of available data modalities and their intersections
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
from typing import Dict, List, Set, Tuple
import warnings
warnings.filterwarnings('ignore')

class UKBBDataMapper:
    """Main class for UK Biobank data mapping and intersection analysis"""
    
    def __init__(self, data_dir: str = "/mnt/data1/UKBB", 
                 retinal_dir: str = "/mnt/data1/UKBB_retinal_img/UKB_new_2024"):
        self.data_dir = data_dir
        self.retinal_dir = retinal_dir
        self.participants = {}
        self.biomarker_mapping = {}
        self.data_availability = {}
        
    def map_literature_biomarkers(self):
        """Map literature-identified biomarkers to UK Biobank field IDs"""
        
        print("=" * 80)
        print("MAPPING LITERATURE BIOMARKERS TO UK BIOBANK FIELDS")
        print("=" * 80)
        
        # Literature-based biomarker categories
        biomarker_categories = {
            'PhysAge_Clock_Surrogates': {
                'albumin': 'Albumin',
                'alkaline_phosphatase': 'Alkaline_phosphatase',
                'creatinine': 'Creatinine',
                'c_reactive_protein': 'C_reactive_protein',
                'glucose': 'Glucose',
                'red_cell_distribution_width': 'Red_blood_cell_erythrocyte_distribution_width',
                'white_blood_cell_count': 'White_blood_cell_leukocyte_count',
                'forced_expiratory_volume': 'Forced_expiratory_volume_in_1_second_FEV1'
            },
            'Inflammatory_Markers': {
                'crp': 'C_reactive_protein',
                'neutrophil_count': 'Neutrophill_count',
                'lymphocyte_count': 'Lymphocyte_count',
                'monocyte_count': 'Monocyte_count'
            },
            'Metabolic_Panel': {
                'hba1c': 'Glycated_haemoglobin_HbA1c',
                'glucose': 'Glucose',
                'cholesterol_total': 'Cholesterol',
                'hdl_cholesterol': 'HDL_cholesterol',
                'ldl_cholesterol': 'LDL_direct',
                'triglycerides': 'Triglycerides'
            },
            'Liver_Kidney_Function': {
                'ast': 'Aspartate_aminotransferase',
                'alt': 'Alanine_aminotransferase',
                'ggt': 'Gamma_glutamyltransferase',
                'urea': 'Urea',
                'creatinine': 'Creatinine',
                'egfr': 'eGFR'
            },
            'Cardiovascular_Markers': {
                'systolic_bp': 'Systolic_blood_pressure',
                'diastolic_bp': 'Diastolic_blood_pressure',
                'pulse_rate': 'Pulse_rate'
            },
            'Complete_Blood_Count': {
                'rbc_count': 'Red_blood_cell_erythrocyte_count',
                'wbc_count': 'White_blood_cell_leukocyte_count',
                'platelet_count': 'Platelet_count',
                'haemoglobin': 'Haemoglobin_concentration',
                'haematocrit': 'Haematocrit',
                'mcv': 'Mean_corpuscular_volume',
                'mch': 'Mean_corpuscular_haemoglobin',
                'mchc': 'Mean_corpuscular_haemoglobin_concentration'
            }
        }
        
        # Check availability in blood file
        try:
            blood_df = pd.read_csv(f"{self.data_dir}/ukb_blood.csv", nrows=5)
            blood_cols = blood_df.columns.tolist()
            
            print("\n1. BLOOD BIOMARKERS MAPPING:")
            print("-" * 40)
            
            for category, markers in biomarker_categories.items():
                print(f"\n{category}:")
                available = 0
                for lit_name, ukb_pattern in markers.items():
                    matching = [col for col in blood_cols if ukb_pattern.lower() in col.lower() 
                               and not 'date' in col.lower() and not 'aliquot' in col.lower()]
                    if matching:
                        print(f"  ✓ {lit_name}: {matching[0]}")
                        available += 1
                    else:
                        print(f"  ✗ {lit_name}: NOT FOUND")
                print(f"  Coverage: {available}/{len(markers)} ({100*available/len(markers):.1f}%)")
                
        except Exception as e:
            print(f"Error loading blood data: {e}")
            
        # Check body measurements
        try:
            body_df = pd.read_csv(f"{self.data_dir}/ukb_body.csv", nrows=5)
            body_cols = body_df.columns.tolist()
            
            print("\n2. BODY MEASUREMENTS MAPPING:")
            print("-" * 40)
            
            body_measures = {
                'bmi': 'Body_mass_index_BMI',
                'weight': 'Weight',
                'height': 'Standing_height',
                'waist_circumference': 'Waist_circumference',
                'hip_circumference': 'Hip_circumference',
                'body_fat_percentage': 'Body_fat_percentage',
                'systolic_bp': 'Systolic_blood_pressure',
                'diastolic_bp': 'Diastolic_blood_pressure',
                'pulse_rate': 'Pulse_rate'
            }
            
            for measure, pattern in body_measures.items():
                matching = [col for col in body_cols if pattern.lower() in col.lower()]
                if matching:
                    print(f"  ✓ {measure}: {matching[0]}")
                else:
                    print(f"  ✗ {measure}: NOT FOUND")
                    
        except Exception as e:
            print(f"Error loading body data: {e}")
            
    def analyze_participant_intersections(self):
        """Analyze participant overlaps across different data modalities"""
        
        print("\n" + "=" * 80)
        print("PARTICIPANT INTERSECTION ANALYSIS")
        print("=" * 80)
        
        participants_by_modality = {}
        
        # 1. Blood biomarkers participants
        try:
            print("\nLoading blood biomarker data...")
            blood_df = pd.read_csv(f"{self.data_dir}/ukb_blood.csv", 
                                  usecols=['f.eid', 'Albumin', 'Creatinine', 'C_reactive_protein'])
            blood_participants = set(blood_df['f.eid'].dropna())
            # Count with complete key biomarkers
            complete_blood = blood_df.dropna(subset=['Albumin', 'Creatinine', 'C_reactive_protein'])
            participants_by_modality['blood_complete'] = set(complete_blood['f.eid'])
            participants_by_modality['blood_any'] = blood_participants
            print(f"  Total with any blood data: {len(blood_participants):,}")
            print(f"  With complete key biomarkers: {len(participants_by_modality['blood_complete']):,}")
            del blood_df
        except Exception as e:
            print(f"  Error: {e}")
            participants_by_modality['blood_any'] = set()
            participants_by_modality['blood_complete'] = set()
            
        # 2. Body measurements participants
        try:
            print("\nLoading body measurement data...")
            body_df = pd.read_csv(f"{self.data_dir}/ukb_body.csv", 
                                 usecols=['f.eid'], nrows=1000000)
            body_participants = set(body_df['f.eid'].dropna())
            participants_by_modality['body'] = body_participants
            print(f"  Total with body measurements: {len(body_participants):,}")
            del body_df
        except Exception as e:
            print(f"  Error: {e}")
            participants_by_modality['body'] = set()
            
        # 3. OCT data participants
        try:
            print("\nLoading OCT data...")
            oct_df = pd.read_csv(f"{self.data_dir}/ukb_OCT.csv", 
                                usecols=['f.eid'])
            oct_participants = set(oct_df['f.eid'].dropna())
            participants_by_modality['oct'] = oct_participants
            print(f"  Total with OCT data: {len(oct_participants):,}")
            del oct_df
        except Exception as e:
            print(f"  Error: {e}")
            participants_by_modality['oct'] = set()
            
        # 4. Retinal imaging participants (from directory structure)
        print("\nAnalyzing retinal imaging data...")
        retinal_participants = set()
        try:
            # Count directories which represent participant IDs
            retinal_dirs = os.listdir(self.retinal_dir)
            print(f"  Total retinal imaging directories: {len(retinal_dirs):,}")
            # Sample check for actual images
            sample_dir = retinal_dirs[0] if retinal_dirs else None
            if sample_dir:
                sample_files = os.listdir(os.path.join(self.retinal_dir, sample_dir))
                print(f"  Sample directory '{sample_dir}' contains {len(sample_files)} files")
            participants_by_modality['retinal'] = set(range(1, len(retinal_dirs) + 1))  # Placeholder
        except Exception as e:
            print(f"  Error: {e}")
            participants_by_modality['retinal'] = set()
            
        # 5. NMR metabolomics participants
        try:
            print("\nLoading NMR metabolomics data...")
            nmr_df = pd.read_csv(f"{self.data_dir}/ukb_NMR.csv", 
                                usecols=['f.eid'], nrows=500000)
            nmr_participants = set(nmr_df['f.eid'].dropna())
            participants_by_modality['nmr'] = nmr_participants
            print(f"  Total with NMR data: {len(nmr_participants):,}")
            del nmr_df
        except Exception as e:
            print(f"  Error: {e}")
            participants_by_modality['nmr'] = set()
            
        # Calculate intersections
        print("\n" + "=" * 80)
        print("INTERSECTION ANALYSIS RESULTS")
        print("=" * 80)
        
        # Key intersections for biological age algorithms
        if participants_by_modality['blood_complete'] and participants_by_modality['body']:
            blood_body = participants_by_modality['blood_complete'] & participants_by_modality['body']
            print(f"\n1. Blood biomarkers ∩ Body measurements: {len(blood_body):,}")
            
        if participants_by_modality['blood_complete'] and participants_by_modality['oct']:
            blood_oct = participants_by_modality['blood_complete'] & participants_by_modality['oct']
            print(f"2. Blood biomarkers ∩ OCT imaging: {len(blood_oct):,}")
            
        if participants_by_modality['blood_complete'] and participants_by_modality['nmr']:
            blood_nmr = participants_by_modality['blood_complete'] & participants_by_modality['nmr']
            print(f"3. Blood biomarkers ∩ NMR metabolomics: {len(blood_nmr):,}")
            
        # Triple intersection
        if (participants_by_modality['blood_complete'] and 
            participants_by_modality['body'] and 
            participants_by_modality['oct']):
            triple = (participants_by_modality['blood_complete'] & 
                     participants_by_modality['body'] & 
                     participants_by_modality['oct'])
            print(f"\n4. Blood ∩ Body ∩ OCT (TRIPLE): {len(triple):,}")
            
        return participants_by_modality
        
    def assess_data_quality(self):
        """Assess data quality and completeness for key variables"""
        
        print("\n" + "=" * 80)
        print("DATA QUALITY ASSESSMENT")
        print("=" * 80)
        
        # Sample blood data for quality assessment
        try:
            print("\nAssessing blood biomarker quality...")
            key_biomarkers = ['f.eid', 'Albumin', 'Alkaline_phosphatase', 
                             'Creatinine', 'C_reactive_protein', 'Glucose',
                             'White_blood_cell_leukocyte_count', 'Haemoglobin_concentration']
            
            blood_sample = pd.read_csv(f"{self.data_dir}/ukb_blood.csv", 
                                      usecols=lambda x: any(marker in x for marker in key_biomarkers),
                                      nrows=100000)
            
            print("\nCompleteness for key PhysAge biomarkers (first 100k participants):")
            for col in blood_sample.columns:
                if col != 'f.eid':
                    completeness = 100 * blood_sample[col].notna().sum() / len(blood_sample)
                    print(f"  {col[:40]:40s}: {completeness:5.1f}% complete")
                    
            # Check for outliers
            print("\nOutlier detection (values beyond 5 SD from mean):")
            for col in blood_sample.select_dtypes(include=[np.number]).columns:
                if col != 'f.eid':
                    mean = blood_sample[col].mean()
                    std = blood_sample[col].std()
                    outliers = ((blood_sample[col] < mean - 5*std) | 
                               (blood_sample[col] > mean + 5*std)).sum()
                    if outliers > 0:
                        print(f"  {col[:40]:40s}: {outliers} potential outliers")
                        
        except Exception as e:
            print(f"Error in quality assessment: {e}")
            
    def generate_recommendations(self, participants_by_modality):
        """Generate recommendations for algorithm development"""
        
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS FOR BIOLOGICAL AGE ALGORITHM DEVELOPMENT")
        print("=" * 80)
        
        print("\n1. HIGH-PRIORITY FEATURE SETS (Based on Coverage):")
        print("-" * 40)
        
        recommendations = [
            {
                'name': 'Basic PhysAge Panel',
                'features': ['Albumin', 'Creatinine', 'CRP', 'Glucose', 'WBC', 'RDW'],
                'estimated_n': len(participants_by_modality.get('blood_complete', [])),
                'priority': 'HIGH',
                'rationale': 'Direct replication of validated PhysAge clock'
            },
            {
                'name': 'Extended Metabolic Panel',
                'features': ['HbA1c', 'Lipids', 'Liver enzymes', 'eGFR'],
                'estimated_n': len(participants_by_modality.get('blood_any', [])),
                'priority': 'HIGH',
                'rationale': 'Comprehensive metabolic health assessment'
            },
            {
                'name': 'Multimodal Integration',
                'features': ['Blood biomarkers', 'OCT metrics', 'Body composition'],
                'estimated_n': len(participants_by_modality.get('oct', [])),
                'priority': 'MEDIUM',
                'rationale': 'Novel cross-modal biological age estimation'
            },
            {
                'name': 'NMR Metabolomics',
                'features': ['249 NMR biomarkers', 'Lipoprotein subfractions'],
                'estimated_n': len(participants_by_modality.get('nmr', [])),
                'priority': 'MEDIUM',
                'rationale': 'Deep metabolic profiling for precision aging'
            }
        ]
        
        for rec in recommendations:
            print(f"\n{rec['name']}:")
            print(f"  Priority: {rec['priority']}")
            print(f"  Estimated N: {rec['estimated_n']:,}")
            print(f"  Rationale: {rec['rationale']}")
            
        print("\n2. FEASIBILITY RANKING:")
        print("-" * 40)
        print("  1. Blood-based PhysAge replication: HIGHLY FEASIBLE")
        print("  2. Blood + Body measurements: HIGHLY FEASIBLE")
        print("  3. Blood + OCT integration: MODERATELY FEASIBLE")
        print("  4. Full multimodal (Blood+OCT+NMR): LIMITED FEASIBILITY")
        
        print("\n3. NOVEL OPPORTUNITIES:")
        print("-" * 40)
        print("  • Retinal age gap validation with blood biomarkers")
        print("  • Longitudinal validation using repeat assessments")
        print("  • Sex-stratified biological age models")
        print("  • Integration of frailty indices with molecular markers")
        
    def run_complete_analysis(self):
        """Run the complete mapping and analysis pipeline"""
        
        print("\n" + "="*80)
        print(" UK BIOBANK DATA MAPPING AND INTERSECTION ANALYSIS")
        print(" Analysis Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("="*80)
        
        # Step 1: Map biomarkers
        self.map_literature_biomarkers()
        
        # Step 2: Analyze intersections
        participants = self.analyze_participant_intersections()
        
        # Step 3: Assess quality
        self.assess_data_quality()
        
        # Step 4: Generate recommendations
        self.generate_recommendations(participants)
        
        print("\n" + "="*80)
        print(" ANALYSIS COMPLETE")
        print("="*80)

if __name__ == "__main__":
    mapper = UKBBDataMapper()
    mapper.run_complete_analysis()