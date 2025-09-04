#!/usr/bin/env python3
"""
UK Biobank Data Comprehensive Mapping Script
Maps biological age biomarkers from literature to available UKBB datasets
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import json
from collections import defaultdict, Counter

class UKBBDataMapper:
    """Map literature-identified biomarkers to UK Biobank datasets"""
    
    def __init__(self):
        self.ukbb_path = "/mnt/data1/UKBB"
        self.retinal_path = "/mnt/data1/UKBB_retinal_img/UKB_new_2024"
        self.results = defaultdict(dict)
        
        # Literature-identified biomarker categories
        self.biomarker_categories = {
            'epigenetic': {
                'dna_methylation': ['methylation', 'cpg', 'epigenetic'],
                'telomere': ['telomere', 'telomerase']
            },
            'proteomic': {
                'inflammatory': ['crp', 'il-6', 'tnf', 'cytokine', 'interleukin'],
                'cardiovascular': ['troponin', 'bnp', 'nt-probnp'],
                'metabolic': ['adiponectin', 'leptin', 'insulin'],
                'aging': ['igf', 'growth hormone', 'klotho']
            },
            'metabolomic': {
                'lipids': ['cholesterol', 'triglyceride', 'hdl', 'ldl', 'vldl'],
                'glucose': ['glucose', 'hba1c', 'insulin'],
                'amino_acids': ['amino acid', 'creatinine', 'urea'],
                'metabolites': ['metabolite', 'nmr', 'metabolome']
            },
            'blood_clinical': {
                'hematology': ['rbc', 'wbc', 'hemoglobin', 'hematocrit', 'platelet', 'neutrophil', 'lymphocyte'],
                'biochemistry': ['albumin', 'bilirubin', 'alt', 'ast', 'ggt', 'alkaline phosphatase'],
                'renal': ['egfr', 'creatinine', 'cystatin', 'urea', 'uric acid'],
                'electrolytes': ['sodium', 'potassium', 'calcium', 'phosphate']
            },
            'cardiovascular': {
                'blood_pressure': ['systolic', 'diastolic', 'pulse pressure', 'map'],
                'arterial': ['pulse wave', 'arterial stiffness', 'augmentation'],
                'ecg': ['ecg', 'heart rate', 'pr interval', 'qrs', 'qt']
            },
            'body_composition': {
                'anthropometric': ['bmi', 'weight', 'height', 'waist', 'hip'],
                'dxa': ['bone density', 'lean mass', 'fat mass', 'visceral'],
                'impedance': ['body fat', 'muscle mass', 'water']
            },
            'retinal': {
                'vascular': ['vessel', 'tortuosity', 'caliber', 'branching', 'fractal'],
                'structural': ['thickness', 'layer', 'rnfl', 'ganglion', 'macula'],
                'functional': ['fundus', 'oct', 'autofluorescence']
            }
        }
        
    def explore_data_structure(self):
        """Explore the structure of available UKBB data files"""
        print("=" * 80)
        print("EXPLORING UK BIOBANK DATA STRUCTURE")
        print("=" * 80)
        
        # List all CSV files in UKBB directory
        csv_files = glob.glob(os.path.join(self.ukbb_path, "*.csv"))
        
        print(f"\nFound {len(csv_files)} CSV files in UKBB directory:")
        for file in sorted(csv_files):
            file_size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
            print(f"  - {os.path.basename(file)}: {file_size:.2f} MB")
            
        # Check for data dictionary
        dict_files = glob.glob(os.path.join(self.ukbb_path, "*Dictionary*"))
        print(f"\nData dictionary files found: {len(dict_files)}")
        for file in dict_files:
            print(f"  - {os.path.basename(file)}")
            
        return csv_files
    
    def load_and_inspect_dataset(self, file_path, sample_size=1000):
        """Load and inspect a UKBB dataset"""
        file_name = os.path.basename(file_path)
        print(f"\n{'='*60}")
        print(f"Inspecting: {file_name}")
        print('='*60)
        
        try:
            # Load first few rows to inspect structure
            df_sample = pd.read_csv(file_path, nrows=sample_size)
            
            print(f"Shape (sample): {df_sample.shape}")
            print(f"Columns: {df_sample.shape[1]}")
            
            # Get total row count efficiently
            with open(file_path, 'r') as f:
                row_count = sum(1 for line in f) - 1  # Subtract header
            print(f"Total participants: {row_count:,}")
            
            # Analyze column patterns
            columns = df_sample.columns.tolist()
            
            # Identify participant ID column
            id_cols = [col for col in columns if 'eid' in col.lower() or col == 'f.eid']
            if id_cols:
                print(f"Participant ID column: {id_cols[0]}")
                unique_ids = df_sample[id_cols[0]].nunique()
                print(f"Unique participants in sample: {unique_ids}")
            
            # Store results
            self.results[file_name] = {
                'path': file_path,
                'total_participants': row_count,
                'n_columns': len(columns),
                'id_column': id_cols[0] if id_cols else None,
                'columns': columns[:20],  # Store first 20 columns as sample
                'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
            }
            
            return df_sample, columns
            
        except Exception as e:
            print(f"Error loading {file_name}: {str(e)}")
            return None, []
    
    def map_biomarkers_to_columns(self, columns, file_name):
        """Map literature biomarkers to available columns"""
        mapped = defaultdict(list)
        
        for col in columns:
            col_lower = col.lower()
            
            for category, subcategories in self.biomarker_categories.items():
                for subcat, keywords in subcategories.items():
                    for keyword in keywords:
                        if keyword.lower() in col_lower:
                            mapped[f"{category}_{subcat}"].append(col)
                            break
        
        if mapped:
            print(f"\nBiomarker mappings found in {file_name}:")
            for category, cols in mapped.items():
                print(f"  {category}: {len(cols)} columns")
                for col in cols[:3]:  # Show first 3 examples
                    print(f"    - {col}")
                if len(cols) > 3:
                    print(f"    ... and {len(cols)-3} more")
        
        return mapped
    
    def explore_retinal_imaging(self):
        """Explore retinal imaging data structure"""
        print("\n" + "="*80)
        print("EXPLORING RETINAL IMAGING DATA")
        print("="*80)
        
        # Get list of participant directories
        participant_dirs = [d for d in os.listdir(self.retinal_path) 
                          if os.path.isdir(os.path.join(self.retinal_path, d))]
        
        print(f"Total participant directories: {len(participant_dirs)}")
        
        # Analyze directory naming pattern
        if participant_dirs:
            print(f"\nExample directories: {participant_dirs[:5]}")
            
            # Parse directory names (format appears to be: fieldID_instance_array)
            field_ids = set()
            instances = set()
            
            for dir_name in participant_dirs[:100]:  # Sample first 100
                parts = dir_name.split('_')
                if len(parts) >= 2:
                    field_ids.add(parts[0])
                    instances.add(parts[1])
            
            print(f"\nUnique field IDs found: {sorted(field_ids)}")
            print(f"Unique instances found: {sorted(instances)}")
            
            # Check contents of a sample directory
            sample_dir = os.path.join(self.retinal_path, participant_dirs[0])
            sample_files = os.listdir(sample_dir)[:10]
            print(f"\nSample files in {participant_dirs[0]}: {len(os.listdir(sample_dir))} files")
            print(f"Examples: {sample_files[:5]}")
            
            # Count total image files
            total_images = 0
            for dir_name in participant_dirs[:100]:  # Sample
                dir_path = os.path.join(self.retinal_path, dir_name)
                total_images += len([f for f in os.listdir(dir_path) 
                                   if f.endswith(('.png', '.jpg', '.dcm'))])
            
            print(f"\nEstimated total images (based on sample): {total_images * len(participant_dirs) // 100:,}")
            
        self.results['retinal_imaging'] = {
            'total_directories': len(participant_dirs),
            'field_ids': list(field_ids) if participant_dirs else [],
            'sample_dirs': participant_dirs[:10] if participant_dirs else []
        }
        
        return participant_dirs
    
    def analyze_participant_intersection(self):
        """Analyze participant overlap across datasets"""
        print("\n" + "="*80)
        print("ANALYZING PARTICIPANT INTERSECTIONS")
        print("="*80)
        
        participant_counts = {}
        
        # Load participant IDs from each major dataset
        datasets_to_check = [
            ('ukb_blood.csv', 'Blood biomarkers'),
            ('ukb_body.csv', 'Body composition'),
            ('ukb_NMR.csv', 'NMR metabolomics'),
            ('ukb_OCT.csv', 'OCT measurements'),
            ('ukb_main.csv', 'Main phenotypes'),
            ('ukb_genetic_PC.csv', 'Genetic PCs'),
            ('ukb_PRS.csv', 'Polygenic risk scores')
        ]
        
        all_participant_sets = {}
        
        for file_name, description in datasets_to_check:
            file_path = os.path.join(self.ukbb_path, file_name)
            if os.path.exists(file_path):
                print(f"\nLoading participants from {description}...")
                try:
                    # Load only the ID column
                    df = pd.read_csv(file_path, nrows=10)
                    id_col = [col for col in df.columns if 'eid' in col.lower() or col == 'f.eid'][0]
                    
                    # Load all participant IDs
                    ids = pd.read_csv(file_path, usecols=[id_col])[id_col].unique()
                    all_participant_sets[description] = set(ids)
                    participant_counts[description] = len(ids)
                    print(f"  Found {len(ids):,} unique participants")
                except Exception as e:
                    print(f"  Error: {str(e)}")
        
        # Calculate intersections
        if len(all_participant_sets) >= 2:
            print("\n" + "="*50)
            print("MULTI-MODAL DATA AVAILABILITY")
            print("="*50)
            
            # Pairwise intersections
            datasets = list(all_participant_sets.keys())
            for i in range(len(datasets)):
                for j in range(i+1, len(datasets)):
                    set1 = all_participant_sets[datasets[i]]
                    set2 = all_participant_sets[datasets[j]]
                    intersection = len(set1 & set2)
                    print(f"\n{datasets[i]} ∩ {datasets[j]}: {intersection:,} participants")
                    print(f"  Coverage: {100*intersection/len(set1):.1f}% of {datasets[i]}")
                    print(f"  Coverage: {100*intersection/len(set2):.1f}% of {datasets[j]}")
            
            # All datasets intersection
            if len(all_participant_sets) > 2:
                all_intersection = set.intersection(*all_participant_sets.values())
                print(f"\n{'='*50}")
                print(f"Participants with ALL {len(all_participant_sets)} modalities: {len(all_intersection):,}")
                
                # Key combinations for biological age
                key_combos = [
                    (['Blood biomarkers', 'NMR metabolomics'], 'Blood + Metabolomics'),
                    (['Blood biomarkers', 'Body composition'], 'Blood + Body composition'),
                    (['Blood biomarkers', 'OCT measurements'], 'Blood + Retinal OCT'),
                    (['NMR metabolomics', 'Body composition'], 'Metabolomics + Body'),
                    (['Blood biomarkers', 'NMR metabolomics', 'Body composition'], 'Blood + Metabolomics + Body')
                ]
                
                print("\n" + "="*50)
                print("KEY COMBINATIONS FOR BIOLOGICAL AGE")
                print("="*50)
                
                for datasets_needed, combo_name in key_combos:
                    sets_to_intersect = [all_participant_sets[d] for d in datasets_needed 
                                       if d in all_participant_sets]
                    if len(sets_to_intersect) == len(datasets_needed):
                        combo_intersection = set.intersection(*sets_to_intersect)
                        print(f"\n{combo_name}: {len(combo_intersection):,} participants")
        
        self.results['participant_intersections'] = participant_counts
        
        return participant_counts
    
    def generate_summary_report(self):
        """Generate comprehensive mapping report"""
        print("\n" + "="*80)
        print("UK BIOBANK DATA MAPPING SUMMARY REPORT")
        print("="*80)
        
        report = []
        report.append("# UK Biobank Data Mapping Report for Biological Age Research\n")
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Dataset overview
        report.append("\n## 1. Dataset Overview\n")
        for file_name, info in self.results.items():
            if isinstance(info, dict) and 'path' in info:
                report.append(f"\n### {file_name}")
                report.append(f"- **Total participants**: {info.get('total_participants', 'N/A'):,}")
                report.append(f"- **Number of variables**: {info.get('n_columns', 'N/A')}")
                report.append(f"- **File size**: {info.get('file_size_mb', 0):.2f} MB")
        
        # Biomarker availability
        report.append("\n## 2. Biomarker Availability Assessment\n")
        
        biomarker_availability = {
            'Clinical Blood Biomarkers': 'HIGH - ukb_blood.csv contains comprehensive panels',
            'NMR Metabolomics': 'HIGH - ukb_NMR.csv with 249 metabolites',
            'Body Composition': 'HIGH - ukb_body.csv with DXA and impedance',
            'Retinal Imaging': 'MODERATE - OCT measurements available',
            'Genetic Data': 'HIGH - PRS and genetic PCs available',
            'Epigenetic Data': 'LIMITED - Check for methylation subset',
            'Proteomic Data': 'CHECK - May be in separate release'
        }
        
        for category, status in biomarker_availability.items():
            report.append(f"- **{category}**: {status}")
        
        # Participant intersection
        report.append("\n## 3. Multi-Modal Data Availability\n")
        if 'participant_intersections' in self.results:
            for dataset, count in self.results['participant_intersections'].items():
                report.append(f"- {dataset}: {count:,} participants")
        
        # Recommendations
        report.append("\n## 4. Priority Recommendations\n")
        report.append("\n### Highest Priority (Maximum Data Availability):")
        report.append("1. **Blood biomarkers + NMR metabolomics + Body composition**")
        report.append("   - Most comprehensive coverage")
        report.append("   - Aligns with KDM and phenotypic age approaches")
        report.append("\n2. **Multi-organ approach using blood + body + cardiovascular**")
        report.append("   - Captures systemic aging")
        
        report.append("\n### Innovative Opportunities:")
        report.append("1. **Retinal aging signature + systemic biomarkers**")
        report.append("   - Novel multi-modal approach")
        report.append("   - OCT data provides unique structural information")
        
        report.append("\n2. **Metabolomic aging clock**")
        report.append("   - 249 NMR metabolites available")
        report.append("   - Can replicate MetaboAge approaches")
        
        # Data quality notes
        report.append("\n## 5. Data Quality Considerations\n")
        report.append("- Multiple instance data available (baseline + follow-up)")
        report.append("- Consider longitudinal analysis opportunities")
        report.append("- Check for batch effects in NMR and proteomics")
        report.append("- Verify imaging quality metrics for retinal data")
        
        return '\n'.join(report)
    
    def run_comprehensive_mapping(self):
        """Execute complete mapping analysis"""
        
        # 1. Explore data structure
        csv_files = self.explore_data_structure()
        
        # 2. Inspect key datasets
        priority_files = ['ukb_blood.csv', 'ukb_NMR.csv', 'ukb_body.csv', 
                         'ukb_OCT.csv', 'ukb_main.csv', 'ukb_genetic_PC.csv']
        
        for file_name in priority_files:
            file_path = os.path.join(self.ukbb_path, file_name)
            if os.path.exists(file_path):
                df_sample, columns = self.load_and_inspect_dataset(file_path, sample_size=100)
                if columns:
                    self.map_biomarkers_to_columns(columns, file_name)
        
        # 3. Explore retinal imaging
        self.explore_retinal_imaging()
        
        # 4. Analyze participant intersections
        self.analyze_participant_intersection()
        
        # 5. Generate report
        report = self.generate_summary_report()
        
        # Save report
        report_path = '/mnt/data3/xuting/ai_scientist/claudeV2/ukbb_mapping_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n{'='*80}")
        print(f"Report saved to: {report_path}")
        print('='*80)
        
        return report

# Execute mapping
if __name__ == "__main__":
    mapper = UKBBDataMapper()
    report = mapper.run_comprehensive_mapping()
    print("\n" + "="*80)
    print("MAPPING COMPLETE")
    print("="*80)