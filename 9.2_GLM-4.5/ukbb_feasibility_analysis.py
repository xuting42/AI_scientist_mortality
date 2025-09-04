#!/usr/bin/env python3
"""
UK Biobank Biological Age Feasibility Analysis
Analyzes available data fields for multi-modal biological age algorithm implementation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import defaultdict

class UKBBFeasibilityAnalyzer:
    def __init__(self, ukbb_path):
        self.ukbb_path = Path(ukbb_path)
        self.data_files = {
            'main': 'ukb_main.csv',
            'blood': 'ukb_blood.csv', 
            'nmr': 'ukb_NMR.csv',
            'oct': 'ukb_OCT.csv',
            'brain': 'ukb_brain.csv',
            'prs': 'ukb_PRS.csv',
            'body': 'ukb_body.csv',
            'frailty': 'ukb_frailty.csv',
            'bodysize': 'ukb_bodysize.csv',
            'genetic_pc': 'ukb_genetic_PC.csv',
            'dxa': 'ukb20240116_DXA_long_named.csv'
        }
        
        # Literature-identified key biomarkers
        self.key_biomarkers = {
            'epigenetic': [
                'DNA methylation arrays',
                'CpG sites', 
                'Methylation age acceleration'
            ],
            'blood_biomarkers': [
                'Albumin', 'Glucose', 'Alkaline phosphatase', 'Urea', 'Erythrocytes',
                'C-reactive protein (CRP)', 'IGF-1', 'IL-6', 'GDF-15', 'Cortisol',
                'DHEAS', 'Cholesterol', 'HDL', 'LDL', 'Triglycerides', 'HbA1c',
                'Creatinine', 'Uric acid', 'Bilirubin', 'ALT', 'AST', 'ALP', 'GGT'
            ],
            'inflammatory': [
                'IL-1ra', 'IL-6', 'IL-17a', 'IL-18', 'CCL-2', 'CCL-4', 
                'TNF-α', 'IFN-γ', 'CD40L', 'White blood cell count'
            ],
            'retinal': [
                'Macular thickness', 'Retinal nerve fiber layer', 'Cup-disc ratio',
                'Retinal vessel caliber', 'Choroidal thickness', 'Optical coherence tomography'
            ],
            'body_composition': [
                'BMI', 'Waist circumference', 'Hip circumference', 'Hand grip strength',
                'Spirometry (FEV1, FVC)', 'Bone density', 'Muscle mass'
            ]
        }
        
        self.data_availability = {}
        self.participant_counts = {}
        self.field_mappings = {}
        
    def load_column_names(self, file_key):
        """Load column names from CSV file"""
        file_path = self.ukbb_path / self.data_files[file_key]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                header = f.readline().strip()
            columns = [col.strip('"') for col in header.split(',')]
            return columns
        except Exception as e:
            print(f"Error loading {file_key}: {e}")
            return []
    
    def find_biomarker_matches(self, columns, biomarker_list):
        """Find matching columns for literature-identified biomarkers"""
        matches = defaultdict(list)
        
        for biomarker in biomarker_list:
            # Create search patterns for each biomarker
            patterns = [
                biomarker.lower(),
                biomarker.lower().replace(' ', '_'),
                biomarker.lower().replace('-', '_'),
                re.sub(r'[^a-z0-9]', '', biomarker.lower())
            ]
            
            for col in columns:
                col_lower = col.lower()
                # Remove instance suffixes for matching
                col_base = re.sub(r'\.\d+$', '', col_lower)
                
                for pattern in patterns:
                    if pattern in col_base and not any(skip in col_base for skip in 
                           ['freeze_thaw', 'cycles', 'aliquot', 'date', 'level', 
                            'reason', 'missing', 'reportability', 'assay', 'correction',
                            'qc_flag']):
                        matches[biomarker].append(col)
                        break
        
        return dict(matches)
    
    def get_participant_count(self, file_key):
        """Get number of participants in each file"""
        file_path = self.ukbb_path / self.data_files[file_key]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f) - 1  # Subtract header
            return line_count
        except Exception as e:
            print(f"Error counting participants in {file_key}: {e}")
            return 0
    
    def analyze_blood_biomarkers(self):
        """Analyze blood biomarker availability"""
        print("Analyzing blood biomarkers...")
        
        # Load blood data columns
        blood_columns = self.load_column_names('blood')
        print(f"Blood data columns: {len(blood_columns)}")
        
        # Find matches for key blood biomarkers
        blood_matches = self.find_biomarker_matches(blood_columns, self.key_biomarkers['blood_biomarkers'])
        
        # Also check NMR data for additional metabolic markers
        nmr_columns = self.load_column_names('nmr')
        nmr_matches = self.find_biomarker_matches(nmr_columns, self.key_biomarkers['blood_biomarkers'])
        
        # Filter NMR matches to actual measurements (not QC flags)
        nmr_measurements = {}
        for biomarker, columns in nmr_matches.items():
            nmr_measurements[biomarker] = [col for col in columns if not col.endswith('QC_Flag')]
        
        return {
            'blood_biomarkers': blood_matches,
            'nmr_biomarkers': nmr_measurements,
            'blood_participants': self.get_participant_count('blood'),
            'nmr_participants': self.get_participant_count('nmr')
        }
    
    def analyze_retinal_data(self):
        """Analyze retinal imaging data availability"""
        print("Analyzing retinal imaging data...")
        
        oct_columns = self.load_column_names('oct')
        print(f"OCT data columns: {len(oct_columns)}")
        
        retinal_matches = self.find_biomarker_matches(oct_columns, self.key_biomarkers['retinal'])
        
        return {
            'retinal_biomarkers': retinal_matches,
            'oct_participants': self.get_participant_count('oct')
        }
    
    def analyze_body_composition(self):
        """Analyze body composition and physical function data"""
        print("Analyzing body composition data...")
        
        body_columns = self.load_column_names('body')
        bodysize_columns = self.load_column_names('bodysize')
        frailty_columns = self.load_column_names('frailty')
        
        all_body_columns = body_columns + bodysize_columns + frailty_columns
        
        body_matches = self.find_biomarker_matches(all_body_columns, self.key_biomarkers['body_composition'])
        
        return {
            'body_biomarkers': body_matches,
            'body_participants': self.get_participant_count('body'),
            'bodysize_participants': self.get_participant_count('bodysize'),
            'frailty_participants': self.get_participant_count('frailty')
        }
    
    def analyze_genetic_data(self):
        """Analyze genetic and epigenetic data availability"""
        print("Analyzing genetic data...")
        
        prs_columns = self.load_column_names('prs')
        genetic_pc_columns = self.load_column_names('genetic_pc')
        
        return {
            'prs_count': len(prs_columns) - 1,  # Subtract f.eid
            'genetic_pc_count': len(genetic_pc_columns) - 1,
            'prs_participants': self.get_participant_count('prs'),
            'genetic_pc_participants': self.get_participant_count('genetic_pc')
        }
    
    def analyze_demographics(self):
        """Analyze demographic and baseline data"""
        print("Analyzing demographic data...")
        
        main_columns = self.load_column_names('main')
        main_participants = self.get_participant_count('main')
        
        # Find key demographic variables
        demo_vars = ['sex', 'birth', 'age', 'ethnic', 'race', 'center', 'income', 'education', 'townsend']
        demo_matches = self.find_biomarker_matches(main_columns, demo_vars)
        
        return {
            'demographic_variables': demo_matches,
            'main_participants': main_participants
        }
    
    def calculate_data_overlap(self):
        """Calculate participant overlap across different data modalities"""
        print("Calculating data overlap...")
        
        # This would require loading actual participant IDs
        # For now, provide theoretical maximums based on file sizes
        overlap_analysis = {
            'total_participants': self.get_participant_count('main'),
            'blood_coverage': self.get_participant_count('blood'),
            'genetic_coverage': self.get_participant_count('prs'),
            'imaging_coverage': {
                'oct': self.get_participant_count('oct'),
                'brain': self.get_participant_count('brain'),
                'dxa': self.get_participant_count('dxa')
            }
        }
        
        return overlap_analysis
    
    def generate_feasibility_report(self):
        """Generate comprehensive feasibility assessment"""
        print("Generating feasibility report...")
        
        # Analyze each data domain
        blood_analysis = self.analyze_blood_biomarkers()
        retinal_analysis = self.analyze_retinal_data()
        body_analysis = self.analyze_body_composition()
        genetic_analysis = self.analyze_genetic_data()
        demographic_analysis = self.analyze_demographics()
        overlap_analysis = self.calculate_data_overlap()
        
        # Compile comprehensive report
        report = {
            'data_availability': {
                'blood_biomarkers': blood_analysis,
                'retinal_imaging': retinal_analysis,
                'body_composition': body_analysis,
                'genetic_data': genetic_analysis,
                'demographics': demographic_analysis,
                'participant_overlap': overlap_analysis
            },
            'feasibility_assessment': self._assess_feasibility(
                blood_analysis, retinal_analysis, body_analysis, 
                genetic_analysis, demographic_analysis, overlap_analysis
            )
        }
        
        return report
    
    def _assess_feasibility(self, blood, retinal, body, genetic, demo, overlap):
        """Assess feasibility for different algorithmic approaches"""
        
        # Calculate coverage percentages
        total_participants = overlap['total_participants']
        
        coverage = {
            'blood_biomarkers': (blood['blood_participants'] / total_participants) * 100,
            'metabolomics': (blood['nmr_participants'] / total_participants) * 100,
            'retinal_imaging': (retinal['oct_participants'] / total_participants) * 100,
            'genetic_data': (genetic['prs_participants'] / total_participants) * 100,
            'brain_imaging': (overlap['imaging_coverage']['brain'] / total_participants) * 100,
            'dxa_imaging': (overlap['imaging_coverage']['dxa'] / total_participants) * 100
        }
        
        # Assess feasibility for different approaches
        feasibility = {
            'coverage_percentages': coverage,
            'approach_feasibility': {},
            'sample_size_estimates': {},
            'recommendations': []
        }
        
        # Blood biomarker approach
        if coverage['blood_biomarkers'] > 80:
            feasibility['approach_feasibility']['blood_biomarker_only'] = 'HIGH'
            feasibility['sample_size_estimates']['blood_biomarker_only'] = blood['blood_participants']
            feasibility['recommendations'].append(
                "Blood biomarker approach highly feasible with >80% coverage"
            )
        elif coverage['blood_biomarkers'] > 60:
            feasibility['approach_feasibility']['blood_biomarker_only'] = 'MEDIUM'
            feasibility['sample_size_estimates']['blood_biomarker_only'] = blood['blood_participants']
            feasibility['recommendations'].append(
                "Blood biomarker approach feasible but requires missing data handling"
            )
        
        # Multi-modal approach
        multi_modal_min = min(
            blood['blood_participants'],
            retinal['oct_participants'],
            genetic['prs_participants']
        )
        multi_modal_coverage = (multi_modal_min / total_participants) * 100
        
        if multi_modal_coverage > 10:
            feasibility['approach_feasibility']['multi_modal'] = 'MEDIUM'
            feasibility['sample_size_estimates']['multi_modal'] = multi_modal_min
            feasibility['recommendations'].append(
                f"Multi-modal approach feasible with {multi_modal_coverage:.1f}% coverage ({multi_modal_min:,} participants)"
            )
        elif multi_modal_coverage > 5:
            feasibility['approach_feasibility']['multi_modal'] = 'LOW'
            feasibility['sample_size_estimates']['multi_modal'] = multi_modal_min
            feasibility['recommendations'].append(
                f"Multi-modal approach limited by sample size ({multi_modal_min:,} participants)"
            )
        
        # NMR metabolomics approach
        if coverage['metabolomics'] > 80:
            feasibility['approach_feasibility']['metabolomics'] = 'HIGH'
            feasibility['recommendations'].append(
                "NMR metabolomics provides comprehensive metabolic profiling"
            )
        
        # Retinal imaging approach
        if coverage['retinal_imaging'] > 10:
            feasibility['approach_feasibility']['retinal_imaging'] = 'MEDIUM'
            feasibility['recommendations'].append(
                f"Retinal imaging viable for {coverage['retinal_imaging']:.1f}% of cohort"
            )
        
        return feasibility

def main():
    """Main analysis function"""
    ukbb_path = "D:\\NUS-AI-Scientist\\UKBB"
    
    print("UK Biobank Biological Age Feasibility Analysis")
    print("=" * 50)
    
    analyzer = UKBBFeasibilityAnalyzer(ukbb_path)
    report = analyzer.generate_feasibility_report()
    
    # Print summary results
    print("\n" + "=" * 50)
    print("FEASIBILITY ANALYSIS SUMMARY")
    print("=" * 50)
    
    # Data availability summary
    availability = report['data_availability']
    
    print(f"\n1. PARTICIPANT COVERAGE:")
    print(f"   Total participants: {availability['participant_overlap']['total_participants']:,}")
    print(f"   Blood biomarkers: {availability['blood_biomarkers']['blood_participants']:,} "
          f"({availability['blood_biomarkers']['blood_participants']/availability['participant_overlap']['total_participants']*100:.1f}%)")
    print(f"   NMR metabolomics: {availability['blood_biomarkers']['nmr_participants']:,} records")
    print(f"   Retinal imaging (OCT): {availability['retinal_imaging']['oct_participants']:,} "
          f"({availability['retinal_imaging']['oct_participants']/availability['participant_overlap']['total_participants']*100:.1f}%)")
    print(f"   Genetic data: {availability['genetic_data']['prs_participants']:,} "
          f"({availability['genetic_data']['prs_participants']/availability['participant_overlap']['total_participants']*100:.1f}%)")
    print(f"   Brain imaging: {availability['participant_overlap']['imaging_coverage']['brain']:,} "
          f"({availability['participant_overlap']['imaging_coverage']['brain']/availability['participant_overlap']['total_participants']*100:.1f}%)")
    print(f"   DXA imaging: {availability['participant_overlap']['imaging_coverage']['dxa']:,} "
          f"({availability['participant_overlap']['imaging_coverage']['dxa']/availability['participant_overlap']['total_participants']*100:.1f}%)")
    
    print(f"\n2. BIOMARKER AVAILABILITY:")
    
    # Blood biomarkers found
    blood_found = list(availability['blood_biomarkers']['blood_biomarkers'].keys())
    nmr_found = list(availability['blood_biomarkers']['nmr_biomarkers'].keys())
    
    print(f"   Blood biomarkers identified: {len(blood_found)}")
    for biomarker in blood_found[:10]:  # Show first 10
        count = len(availability['blood_biomarkers']['blood_biomarkers'][biomarker])
        print(f"     - {biomarker}: {count} measurement(s)")
    
    print(f"   NMR metabolomics identified: {len(nmr_found)}")
    for biomarker in nmr_found[:10]:  # Show first 10
        count = len(availability['blood_biomarkers']['nmr_biomarkers'][biomarker])
        print(f"     - {biomarker}: {count} measurement(s)")
    
    # Retinal biomarkers
    retinal_found = list(availability['retinal_imaging']['retinal_biomarkers'].keys())
    print(f"   Retinal biomarkers identified: {len(retinal_found)}")
    for biomarker in retinal_found:
        count = len(availability['retinal_imaging']['retinal_biomarkers'][biomarker])
        print(f"     - {biomarker}: {count} measurement(s)")
    
    print(f"\n3. FEASIBILITY ASSESSMENT:")
    feasibility = report['feasibility_assessment']
    
    for approach, level in feasibility['approach_feasibility'].items():
        sample_size = feasibility['sample_size_estimates'].get(approach, 'N/A')
        if isinstance(sample_size, int):
            sample_size = f"{sample_size:,}"
        print(f"   {approach.replace('_', ' ').title()}: {level} (Sample: {sample_size})")
    
    print(f"\n4. COVERAGE PERCENTAGES:")
    for modality, percentage in feasibility['coverage_percentages'].items():
        print(f"   {modality.replace('_', ' ').title()}: {percentage:.1f}%")
    
    print(f"\n5. KEY RECOMMENDATIONS:")
    for i, rec in enumerate(feasibility['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    # Save detailed report
    import json
    with open("ukbb_feasibility_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: ukbb_feasibility_report.json")

if __name__ == "__main__":
    main()