#!/usr/bin/env python3
"""
UK Biobank Population Characteristics Analysis
Analyzes demographic diversity and population structure for biological age research
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

class UKBBPopulationAnalyzer:
    def __init__(self, ukbb_path):
        self.ukbb_path = Path(ukbb_path)
        
    def load_demographic_data(self):
        """Load key demographic variables from main file"""
        print("Loading demographic data...")
        
        # Key demographic columns to load
        demo_cols = [
            'f.eid', 'Sex.0', 'Birth_year.0', 'Birth_month.0', 
            'Age', 'Ethnicity',
            'Townsend_deprivation_index_at_recruitment.0'
        ]
        
        file_path = self.ukbb_path / 'ukb_main.csv'
        
        # Load only the columns we need
        try:
            df = pd.read_csv(file_path, usecols=demo_cols)
            print(f"Loaded demographic data for {len(df)} participants")
            return df
        except Exception as e:
            print(f"Error loading demographic data: {e}")
            return None
    
    def analyze_age_distribution(self, df):
        """Analyze age distribution of participants"""
        print("Analyzing age distribution...")
        
        age_col = 'Age'
        if age_col not in df.columns:
            print(f"Age column {age_col} not found")
            return {}
        
        # Remove missing values
        age_data = df[age_col].dropna()
        
        age_analysis = {
            'total_participants': len(age_data),
            'mean_age': age_data.mean(),
            'median_age': age_data.median(),
            'std_age': age_data.std(),
            'min_age': age_data.min(),
            'max_age': age_data.max(),
            'age_distribution': {
                '18-30': len(age_data[(age_data >= 18) & (age_data < 30)]),
                '30-40': len(age_data[(age_data >= 30) & (age_data < 40)]),
                '40-50': len(age_data[(age_data >= 40) & (age_data < 50)]),
                '50-60': len(age_data[(age_data >= 50) & (age_data < 60)]),
                '60-70': len(age_data[(age_data >= 60) & (age_data < 70)]),
                '70-80': len(age_data[(age_data >= 70) & (age_data < 80)]),
                '80+': len(age_data[age_data >= 80])
            }
        }
        
        return age_analysis
    
    def analyze_sex_distribution(self, df):
        """Analyze sex distribution"""
        print("Analyzing sex distribution...")
        
        sex_col = 'Sex.0'
        if sex_col not in df.columns:
            print(f"Sex column {sex_col} not found")
            return {}
        
        sex_data = df[sex_col].dropna()
        
        # Convert to meaningful labels (assuming 1=Male, 0=Female based on typical UKBB coding)
        sex_counts = sex_data.value_counts()
        sex_analysis = {
            'total_with_sex_data': len(sex_data),
            'male_count': int(sex_counts.get(1, 0)),
            'female_count': int(sex_counts.get(0, 0)),
            'male_percentage': (sex_counts.get(1, 0) / len(sex_data)) * 100,
            'female_percentage': (sex_counts.get(0, 0) / len(sex_data)) * 100
        }
        
        return sex_analysis
    
    def analyze_ethnic_distribution(self, df):
        """Analyze ethnic diversity"""
        print("Analyzing ethnic distribution...")
        
        ethnic_col = 'Ethnicity'
        if ethnic_col not in df.columns:
            print(f"Ethnic column {ethnic_col} not found")
            return {}
        
        ethnic_data = df[ethnic_col].dropna()
        
        # Count unique ethnic categories
        ethnic_counts = ethnic_data.value_counts()
        
        ethnic_analysis = {
            'total_with_ethnic_data': len(ethnic_data),
            'unique_ethnic_categories': len(ethnic_counts),
            'ethnic_distribution': ethnic_counts.to_dict(),
            'ethnic_percentages': (ethnic_counts / len(ethnic_data) * 100).round(2).to_dict()
        }
        
        return ethnic_analysis
    
    def analyze_socioeconomic_distribution(self, df):
        """Analyze socioeconomic indicators"""
        print("Analyzing socioeconomic distribution...")
        
        townsend_col = 'Townsend_deprivation_index_at_recruitment.0'
        if townsend_col not in df.columns:
            print(f"Townsend column {townsend_col} not found")
            return {}
        
        townsend_data = df[townsend_col].dropna()
        
        townsend_analysis = {
            'total_with_townsend_data': len(townsend_data),
            'mean_townsend': townsend_data.mean(),
            'median_townsend': townsend_data.median(),
            'std_townsend': townsend_data.std(),
            'min_townsend': townsend_data.min(),
            'max_townsend': townsend_data.max(),
            'deprivation_quintiles': {
                'least_deprived (Q1)': len(townsend_data[townsend_data <= townsend_data.quantile(0.2)]),
                'Q2': len(townsend_data[(townsend_data > townsend_data.quantile(0.2)) & 
                                     (townsend_data <= townsend_data.quantile(0.4))]),
                'Q3': len(townsend_data[(townsend_data > townsend_data.quantile(0.4)) & 
                                     (townsend_data <= townsend_data.quantile(0.6))]),
                'Q4': len(townsend_data[(townsend_data > townsend_data.quantile(0.6)) & 
                                     (townsend_data <= townsend_data.quantile(0.8))]),
                'most_deprived (Q5)': len(townsend_data[townsend_data > townsend_data.quantile(0.8)])
            }
        }
        
        return townsend_analysis
    
    def calculate_data_modality_overlap(self):
        """Calculate participant overlap across different data modalities"""
        print("Calculating data modality overlap...")
        
        # Load participant IDs from different files
        participant_ids = {}
        
        files_to_check = [
            ('main', 'ukb_main.csv'),
            ('blood', 'ukb_blood.csv'),
            ('nmr', 'ukb_NMR.csv'),
            ('oct', 'ukb_OCT.csv'),
            ('brain', 'ukb_brain.csv'),
            ('prs', 'ukb_PRS.csv'),
            ('dxa', 'ukb20240116_DXA_long_named.csv')
        ]
        
        for modality, filename in files_to_check:
            try:
                file_path = self.ukbb_path / filename
                # Load just the participant ID column
                df = pd.read_csv(file_path, usecols=['f.eid'])
                participant_ids[modality] = set(df['f.eid'].tolist())
                print(f"Loaded {len(participant_ids[modality])} participants from {modality}")
            except Exception as e:
                print(f"Error loading {modality}: {e}")
                participant_ids[modality] = set()
        
        # Calculate overlaps
        overlap_analysis = {}
        
        # Individual modality coverage
        for modality, ids in participant_ids.items():
            overlap_analysis[f'{modality}_coverage'] = len(ids)
        
        # Multi-modal overlaps
        if all(len(ids) > 0 for ids in participant_ids.values()):
            # All modalities intersection
            all_modalities = set.intersection(*participant_ids.values())
            overlap_analysis['all_modalities_intersection'] = len(all_modalities)
            
            # Key combinations
            blood_retinal_genetic = participant_ids['blood'] & participant_ids['oct'] & participant_ids['prs']
            overlap_analysis['blood_retinal_genetic'] = len(blood_retinal_genetic)
            
            blood_genetic = participant_ids['blood'] & participant_ids['prs']
            overlap_analysis['blood_genetic'] = len(blood_genetic)
            
            retinal_genetic = participant_ids['oct'] & participant_ids['prs']
            overlap_analysis['retinal_genetic'] = len(retinal_genetic)
            
            imaging_modalities = participant_ids['oct'] & participant_ids['brain'] & participant_ids['dxa']
            overlap_analysis['all_imaging'] = len(imaging_modalities)
        
        return overlap_analysis, participant_ids
    
    def generate_population_report(self):
        """Generate comprehensive population analysis report"""
        print("Generating population analysis report...")
        
        # Load demographic data
        demo_df = self.load_demographic_data()
        if demo_df is None:
            return None
        
        # Analyze different aspects
        age_analysis = self.analyze_age_distribution(demo_df)
        sex_analysis = self.analyze_sex_distribution(demo_df)
        ethnic_analysis = self.analyze_ethnic_distribution(demo_df)
        socioeconomic_analysis = self.analyze_socioeconomic_distribution(demo_df)
        overlap_analysis, participant_ids = self.calculate_data_modality_overlap()
        
        # Compile comprehensive report
        report = {
            'demographic_analysis': {
                'age_distribution': age_analysis,
                'sex_distribution': sex_analysis,
                'ethnic_distribution': ethnic_analysis,
                'socioeconomic_distribution': socioeconomic_analysis
            },
            'data_modality_overlap': overlap_analysis,
            'methodological_considerations': self._generate_methodological_considerations(
                age_analysis, sex_analysis, ethnic_analysis, overlap_analysis
            )
        }
        
        return report
    
    def _generate_methodological_considerations(self, age_analysis, sex_analysis, ethnic_analysis, overlap_analysis):
        """Generate methodological considerations based on population characteristics"""
        
        considerations = {
            'strengths': [],
            'limitations': [],
            'recommendations': [],
            'bias_assessment': {}
        }
        
        # Age distribution considerations
        if age_analysis:
            age_range = age_analysis.get('max_age', 0) - age_analysis.get('min_age', 0)
            if age_range > 50:
                considerations['strengths'].append("Wide age range supports cross-sectional aging studies")
            
            # Check for middle-aged bias
            middle_aged = (age_analysis.get('age_distribution', {}).get('40-60', 0) + 
                          age_analysis.get('age_distribution', {}).get('60-70', 0))
            total = age_analysis.get('total_participants', 1)
            if (middle_aged / total) > 0.6:
                considerations['limitations'].append("Cohort shows middle-age bias, may limit generalizability to older adults")
                considerations['bias_assessment']['age_bias'] = "Middle-aged overrepresentation"
        
        # Sex distribution considerations
        if sex_analysis:
            sex_ratio = sex_analysis.get('male_percentage', 50) / sex_analysis.get('female_percentage', 50)
            if 0.8 <= sex_ratio <= 1.2:
                considerations['strengths'].append("Balanced sex distribution supports sex-specific analyses")
            else:
                considerations['limitations'].append(f"Sex imbalance (M:F ratio = {sex_ratio:.2f}) may bias findings")
        
        # Ethnic diversity considerations
        if ethnic_analysis:
            diversity_index = ethnic_analysis.get('unique_ethnic_categories', 0)
            if diversity_index > 5:
                considerations['strengths'].append(f"Ethnic diversity with {diversity_index} categories supports cross-population validation")
            else:
                considerations['limitations'].append("Limited ethnic diversity may restrict generalizability")
        
        # Data modality considerations
        total_main = overlap_analysis.get('main_coverage', 0)
        multi_modal_count = overlap_analysis.get('blood_retinal_genetic', 0)
        multi_modal_percentage = (multi_modal_count / total_main) * 100 if total_main > 0 else 0
        
        if multi_modal_percentage > 10:
            considerations['strengths'].append(f"Multi-modal analysis feasible with {multi_modal_percentage:.1f}% coverage")
        else:
            considerations['limitations'].append(f"Limited multi-modal coverage ({multi_modal_percentage:.1f}%) restricts integrated approaches")
        
        # General recommendations
        considerations['recommendations'].extend([
            "Account for age distribution biases in age prediction models",
            "Consider sex-stratified analyses to identify sex-specific aging patterns",
            "Evaluate model performance across different ethnic groups",
            "Use appropriate sampling strategies for multi-modal analyses to avoid selection bias",
            "Consider longitudinal analysis where multiple time points are available"
        ])
        
        return considerations

def main():
    """Main analysis function"""
    ukbb_path = "D:\\NUS-AI-Scientist\\UKBB"
    
    print("UK Biobank Population Characteristics Analysis")
    print("=" * 50)
    
    analyzer = UKBBPopulationAnalyzer(ukbb_path)
    report = analyzer.generate_population_report()
    
    if report is None:
        print("Failed to generate report")
        return
    
    # Print summary results
    print("\n" + "=" * 50)
    print("POPULATION ANALYSIS SUMMARY")
    print("=" * 50)
    
    # Age distribution
    age_analysis = report['demographic_analysis']['age_distribution']
    if age_analysis:
        print(f"\n1. AGE DISTRIBUTION:")
        print(f"   Total participants: {age_analysis.get('total_participants', 0):,}")
        print(f"   Mean age: {age_analysis.get('mean_age', 0):.1f} years")
        print(f"   Age range: {age_analysis.get('min_age', 0):.0f} - {age_analysis.get('max_age', 0):.0f} years")
        print(f"   Age distribution:")
        for age_group, count in age_analysis.get('age_distribution', {}).items():
            percentage = (count / age_analysis.get('total_participants', 1)) * 100
            print(f"     {age_group}: {count:,} ({percentage:.1f}%)")
    
    # Sex distribution
    sex_analysis = report['demographic_analysis']['sex_distribution']
    if sex_analysis:
        print(f"\n2. SEX DISTRIBUTION:")
        print(f"   Total with sex data: {sex_analysis.get('total_with_sex_data', 0):,}")
        print(f"   Male: {sex_analysis.get('male_count', 0):,} ({sex_analysis.get('male_percentage', 0):.1f}%)")
        print(f"   Female: {sex_analysis.get('female_count', 0):,} ({sex_analysis.get('female_percentage', 0):.1f}%)")
    
    # Ethnic distribution
    ethnic_analysis = report['demographic_analysis']['ethnic_distribution']
    if ethnic_analysis:
        print(f"\n3. ETHNIC DIVERSITY:")
        print(f"   Total with ethnic data: {ethnic_analysis.get('total_with_ethnic_data', 0):,}")
        print(f"   Unique ethnic categories: {ethnic_analysis.get('unique_ethnic_categories', 0)}")
        print(f"   Top ethnic groups:")
        ethnic_pct = ethnic_analysis.get('ethnic_percentages', {})
        sorted_ethnic = sorted(ethnic_pct.items(), key=lambda x: x[1], reverse=True)[:5]
        for group, percentage in sorted_ethnic:
            count = ethnic_analysis.get('ethnic_distribution', {}).get(group, 0)
            print(f"     {group}: {count:,} ({percentage:.1f}%)")
    
    # Socioeconomic distribution
    socioeconomic_analysis = report['demographic_analysis']['socioeconomic_distribution']
    if socioeconomic_analysis:
        print(f"\n4. SOCIOECONOMIC DISTRIBUTION:")
        print(f"   Total with Townsend data: {socioeconomic_analysis.get('total_with_townsend_data', 0):,}")
        print(f"   Mean Townsend index: {socioeconomic_analysis.get('mean_townsend', 0):.2f}")
        print(f"   Deprivation quintiles:")
        for quintile, count in socioeconomic_analysis.get('deprivation_quintiles', {}).items():
            total = socioeconomic_analysis.get('total_with_townsend_data', 1)
            percentage = (count / total) * 100
            print(f"     {quintile}: {count:,} ({percentage:.1f}%)")
    
    # Data modality overlap
    overlap_analysis = report['data_modality_overlap']
    print(f"\n5. DATA MODALITY OVERLAP:")
    total_main = overlap_analysis.get('main_coverage', 0)
    
    print(f"   Individual coverage:")
    for modality in ['main', 'blood', 'nmr', 'oct', 'brain', 'prs', 'dxa']:
        count = overlap_analysis.get(f'{modality}_coverage', 0)
        percentage = (count / total_main) * 100 if total_main > 0 else 0
        print(f"     {modality.upper()}: {count:,} ({percentage:.1f}%)")
    
    print(f"   Multi-modal intersections:")
    multi_modal_intersections = [
        ('blood_retinal_genetic', 'Blood + Retinal + Genetic'),
        ('blood_genetic', 'Blood + Genetic'),
        ('retinal_genetic', 'Retinal + Genetic'),
        ('all_imaging', 'All Imaging (OCT + Brain + DXA)')
    ]
    
    for key, label in multi_modal_intersections:
        count = overlap_analysis.get(key, 0)
        percentage = (count / total_main) * 100 if total_main > 0 else 0
        print(f"     {label}: {count:,} ({percentage:.1f}%)")
    
    # Methodological considerations
    considerations = report['methodological_considerations']
    print(f"\n6. METHODOLOGICAL CONSIDERATIONS:")
    
    print(f"   Strengths:")
    for i, strength in enumerate(considerations.get('strengths', []), 1):
        print(f"     {i}. {strength}")
    
    print(f"   Limitations:")
    for i, limitation in enumerate(considerations.get('limitations', []), 1):
        print(f"     {i}. {limitation}")
    
    print(f"   Recommendations:")
    for i, rec in enumerate(considerations.get('recommendations', []), 1):
        print(f"     {i}. {rec}")
    
    # Save detailed report
    with open("ukbb_population_analysis.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDetailed population analysis saved to: ukbb_population_analysis.json")

if __name__ == "__main__":
    main()