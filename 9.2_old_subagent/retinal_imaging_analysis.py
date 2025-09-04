#!/usr/bin/env python3
"""
UK Biobank Retinal Imaging Data Deep Analysis
Detailed exploration of retinal imaging data structure and participant coverage
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Set
import re

class RetinalImagingAnalyzer:
    """Analyze UK Biobank retinal imaging data structure and availability"""
    
    def __init__(self, retinal_dir: str = "/mnt/data1/UKBB_retinal_img/UKB_new_2024",
                 ukbb_dir: str = "/mnt/data1/UKBB"):
        self.retinal_dir = retinal_dir
        self.ukbb_dir = ukbb_dir
        self.imaging_metadata = {}
        
    def analyze_retinal_structure(self):
        """Analyze the structure of retinal imaging directories"""
        
        print("=" * 80)
        print("RETINAL IMAGING DATA STRUCTURE ANALYSIS")
        print("=" * 80)
        
        # Get all directories
        try:
            dirs = os.listdir(self.retinal_dir)
            print(f"\nTotal directories found: {len(dirs)}")
            
            # Parse directory naming convention
            dir_patterns = {}
            for d in dirs:
                # Pattern appears to be: fieldID_instance_index
                parts = d.split('_')
                if len(parts) >= 2:
                    field_id = parts[0]
                    if field_id not in dir_patterns:
                        dir_patterns[field_id] = []
                    dir_patterns[field_id].append(d)
            
            print("\nDirectory structure by field ID:")
            for field_id, dir_list in sorted(dir_patterns.items()):
                print(f"\nField {field_id}:")
                print(f"  Total directories: {len(dir_list)}")
                
                # Sample one directory to check contents
                sample_dir = dir_list[0]
                sample_path = os.path.join(self.retinal_dir, sample_dir)
                
                if os.path.isdir(sample_path):
                    files = os.listdir(sample_path)
                    print(f"  Sample directory '{sample_dir}':")
                    print(f"    Total files: {len(files)}")
                    
                    # Analyze file types
                    file_types = {}
                    for f in files[:100]:  # Sample first 100 files
                        ext = os.path.splitext(f)[1].lower()
                        file_types[ext] = file_types.get(ext, 0) + 1
                    
                    print("    File types (sample):")
                    for ext, count in sorted(file_types.items()):
                        print(f"      {ext if ext else 'no extension'}: {count}")
                    
                    # Check file naming pattern
                    if files:
                        print(f"    Sample filenames:")
                        for f in files[:5]:
                            print(f"      - {f}")
                            
            # Field ID mapping (based on UK Biobank documentation)
            field_mapping = {
                '21015': 'Fundus camera image (left eye)',
                '21016': 'Fundus camera image (right eye)'
            }
            
            print("\n" + "-" * 40)
            print("FIELD ID INTERPRETATION:")
            for field_id, description in field_mapping.items():
                if field_id in dir_patterns:
                    print(f"  {field_id}: {description}")
                    print(f"    Available instances: {len(dir_patterns[field_id])}")
                    
        except Exception as e:
            print(f"Error analyzing retinal structure: {e}")
            
    def extract_participant_ids(self):
        """Extract participant IDs from retinal imaging files"""
        
        print("\n" + "=" * 80)
        print("EXTRACTING PARTICIPANT IDs FROM RETINAL IMAGING")
        print("=" * 80)
        
        participant_ids = set()
        field_participant_map = {}
        
        try:
            dirs = os.listdir(self.retinal_dir)
            
            for dir_name in dirs:
                dir_path = os.path.join(self.retinal_dir, dir_name)
                if os.path.isdir(dir_path):
                    field_id = dir_name.split('_')[0]
                    
                    if field_id not in field_participant_map:
                        field_participant_map[field_id] = set()
                    
                    # Extract participant IDs from filenames
                    files = os.listdir(dir_path)
                    for filename in files:
                        # Typical pattern: participantID_fieldID_instance_index.ext
                        match = re.match(r'^(\d+)_', filename)
                        if match:
                            pid = int(match.group(1))
                            participant_ids.add(pid)
                            field_participant_map[field_id].add(pid)
                            
            print(f"\nTotal unique participants with retinal imaging: {len(participant_ids):,}")
            
            print("\nParticipants by imaging type:")
            for field_id, pids in sorted(field_participant_map.items()):
                print(f"  Field {field_id}: {len(pids):,} participants")
                
            # Check for participants with both eyes
            if '21015' in field_participant_map and '21016' in field_participant_map:
                both_eyes = field_participant_map['21015'] & field_participant_map['21016']
                print(f"\nParticipants with BOTH eyes imaged: {len(both_eyes):,}")
                
            return participant_ids
            
        except Exception as e:
            print(f"Error extracting participant IDs: {e}")
            return set()
            
    def cross_reference_with_oct(self, retinal_participants):
        """Cross-reference retinal imaging participants with OCT data"""
        
        print("\n" + "=" * 80)
        print("CROSS-REFERENCING RETINAL IMAGING WITH OCT DATA")
        print("=" * 80)
        
        try:
            # Load OCT participant IDs
            oct_df = pd.read_csv(f"{self.ukbb_dir}/ukb_OCT.csv", usecols=['f.eid'])
            oct_participants = set(oct_df['f.eid'].dropna())
            
            print(f"\nTotal OCT participants: {len(oct_participants):,}")
            print(f"Total retinal imaging participants: {len(retinal_participants):,}")
            
            # Find intersection
            both_modalities = retinal_participants & oct_participants
            print(f"\nParticipants with BOTH retinal fundus AND OCT: {len(both_modalities):,}")
            
            # Calculate coverage
            if retinal_participants:
                coverage = 100 * len(both_modalities) / len(retinal_participants)
                print(f"OCT coverage in retinal imaging cohort: {coverage:.1f}%")
                
            return both_modalities
            
        except Exception as e:
            print(f"Error cross-referencing with OCT: {e}")
            return set()
            
    def generate_retinal_summary(self, retinal_participants):
        """Generate comprehensive summary of retinal imaging availability"""
        
        print("\n" + "=" * 80)
        print("RETINAL IMAGING FEATURE SUMMARY")
        print("=" * 80)
        
        summary = {
            'total_participants': len(retinal_participants),
            'imaging_types': {},
            'quality_metrics': {},
            'temporal_coverage': {}
        }
        
        print("\nKEY RETINAL FEATURES FOR BIOLOGICAL AGE:")
        print("-" * 40)
        
        retinal_features = [
            "Vessel tortuosity",
            "Fractal dimension", 
            "Arteriovenous ratio (AVR)",
            "Vessel caliber (CRAE/CRVE)",
            "Optic disc parameters",
            "Retinal age gap",
            "Macular thickness (if OCT available)",
            "RNFL thickness (if OCT available)"
        ]
        
        for i, feature in enumerate(retinal_features, 1):
            availability = "Available (fundus)" if i <= 6 else "Requires OCT"
            print(f"  {i}. {feature}: {availability}")
            
        print("\nRETINAL AGE ALGORITHM REQUIREMENTS:")
        print("-" * 40)
        print("  • High-quality fundus photographs")
        print("  • Both eyes preferred for reliability")
        print("  • Vessel segmentation capability")
        print("  • Age-matched reference population")
        
        return summary
        
    def run_analysis(self):
        """Run complete retinal imaging analysis"""
        
        # Analyze structure
        self.analyze_retinal_structure()
        
        # Extract participant IDs
        retinal_participants = self.extract_participant_ids()
        
        # Cross-reference with OCT
        if retinal_participants:
            oct_overlap = self.cross_reference_with_oct(retinal_participants)
        
        # Generate summary
        summary = self.generate_retinal_summary(retinal_participants)
        
        return retinal_participants

if __name__ == "__main__":
    analyzer = RetinalImagingAnalyzer()
    retinal_participants = analyzer.run_analysis()