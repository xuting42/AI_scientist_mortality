# UK Biobank Data Mapping Report for Biological Age Research

Generated: 2025-09-03 18:13:45


## 1. Dataset Overview


### ukb_blood.csv
- **Total participants**: 604,514
- **Number of variables**: 384
- **File size**: 1046.40 MB

### ukb_NMR.csv
- **Total participants**: 1,004,632
- **Number of variables**: 409
- **File size**: 893.96 MB

### ukb_body.csv
- **Total participants**: 603,556
- **Number of variables**: 222
- **File size**: 2118.40 MB

### ukb_OCT.csv
- **Total participants**: 84,864
- **Number of variables**: 106
- **File size**: 44.20 MB

### ukb_main.csv
- **Total participants**: 606,361
- **Number of variables**: 251
- **File size**: 375.83 MB

### ukb_genetic_PC.csv
- **Total participants**: 502,316
- **Number of variables**: 42
- **File size**: 168.01 MB

## 2. Biomarker Availability Assessment

- **Clinical Blood Biomarkers**: HIGH - ukb_blood.csv contains comprehensive panels
- **NMR Metabolomics**: HIGH - ukb_NMR.csv with 249 metabolites
- **Body Composition**: HIGH - ukb_body.csv with DXA and impedance
- **Retinal Imaging**: MODERATE - OCT measurements available
- **Genetic Data**: HIGH - PRS and genetic PCs available
- **Epigenetic Data**: LIMITED - Check for methylation subset
- **Proteomic Data**: CHECK - May be in separate release

## 3. Multi-Modal Data Availability

- Blood biomarkers: 502,316 participants
- Body composition: 500,739 participants
- NMR metabolomics: 502,316 participants
- OCT measurements: 84,402 participants
- Main phenotypes: 606,361 participants
- Genetic PCs: 502,316 participants
- Polygenic risk scores: 502,182 participants

## 4. Priority Recommendations


### Highest Priority (Maximum Data Availability):
1. **Blood biomarkers + NMR metabolomics + Body composition**
   - Most comprehensive coverage
   - Aligns with KDM and phenotypic age approaches

2. **Multi-organ approach using blood + body + cardiovascular**
   - Captures systemic aging

### Innovative Opportunities:
1. **Retinal aging signature + systemic biomarkers**
   - Novel multi-modal approach
   - OCT data provides unique structural information

2. **Metabolomic aging clock**
   - 249 NMR metabolites available
   - Can replicate MetaboAge approaches

## 5. Data Quality Considerations

- Multiple instance data available (baseline + follow-up)
- Consider longitudinal analysis opportunities
- Check for batch effects in NMR and proteomics
- Verify imaging quality metrics for retinal data