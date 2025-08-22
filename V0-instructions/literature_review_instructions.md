# Literature Review Instructions - Multimodal Machine Learning for Biological Age Prediction

## Objective
Conduct comprehensive literature research for a biological age prediction project based on UK Biobank multimodal data to guide feature selection and model design.

## Search Tasks

### 1. Biological Age Prediction Fundamentals
**Search Keywords**: "biological age prediction", "aging biomarkers", "epigenetic clock", "phenotypic age"
**Key Focus Areas**:
- Definitions and measurement methods of biological age
- Classic biological age prediction models (Horvath clock, Hannum clock, PhenoAge, etc.)
- Associations between biological age and disease risk, mortality

### 2. UK Biobank Related Research
**Search Keywords**: "UK Biobank biological age", "UK Biobank aging", "multimodal aging prediction"
**Key Focus Areas**:
- Classic papers using UK Biobank data for aging research
- Validated biological age-related biomarkers
- Best practices for UK Biobank data processing

### 3. Multimodal Machine Learning in Healthcare
**Search Keywords**: "multimodal machine learning healthcare", "medical data fusion", "tabular image fusion"
**Key Focus Areas**:
- Methods for fusing tabular and image data
- Multimodal feature fusion strategies (early fusion vs late fusion)
- Training strategies and evaluation methods for multimodal models

### 4. Retinal Fundus Images and Aging Research
**Search Keywords**: "retinal fundus aging", "retinal biomarkers aging", "fundus image biological age"
**Key Focus Areas**:
- Relationship between retinal vasculature and aging
- Aging-related features in fundus images
- Applications of pretrained models (RetiZero, RETFound, etc.)

### 5. Modality-Specific Features and Aging Associations
**Search by Data Type**:
- **Blood Biochemistry**: "blood biomarkers aging", "CRP aging", "glucose aging", "lipid profile aging"
- **NMR Metabolomics**: "NMR metabolomics aging", "lipid metabolism aging", "amino acid aging"
- **Body Composition**: "DXA aging", "visceral fat aging", "muscle mass aging"
- **Brain MRI**: "brain MRI aging", "brain volume aging", "white matter aging"
- **OCT**: "OCT retinal aging", "macular thickness aging", "RNFL aging"
- **Frailty & Mental Health**: "frailty aging", "grip strength aging", "depression aging"

## Output Requirements

### Primary Output Files:
1. **literature_summary.md**: 
   - Core findings summary for each search topic
   - Important paper list (at least 50 high-quality publications)
   - Key findings in citation format

2. **feature_recommendations.md**:
   - Literature-based key feature recommendations for each modality
   - Feature importance ranking and biomedical interpretation
   - Feature preprocessing suggestions

3. **model_architecture_review.md**:
   - Literature review of multimodal fusion architectures
   - Pros and cons analysis of different fusion strategies
   - Model architecture recommendations suitable for this project

4. **evaluation_metrics_review.md**:
   - Evaluation metrics for biological age prediction models
   - Model interpretability and biomedical significance assessment methods

## Search Strategy
- **Databases**: PubMed, Google Scholar, ArXiv, bioRxiv
- **Time Range**: Focus on 2020-2024 latest research, classic literature can trace back to 2015
- **Paper Selection**: Prioritize high-impact factor journals, studies with sample size >1000
- **Citation Network**: Track citation relationships of important papers to ensure core research coverage

## Special Considerations
1. **Biomedical Interpretability**: Focus on features and models that can explain aging mechanisms
2. **Clinical Translation Potential**: Focus on findings with clinical application value
3. **Technical Feasibility**: Ensure recommended methods can be implemented with modern Python tools
4. **Data Quality**: Focus on best practices for handling missing values and data quality control

## Completion Criteria
- Search and analyze at least 50 high-quality papers
- Form complete feature selection guidance
- Provide 3-5 candidate model architectures
- Establish complete evaluation framework