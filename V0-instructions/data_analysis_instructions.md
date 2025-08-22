# Data Analysis Instructions - UK Biobank Multimodal Data Exploration

## Objective
Conduct comprehensive exploratory data analysis on the UK Biobank multimodal dataset to provide data insights and preprocessing strategies for model design.

## Data Path Configuration
- **Tabular Data Root Directory**: `/mnt/data1/UKBB/`
- **Fundus Images Directory**: `/mnt/data1/UKBB_retinal_img/`
- **Memory Optimization**: All CSV files are large, must use chunked reading or memory mapping

## Tabular Data Analysis Tasks

### 1. Data Overview and Quality Assessment
**Target Files**: All CSV files
**Analysis Content**:
- Shape and memory usage of each file
- Overlap through f.eid column to determine mergeable sample count
- Missing value pattern analysis (by column and by row)
- Data type checking and outlier detection

**Output**: `data_overview_report.html`

### 2. Demographic Analysis
**Target Files**: `ukb_main.csv`, `ukb_bodysize.csv`
**Key Variables**: Age, Sex, Ethnicity, BMI, Height, Weight, Education, Income
**Analysis Content**:
- Age distribution and mortality relationship
- Gender and ethnicity distribution
- BMI distribution and outlier handling
- Correlations among demographic variables

**Special Focus**: 
- Relationship between age and mortality labels (age_at_mortality, year_to_mortality)
- Create ground truth definition for biological age

### 3. Biomarker Analysis
**Target Files**: `ukb_blood.csv`, `ukb_NMR.csv`
**Blood Biochemistry**: Albumin, CRP, Glucose, LDL, HDL, Creatinine, Triglycerides, HbA1c, Urea, Vitamin D, IGF-1
**NMR Metabolomics**: Total Cholesterol, Omega-3, MUFA, PUFA, Alanine, Leucine
**Analysis Content**:
- Distribution characteristics of each biomarker
- Age-related biomarker change trends
- Correlation network analysis among biomarkers
- Literature-based identification of aging-related biomarkers

### 4. Body Composition and Function Analysis
**Target Files**: `ukb_body.csv`, `ukb_frailty.csv`
**Body Composition**: Android/gynoid fat, VAT, total fat/lean mass, BMD
**Functional Indicators**: Grip strength, Physical activity, Falls, Self-rated health
**Analysis Content**:
- Changes in body composition with age
- Distribution and correlations of frailty indicators
- Pattern recognition of functional decline

### 5. Neuroimaging Analysis
**Target Files**: `ukb_brain.csv`, `ukb_OCT.csv`
**Brain Indicators**: Total brain volume, Ventricle volume, WMH volume, Hippocampus T2*
**Ocular Indicators**: Macular thickness, RNFL thickness, Cup-to-disc ratio
**Analysis Content**:
- Relationship between brain volume and age
- White matter hyperintensities and aging
- Age-related changes in retinal structural parameters

## Image Data Analysis Tasks

### 6. Fundus Image Quality Assessment
**Target Directory**: `/mnt/data1/UKBB_retinal_img/`
**Image Naming**: `{eid}_21015_2_1.jpg`
**Analysis Content**:
- Image count statistics (images per participant)
- Image quality assessment (resolution confirmation, blur detection)
- Image pixel value distribution analysis
- Statistics of participants with missing images

**Technical Requirements**:
- Use batch processing to avoid memory overflow
- Sample image visualization
- Image quality grading

## Multimodal Data Integration Analysis

### 7. Cross-Modal Correlation Analysis
**Analysis Content**:
- Correlation ranking of tabular features with age
- Information redundancy analysis across modalities
- Cross-modal missing data pattern analysis
- Sample completeness analysis (number of samples with complete multimodal data)

### 8. Aging Pattern Recognition
**Analysis Content**:
- Clustering-based aging subtype discovery
- Feature change patterns across age groups
- Feature differences between rapid aging vs healthy aging
- Key features related to mortality prediction

## Data Preprocessing Strategy Design

### 9. Missing Value Handling Strategy
**For Different Data Types**:
- Continuous variables: Interpolation method selection
- Categorical variables: Mode or new category
- Image data: Availability marking
- Time series: Forward fill or trend interpolation

### 10. Feature Engineering Recommendations
**Based on EDA Results**:
- Identification of features requiring standardization
- Features needing log transformation
- Categorical variable encoding strategies
- Feature interaction term creation suggestions

## Output Requirements

### Primary Output Files:
1. **comprehensive_eda_report.html**: Complete exploratory data analysis report
2. **data_quality_summary.json**: Data quality metrics summary
3. **correlation_matrix.png**: Cross-modal correlation heatmap
4. **age_biomarker_trends.png**: Biomarker-age relationship visualization
5. **sample_completeness_analysis.csv**: Sample completeness statistics
6. **preprocessing_recommendations.md**: Data preprocessing recommendations

### Technical Requirements:
- **Memory Optimization**: Use `pd.read_csv(chunksize=)` or `dask` for large files
- **Parallel Processing**: Use multiprocessing for image analysis
- **Visualization**: Use seaborn, matplotlib, plotly to create interactive charts
- **Statistical Testing**: Perform appropriate significance tests

## Completion Criteria
- Generate complete data quality report
- Identify key aging-related features
- Provide specific preprocessing pipeline recommendations
- Determine final modeling sample count and feature count
- Provide data-driven insights for next stage model design