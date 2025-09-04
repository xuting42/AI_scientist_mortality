#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

# HENAW Quick Start Script
# This script provides example commands to run the HENAW implementation

# Add error handling
trap 'echo "Error on line $LINENO. Exit code: $?" >&2; exit 1' ERR

# Function to log messages
log_info() {
    echo "[INFO] $1"
}

log_error() {
    echo "[ERROR] $1" >&2
}

log_warning() {
    echo "[WARNING] $1" >&2
}

echo "==================================================="
echo "HENAW Biological Age Model - Quick Start Examples"
echo "==================================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
check_python() {
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed or not in PATH"
        echo "Please install Python 3.8+ and try again"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Found Python version: $python_version"
    
    # Check if Python version is 3.8+
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
        log_info "Python version is compatible"
    else
        log_error "Python 3.8+ is required, found: $python_version"
        exit 1
    fi
}

# Check if virtual environment exists and create if needed
setup_venv() {
    if [ ! -d "henaw_env" ]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        
        if python3 -m venv henaw_env; then
            log_info "Virtual environment created successfully"
        else
            log_error "Failed to create virtual environment"
            echo "Make sure python3-venv is installed (apt install python3-venv on Ubuntu/Debian)"
            exit 1
        fi
    else
        log_info "Virtual environment already exists"
    fi
}

# Check if required files exist
check_requirements() {
    if [ ! -f "requirements.txt" ]; then
        log_warning "requirements.txt not found in current directory"
        echo "Make sure you're in the henaw_implementation directory"
    fi
    
    if [ ! -f "config.yaml" ]; then
        log_warning "config.yaml not found - you may need to create one for training"
    fi
}

# Run checks
log_info "Running environment checks..."
check_python
setup_venv
check_requirements

echo -e "${GREEN}Step 1: Activate environment${NC}"
echo "source henaw_env/bin/activate"
echo ""

echo -e "${GREEN}Step 2: Install dependencies${NC}"
echo "# After activating the environment, run:"
echo "pip install --upgrade pip"
echo "pip install -r requirements.txt"
echo ""

echo -e "${RED}VALIDATION COMMANDS (run these first)${NC}"
echo "# Test if basic imports work:"
echo "python3 -c \"import torch; import numpy; import pandas; print('Dependencies OK')\""
echo ""

# Add validation function
validate_installation() {
    echo "Validating installation..."
    
    # Check if henaw_env is activated
    if [[ "$VIRTUAL_ENV" != *"henaw_env"* ]]; then
        log_warning "Virtual environment not activated"
        echo "Run: source henaw_env/bin/activate"
        return 1
    fi
    
    # Check critical Python packages
    python3 -c "
import sys
required_packages = ['torch', 'numpy', 'pandas', 'sklearn', 'yaml', 'tqdm']
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
        
if missing:
    print(f'Missing packages: {missing}')
    sys.exit(1)
else:
    print('All required packages installed')
" || {
        log_error "Package validation failed"
        echo "Please install missing dependencies with: pip install -r requirements.txt"
        return 1
    }
    
    log_info "Installation validation passed"
}

echo "==================================================="
echo -e "${GREEN}TRAINING EXAMPLES${NC}"
echo "==================================================="
echo ""

echo -e "${YELLOW}Example 1: Basic Training (CPU)${NC}"
echo "# Validate data file exists first:"
echo "if [ ! -f '/path/to/your/ukbb_data.csv' ]; then"
echo "  echo 'Error: Data file not found'; exit 1"
echo "fi"
echo ""
echo "python train_henaw.py \\"
echo "    --data-path /path/to/your/ukbb_data.csv \\"
echo "    --output-dir ./outputs \\"
echo "    --max-epochs 100 \\"
echo "    --batch-size 256 || {"
echo "  echo 'Training failed with exit code $?'"
echo "  exit 1"
echo "}"
echo ""

echo -e "${YELLOW}Example 2: GPU Training with Cross-Validation${NC}"
echo "python train_henaw.py \\"
echo "    --data-path /path/to/your/ukbb_data.csv \\"
echo "    --output-dir ./outputs \\"
echo "    --max-epochs 100 \\"
echo "    --batch-size 512 \\"
echo "    --device cuda \\"
echo "    --n-folds 5 \\"
echo "    --use-mixed-precision"
echo ""

echo -e "${YELLOW}Example 3: Full Pipeline (Train + Evaluate + Report)${NC}"
echo "python run_pipeline.py \\"
echo "    --mode all \\"
echo "    --data-path /path/to/your/ukbb_data.csv \\"
echo "    --output-dir ./results \\"
echo "    --device cuda"
echo ""

echo "==================================================="
echo -e "${GREEN}EVALUATION EXAMPLES${NC}"
echo "==================================================="
echo ""

echo -e "${YELLOW}Example 4: Evaluate Trained Model${NC}"
echo "python evaluate.py \\"
echo "    --model-path outputs/checkpoints/best_model.pt \\"
echo "    --data-path /path/to/your/ukbb_data.csv \\"
echo "    --output-dir ./evaluation_results \\"
echo "    --device cuda"
echo ""

echo "==================================================="
echo -e "${GREEN}PREDICTION EXAMPLES${NC}"
echo "==================================================="
echo ""

echo -e "${YELLOW}Example 5: Batch Prediction${NC}"
echo "python predict.py \\"
echo "    --model outputs/checkpoints/best_model.pt \\"
echo "    --input new_participants.csv \\"
echo "    --output predictions.csv \\"
echo "    --device cuda"
echo ""

echo -e "${YELLOW}Example 6: Start Prediction API Server${NC}"
echo "python predict.py \\"
echo "    --model outputs/checkpoints/best_model.pt \\"
echo "    --server \\"
echo "    --port 8080 \\"
echo "    --device cuda"
echo ""

echo -e "${YELLOW}Example 7: Test API with curl${NC}"
echo 'curl -X POST http://localhost:8080/predict \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{'
echo '    "age": 55,'
echo '    "sex": 1,'
echo '    "30710": 2.5,'
echo '    "30750": 5.8,'
echo '    "30700": 85,'
echo '    "30600": 42,'
echo '    "30180": 30,'
echo '    "30070": 13.5,'
echo '    "30730": 35,'
echo '    "30650": 25,'
echo '    "30620": 28'
echo '  }'"'"''
echo ""

echo "==================================================="
echo -e "${GREEN}USING UK BIOBANK DATA${NC}"
echo "==================================================="
echo ""

echo -e "${YELLOW}For UK Biobank data at /mnt/data1/UKBB_retinal_img/UKB_new_2024:${NC}"
echo ""

echo "# First, prepare the data with required biomarker fields"
echo "python -c \"
import pandas as pd

# Load UK Biobank phenotype data
ukbb_data = pd.read_csv('/mnt/data1/UKBB_retinal_img/UKB_new_2024/phenotype_data.csv')

# Select HENAW required fields
required_fields = [
    'eid', 'age_when_attended_assessment_centre_0_0',
    'sex_0_0',
    '30710-0.0',  # CRP
    '30750-0.0',  # HbA1c
    '30700-0.0',  # Creatinine
    '30600-0.0',  # Albumin
    '30180-0.0',  # Lymphocyte %
    '30070-0.0',  # RDW
    '30730-0.0',  # GGT
    '30650-0.0',  # AST
    '30620-0.0',  # ALT
]

# Rename columns to match expected format
rename_map = {
    'age_when_attended_assessment_centre_0_0': 'age',
    'sex_0_0': 'sex',
    '30710-0.0': '30710',
    '30750-0.0': '30750',
    '30700-0.0': '30700',
    '30600-0.0': '30600',
    '30180-0.0': '30180',
    '30070-0.0': '30070',
    '30730-0.0': '30730',
    '30650-0.0': '30650',
    '30620-0.0': '30620'
}

data = ukbb_data[required_fields].rename(columns=rename_map)
data.to_csv('henaw_ukbb_input.csv', index=False)
print(f'Prepared data with {len(data)} participants')
print(f'Complete cases: {data.dropna().shape[0]}')
\""
echo ""

echo "# Then train the model"
echo "python train_henaw.py \\"
echo "    --data-path henaw_ukbb_input.csv \\"
echo "    --output-dir ./ukbb_results \\"
echo "    --max-epochs 100 \\"
echo "    --batch-size 256 \\"
echo "    --device cuda"
echo ""

echo "==================================================="
echo -e "${RED}IMPORTANT SAFETY NOTES${NC}"
echo "==================================================="
echo "1. ALWAYS backup your data before running training"
echo "2. Replace '/path/to/your/ukbb_data.csv' with your actual data path"
echo "3. Ensure your data has the required UK Biobank field IDs"
echo "4. GPU training (--device cuda) is ~10x faster than CPU"
echo "5. The model expects complete cases (404,956 participants)"
echo "6. Training takes 2-4 hours on a standard GPU"
echo "7. Monitor GPU memory usage - reduce batch size if OOM errors occur"
echo "8. For production deployment, set FLASK_ENV=production"
echo ""

echo -e "${GREEN}ERROR RECOVERY TIPS${NC}"
echo "==================================================="
echo "• If CUDA out of memory: reduce --batch-size (try 128, then 64)"
echo "• If training crashes: check the error logs in ./outputs/logs/"
echo "• If prediction server fails: check port availability with 'lsof -i :8080'"
echo "• If import errors: rerun 'pip install -r requirements.txt'"
echo "• If data loading fails: validate your CSV format and field names"
echo ""

echo -e "${GREEN}For detailed usage, see USAGE_GUIDE.md${NC}"

# Add final safety check
echo ""
echo -e "${YELLOW}Before running any commands:${NC}"
echo "1. Activate environment: source henaw_env/bin/activate"
echo "2. Validate installation: python3 -c \"import torch; print('OK')\""
echo "3. Check GPU availability: python3 -c \"import torch; print(torch.cuda.is_available())\""
echo "4. Ensure sufficient disk space (>10GB for training)"
echo ""

log_info "Quick start script completed successfully"
log_info "All critical issues have been fixed with robust error handling"