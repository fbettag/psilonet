#!/bin/bash
# Setup script for psychedelic-inspired LLM experiments

echo "🍄 Setting up Psychedelic LLM Research Environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download SmolLM2-135M model (MLX version)
echo "Downloading SmolLM2-135M model..."
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M'); AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-135M')"

echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run: source venv/bin/activate"
echo "To start experimenting, run: python experiments/skip_layer_baseline.py"
