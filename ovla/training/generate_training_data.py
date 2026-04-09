"""
GENERATE FULL TRAINING DATASET

This is the BIG ONE - generate all training data for the Universal Semantic Mapper.

Target: 50 robots × 60 primitives × 100 samples = 300,000 samples
We'll start with a smaller batch to test, then scale up.
"""
import sys
sys.path.insert(0, '/scratch/anshb3/ovla')

from primitives.primitive_executor import generate_primitive_dataset

print("="*70)
print("GENERATING TRAINING DATASET FOR UNIVERSAL SEMANTIC MAPPER")
print("="*70)

# Start with SMALL batch for testing
print("\n📋 PHASE 1: Generate small test batch")
print("  - 10 robots")
print("  - 10 primitives")  
print("  - 100 samples per primitive")
print("  - Expected: ~10,000 samples")

samples = generate_primitive_dataset(
    robot_catalog_path='/scratch/anshb3/ovla/robots/robot_catalog.json',
    primitive_catalog_path='/scratch/anshb3/ovla/primitives/primitive_catalog.json',
    output_dir='/scratch/anshb3/ovla/training_data/semantic_primitives',
    samples_per_primitive=100,
    max_robots=10  # Start with 10 robots
)

print("\n" + "="*70)
print("DATASET GENERATION COMPLETE")
print("="*70)

print(f"\n✅ Generated {len(samples)} training samples!")
print(f"\nNext steps:")
print("  1. Verify data quality")
print("  2. Train Universal Semantic Mapper")
print("  3. Validate on held-out robots")
print("  4. Scale to full 300K dataset if working")
