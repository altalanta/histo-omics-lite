# Performance Benchmarks

This project uses pytest-benchmark to track performance regressions and ensure consistent performance across releases.

## Running Benchmarks

### Basic Benchmark Run

```bash
# Run all benchmarks with baseline comparison
make benchmark

# Run specific benchmark test
pytest tests/benchmarks/test_data_generation.py::TestDeterminismBenchmarks::test_benchmark_set_determinism -v
```

### Benchmark Management

```bash
# Save current performance as baseline (run this after optimizations)
make benchmark-save

# Compare current performance to saved baseline
make benchmark-compare

# Reset baseline (save new baseline and clear comparison data)
make benchmark-reset
```

## Benchmark Structure

Benchmarks are organized in `tests/benchmarks/` with the following categories:

### Data Generation Benchmarks (`test_data_generation.py`)
- Synthetic data generation performance
- Determinism setup overhead
- Determinism info collection

### Model Performance Benchmarks (`test_model_performance.py`)
- Forward pass performance (small and large batches)
- Training step performance
- Evaluation metrics computation

## Benchmark Thresholds

The benchmark suite uses these performance thresholds:

- **Regression threshold**: 10% performance degradation fails the build
- **Comparison sorting**: Results sorted by benchmark name for consistency
- **Baseline comparison**: Automatically compares against saved baseline

## Managing Baselines

### When to Update Baselines

Update performance baselines when:

1. **After performance optimizations**: Save new baseline after confirmed improvements
2. **Hardware changes**: Different CI/CD environments may need new baselines
3. **Major dependency updates**: PyTorch or other core dependencies may affect performance

### Baseline Files

Baselines are stored in `.benchmarks/` directory:

```
.benchmarks/
├── Darwin-CPython-3.11-64bit/
│   ├── 0001_baseline.json      # Latest baseline
│   └── 0002_comparison.json    # Previous comparisons
└── Linux-CPython-3.11-64bit/   # Platform-specific baselines
```

### Baseline Refresh Process

```bash
# 1. Run current benchmarks to establish baseline
make benchmark-save

# 2. Verify baseline quality (check for outliers)
cat .benchmarks/*/0001_baseline.json | jq '.benchmarks[].stats'

# 3. Run comparison to ensure threshold is reasonable
make benchmark-compare

# 4. Commit baseline if tests pass and performance is expected
git add .benchmarks/
git commit -m "Update performance baselines"
```

## CI/CD Integration

Benchmarks are integrated into the CI pipeline:

1. **Pull Requests**: Compare against main branch baseline
2. **Main Branch**: Update baseline if all tests pass
3. **Releases**: Validate performance meets release criteria

### CI Benchmark Configuration

```yaml
# In .github/workflows/ci.yml
- name: Run performance benchmarks
  run: make benchmark-compare
  continue-on-error: false  # Fail build on regression
```

## Benchmark Best Practices

### Writing Benchmarks

1. **Use `@pytest.mark.slow`** for expensive benchmarks
2. **Validate results** in addition to timing
3. **Use deterministic seeds** for reproducible results
4. **Clean up resources** (temp files, GPU memory)

Example:
```python
def test_benchmark_model_inference(self, benchmark):
    \"\"\"Benchmark model inference time.\"\"\"
    set_determinism(42)
    
    model = create_test_model()
    input_data = create_test_input()
    
    def inference():
        with torch.no_grad():
            return model(input_data)
    
    result = benchmark(inference)
    
    # Validate output shape/content
    assert result.shape == expected_shape
```

### Benchmark Hygiene

1. **Regular baseline updates**: Update quarterly or after major changes
2. **Platform consistency**: Use same hardware for baseline establishment
3. **Monitoring trends**: Track performance over time, not just regressions
4. **Documentation**: Document performance expectations and thresholds

## Troubleshooting

### Common Issues

1. **High variance in results**
   ```bash
   # Increase benchmark rounds
   pytest tests/benchmarks/ --benchmark-min-rounds=10
   ```

2. **Platform differences**
   ```bash
   # Save platform-specific baselines
   make benchmark-save
   git add .benchmarks/$(uname -s)-*
   ```

3. **Memory pressure affecting results**
   ```bash
   # Run with garbage collection
   pytest tests/benchmarks/ --benchmark-disable-gc
   ```

### Debugging Slow Benchmarks

```bash
# Profile specific benchmark
pytest tests/benchmarks/test_model_performance.py::TestModelBenchmarks::test_benchmark_model_training_step \
    --benchmark-profile=cumtime

# Get detailed timing breakdown
pytest tests/benchmarks/ --benchmark-verbose
```

## Performance Targets

Current performance targets (reference: MacBook Pro M1, 16GB RAM):

| Benchmark | Target | Threshold |
|-----------|--------|-----------|
| Determinism setup | < 200μs | +10% |
| Small data generation (10 patients) | < 2ms | +10% |
| Model forward pass (batch=32) | < 50ms | +10% |
| Retrieval metrics (100 samples) | < 2ms | +10% |

These targets are automatically enforced in CI and will fail the build if exceeded.