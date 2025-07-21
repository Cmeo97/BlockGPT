# Embedding Thermalizer Test Suite

This directory contains comprehensive tests to validate the embedding thermalizer infrastructure before running experiments.

## 🧪 Test Overview

The test suite validates three critical aspects:

1. **Pipeline Dimensions** - Ensures all tensor shapes are compatible throughout the pipeline
2. **Forward/Backward Propagation** - Verifies gradients flow correctly during training
3. **Output Path Management** - Confirms all files are saved to configured locations

## 🚀 Quick Start

Run all tests with a single command:

```bash
bash embedding_thermalizer/scripts/run_tests.sh
```

## 📋 Test Types

### Quick Validation (`--mode quick`)
- ✅ Dependency checks (PyTorch, NumPy, etc.)
- ✅ Module imports
- ✅ Configuration loading
- ✅ Basic functionality

**Runtime:** ~10 seconds

```bash
bash embedding_thermalizer/scripts/run_tests.sh --mode quick
```

### Comprehensive Tests (`--mode comprehensive`)
- ✅ Pipeline dimension validation
- ✅ Token-embedding conversion
- ✅ Model forward/backward passes
- ✅ Output path management
- ✅ File saving/loading
- ✅ TensorBoard integration

**Runtime:** ~2-5 minutes

```bash
bash embedding_thermalizer/scripts/run_tests.sh --mode comprehensive
```

### Integration Test (`--mode integration`)
- ✅ End-to-end pipeline flow
- ✅ Component interaction
- ✅ Real workflow simulation
- ✅ Error handling

**Runtime:** ~30 seconds

```bash
bash embedding_thermalizer/scripts/run_tests.sh --mode integration
```

## 📊 Test Details

### Pipeline Dimension Tests (`test_pipeline_dimensions.py`)

**What it validates:**
- Token ↔ Embedding conversion shapes
- Noise classifier input/output dimensions
- Diffusion model tensor compatibility
- Gradient flow through all components
- Memory allocation and cleanup

**Example Output:**
```
=== Testing Embedding Dimensions ===
✓ Token-embedding conversion test passed
  - Tokens shape: torch.Size([4, 64])
  - Embeddings shape: torch.Size([4, 64, 256])
  - Reconstructed tokens shape: torch.Size([4, 64])
```

### Output Path Tests (`test_output_paths.py`)

**What it validates:**
- Directory creation from config
- Checkpoint saving to correct paths
- Log file generation
- TensorBoard file creation
- Plot saving functionality
- Path validation and error handling

**Example Output:**
```
=== Testing Directory Creation ===
✓ Directory created: /tmp/test/logs
✓ Directory created: /tmp/test/checkpoints
✓ Path config valid: data_dir -> /tmp/test/data
```

## 🔧 Configuration Testing

The tests automatically create temporary configurations to avoid interfering with your actual setup. Key aspects tested:

- **Path Resolution**: Relative → Absolute path conversion
- **Directory Creation**: All configured directories are created
- **File Permissions**: Read/write access validation
- **Config Validation**: Required keys and format checking

## 📈 TensorBoard Integration Testing

The test suite validates:
- ✅ TensorBoard writer initialization
- ✅ Metric logging (scalars, histograms)
- ✅ Model architecture logging
- ✅ Event file generation
- ✅ Directory structure creation

**TensorBoard Validation:**
```python
# Tests log metrics like:
logger.log_step({"train/loss": 1.5, "accuracy": 0.8}, step=0)
logger.log_embeddings_info(embeddings, "clean_embeddings", step=5)
logger.log_gradient_norms(model, step=10)
```

## 🔄 Gradient Flow Validation

Critical for training success, the tests verify:

```python
def test_gradient_flow():
    # Create model and input
    model = torch.nn.Sequential(...)
    input_embeddings = torch.randn(..., requires_grad=True)
    
    # Forward pass
    output = model(input_embeddings)
    loss = torch.nn.functional.mse_loss(output, target)
    
    # Backward pass
    loss.backward()
    
    # Validate gradients exist and are reasonable
    assert input_embeddings.grad is not None
    assert all(param.grad is not None for param in model.parameters())
```

## 📝 Test Reports

Generate detailed reports with:

```bash
bash embedding_thermalizer/scripts/run_tests.sh --report
```

Reports include:
- ✅ Test results summary
- ✅ System information
- ✅ Execution times
- ✅ Next steps recommendations

**Sample Report:**
```markdown
# Embedding Thermalizer Test Report

**Test Date:** 2024-01-15 14:30:25

## Test Results Summary
- **Quick_validation:** ✅ PASSED (8.32s)
- **Dimensions:** ✅ PASSED (45.67s)
- **Paths:** ✅ PASSED (23.41s)
- **Integration:** ✅ PASSED (12.89s)

## System Information
- **PyTorch Version:** 2.1.0
- **CUDA Available:** True
- **GPU:** NVIDIA A100-SXM4-40GB
```

## 🎯 Expected Test Results

**All tests should pass before running experiments!**

### ✅ Success Indicators:
- All tensor shapes match expected dimensions
- Gradients flow through all model components
- Files save to configured directories
- No memory leaks detected
- TensorBoard logs generate correctly

### ❌ Common Failure Causes:

1. **Dimension Mismatches:**
   ```
   Expected shape (4, 64, 256), got (4, 256, 64)
   ```
   *Fix: Check token-embedding conversion logic*

2. **Missing Dependencies:**
   ```
   ModuleNotFoundError: No module named 'tensorboard'
   ```
   *Fix: pip install tensorboard*

3. **Path Permission Errors:**
   ```
   PermissionError: [Errno 13] Permission denied: '/readonly/path'
   ```
   *Fix: Use writable directories in config*

4. **CUDA Memory Issues:**
   ```
   RuntimeError: CUDA out of memory
   ```
   *Fix: Reduce test batch sizes*

## 🛠 Debugging Failed Tests

### Step 1: Run with verbose output
```bash
bash embedding_thermalizer/scripts/run_tests.sh --verbose
```

### Step 2: Check individual test modules
```bash
cd embedding_thermalizer
python tests/test_pipeline_dimensions.py
python tests/test_output_paths.py
```

### Step 3: Review generated logs
```bash
cat embedding_thermalizer/tests/logs/*.log
```

### Step 4: Check test report
```bash
cat embedding_thermalizer/tests/test_report.md
```

## 🔄 Continuous Testing

**Before each experiment run:**
```bash
# Quick validation (10 seconds)
bash embedding_thermalizer/scripts/run_tests.sh --mode quick

# If quick validation passes, proceed with experiments
bash embedding_thermalizer/scripts/run_data_preparation.sh
```

**After infrastructure changes:**
```bash
# Full comprehensive testing
bash embedding_thermalizer/scripts/run_tests.sh --mode comprehensive --report
```

## 📂 Test File Structure

```
embedding_thermalizer/tests/
├── README.md                    # This file
├── run_all_tests.py            # Master test runner
├── test_pipeline_dimensions.py # Pipeline validation
├── test_output_paths.py        # Path management tests
├── logs/                       # Test execution logs
├── temp/                       # Temporary test files
└── results/                    # Test outputs
```

## ⚡ Performance Expectations

| Test Type | Duration | What's Tested |
|-----------|----------|---------------|
| Quick | 10s | Dependencies, imports, basic functionality |
| Comprehensive | 2-5 min | Full pipeline, all components, file I/O |
| Integration | 30s | End-to-end workflow, component interaction |
| All | 3-6 min | Everything above in sequence |

## 🎉 Success Criteria

Your infrastructure is ready when:

1. ✅ All tests pass without errors
2. ✅ No warnings about missing dependencies
3. ✅ TensorBoard files generate correctly
4. ✅ All configured paths work properly
5. ✅ Memory usage is reasonable
6. ✅ Gradient flow is validated

**Next Steps After Passing Tests:**
1. Configure experiment paths in `configs/experiment_config.json`
2. Run data preparation
3. Start training
4. Monitor with TensorBoard

---

*For questions about test failures, check the generated test report or review the verbose output logs.* 