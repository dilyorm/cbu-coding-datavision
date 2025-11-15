# GitHub Setup Instructions

## Quick Start

### 1. Initialize Git Repository

```bash
# If git is not initialized
git init

# Add remote repository
git remote add origin https://github.com/dilyorm/cbu-coding-datavision.git
```

Or use the setup script (Linux/Mac):
```bash
chmod +x setup_git.sh
./setup_git.sh
```

### 2. Review Changes

```bash
git status
```

### 3. Add Files

```bash
git add .
```

### 4. Commit

```bash
git commit -m "Initial commit: Default prediction model with advanced pipeline"
```

### 5. Push to GitHub

```bash
# First time: set upstream
git push -u origin main

# Or if your default branch is master:
git push -u origin master
```

## Important Notes

### Large Files

The `.gitignore` file is configured to exclude:
- Model files (`.pkl` files in `models/` folder)
- Data files (`.csv`, `.parquet`, `.xlsx`, etc. in `datas/` folder)

These files are typically too large for GitHub. If you need to track them:

1. **Use Git LFS** (Large File Storage):
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pkl"
git lfs track "datas/*.csv"
git lfs track "datas/*.parquet"

# Add .gitattributes
git add .gitattributes
```

2. **Or store them separately** (recommended):
   - Upload large files to cloud storage (Google Drive, Dropbox, etc.)
   - Add download links in README.md
   - Keep only code and small example files in GitHub

### Project Structure

After cleanup, your project should have:

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── model_pipeline_advanced.py
├── predict_advanced.py
├── predict_set.py
├── cleanup_and_organize.py
├── models/
│   ├── .gitkeep
│   └── (model files - ignored by git)
├── datas/
│   ├── .gitkeep
│   └── (data files - ignored by git)
└── (raw data files - you may want to ignore these too)
```

### What Gets Committed

✅ **Will be committed:**
- All Python scripts (`.py` files)
- Configuration files (`.gitignore`, `requirements.txt`)
- Documentation (`.md` files)
- Folder structure (`.gitkeep` files)

❌ **Will NOT be committed** (due to `.gitignore`):
- Model files (`.pkl`)
- Large data files (`.csv`, `.parquet`, `.xlsx`, `.jsonl`, `.xml`)
- Cache directories (`__pycache__/`, `catboost_info/`)
- Old/redundant files

### Adding Raw Data Files

If you want to include the raw data files in the repository, you have options:

1. **Add to .gitignore** (recommended for large files):
   - Already configured - raw data files are ignored

2. **Use Git LFS** for specific files:
```bash
git lfs track "*.csv"
git lfs track "*.parquet"
git lfs track "*.xlsx"
git lfs track "*.jsonl"
git lfs track "*.xml"
```

3. **Create a data download script**:
   - Store data files elsewhere
   - Create a script to download them
   - Document in README

## Troubleshooting

### Branch Name Issues

If you get an error about branch name:
```bash
# Rename branch to main
git branch -M main

# Or use master
git branch -M master
```

### Authentication Issues

If you have authentication issues:
```bash
# Use personal access token instead of password
# Or set up SSH keys
git remote set-url origin git@github.com:dilyorm/cbu-coding-datavision.git
```

### Large File Warnings

If GitHub rejects files for being too large:
1. Check `.gitignore` is working
2. Use Git LFS for necessary large files
3. Remove large files from git history if already committed:
```bash
git rm --cached large_file.pkl
git commit -m "Remove large file"
```

