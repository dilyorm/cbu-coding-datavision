#!/bin/bash
# Git setup script for the project

echo "=========================================="
echo "Setting up Git repository"
echo "=========================================="

# Initialize git if not already initialized
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
fi

# Add remote if not exists
if ! git remote | grep -q origin; then
    echo "Adding remote repository..."
    git remote add origin https://github.com/dilyorm/cbu-coding-datavision.git
else
    echo "Remote 'origin' already exists. Updating URL..."
    git remote set-url origin https://github.com/dilyorm/cbu-coding-datavision.git
fi

# Add all files
echo "Adding files to git..."
git add .

# Show status
echo ""
echo "=========================================="
echo "Git Status:"
echo "=========================================="
git status

echo ""
echo "=========================================="
echo "Next steps:"
echo "=========================================="
echo "1. Review the changes: git status"
echo "2. Commit: git commit -m 'Initial commit: Default prediction model'"
echo "3. Push: git push -u origin main"
echo ""
echo "Note: Large model files (.pkl) and data files are ignored by .gitignore"
echo "      You may need to use Git LFS for large files if needed"

