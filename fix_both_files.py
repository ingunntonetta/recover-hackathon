"""
Complete patch script to fix Kaggle authentication issue in both files
Run this once to fix dataset/metadata.py AND dataset/work_operations.py
"""

import os

def patch_file(filepath, file_description):
    """Patch a single file to fix kaggle import"""
    
    if not os.path.exists(filepath):
        print(f"❌ Error: {filepath} not found!")
        return False
    
    print(f"\n📝 Patching {filepath}...")
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Remove kaggle from top-level imports
    lines = content.split('\n')
    new_lines = []
    removed_import = False
    
    for line in lines:
        if line.strip() == 'import kaggle':
            removed_import = True
            print(f"   ✓ Removed 'import kaggle' from top of file")
            continue
        new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    # Add kaggle import inside download() method
    old_patterns = [
        # Pattern 1: with load_dotenv()
        (
            "    def download(self) -> None:\n        if self._check_exists():\n            return\n\n        load_dotenv()\n\n        kaggle.api.authenticate()",
            "    def download(self) -> None:\n        if self._check_exists():\n            return\n\n        load_dotenv()\n        \n        # Import kaggle only when downloading\n        import kaggle\n        kaggle.api.authenticate()"
        ),
        # Pattern 2: without extra blank line
        (
            "    def download(self) -> None:\n        if self._check_exists():\n            return\n\n        load_dotenv()\n        kaggle.api.authenticate()",
            "    def download(self) -> None:\n        if self._check_exists():\n            return\n\n        load_dotenv()\n        \n        # Import kaggle only when downloading\n        import kaggle\n        kaggle.api.authenticate()"
        ),
    ]
    
    patched_download = False
    for old_pattern, new_pattern in old_patterns:
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            print(f"   ✓ Added conditional kaggle import in download() method")
            patched_download = True
            break
    
    if not patched_download:
        # Try a more flexible approach - find download method and patch it
        if "kaggle.api.authenticate()" in content and "def download(self)" in content:
            # Find the authenticate line and add import before it
            content = content.replace(
                "        kaggle.api.authenticate()",
                "        # Import kaggle only when downloading\n        import kaggle\n        kaggle.api.authenticate()"
            )
            print(f"   ✓ Added conditional kaggle import in download() method (flexible match)")
            patched_download = True
    
    # Check if anything changed
    if content == original_content:
        if removed_import or patched_download:
            print(f"   ⚠️  Some changes detected but content unchanged - may already be patched")
        else:
            print(f"   ⚠️  No changes needed - file may already be patched")
        return True
    
    # Write the patched content
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"   ✅ Successfully patched {file_description}!")
    return True

def main():
    print("="*60)
    print("COMPLETE KAGGLE AUTHENTICATION FIX")
    print("="*60)
    print("\nThis will fix both:")
    print("  - dataset/metadata.py")
    print("  - dataset/work_operations.py")
    print("\nSo they only import kaggle when actually downloading.\n")
    
    files_to_patch = [
        ("dataset/metadata.py", "MetadataDataset"),
        ("dataset/work_operations.py", "WorkOperationsDataset"),
    ]
    
    all_success = True
    for filepath, description in files_to_patch:
        success = patch_file(filepath, description)
        if not success:
            all_success = False
    
    print("\n" + "="*60)
    if all_success:
        print("✅ ALL FILES PATCHED SUCCESSFULLY!")
        print("\nYou can now run:")
        print("  python3 eda.py")
    else:
        print("❌ Some files failed to patch.")
        print("Please check the errors above.")
    print("="*60)

if __name__ == "__main__":
    main()