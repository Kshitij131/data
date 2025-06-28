"""
Project cleanup script to remove temporary and unnecessary files.
"""
import os
import shutil
import re

# Files to keep (core project files)
essential_files = [
    # Core functionality
    "gpt_agent.py",
    "main.py",
    "streamlit_app.py",
    "faiss_store.py",
    "file_utils.py",
    "local_validator.py",
    
    # Project configuration
    "requirements.txt",
    ".env",
    ".gitignore",
    "README.md",
    
    # Test files
    "test_project.py",
    
    # Sample data
    "Clients_Sample_Inconsistent.csv",
    "Clients_Clean_Consistent.csv",
    
    # Data directory
    "data",
    "gemini.py",
    ".git",
    "ai",
    # This cleanup script
    "cleanup.py"
]

# Pattern for fixed/temp files to delete
temp_pattern = re.compile(r'^(temp_|fixed_|.*_fixed\d*\.csv$)')

def cleanup_project():
    """Clean up the project by removing unnecessary files"""
    print("=== Project Cleanup ===")
    
    # Get current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # List all files in current directory
    all_files = os.listdir(current_dir)
    
    # Count files before cleanup
    print(f"Total files before cleanup: {len(all_files)}")
    
    # Files to delete
    files_to_delete = []
    
    for file in all_files:
        # Skip directories (except __pycache__)
        if os.path.isdir(file) and file != "__pycache__":
            if file not in essential_files:
                files_to_delete.append(file)
            continue
        
        # Skip essential files
        if file in essential_files:
            continue
        
        # Delete temporary files
        if temp_pattern.match(file):
            files_to_delete.append(file)
            continue
        
        # Delete test files
        if file.startswith("test_") and file != "test_project.py":
            files_to_delete.append(file)
            continue
        
        # Delete Python cache files
        if file.endswith(".pyc") or file == "__pycache__":
            files_to_delete.append(file)
            continue
        
        # Delete any other files not in the essential list
        if file not in essential_files:
            files_to_delete.append(file)
    
    # Delete files
    if files_to_delete:
        print(f"\nFiles to delete ({len(files_to_delete)}):")
        for file in files_to_delete:
            print(f"- {file}")
        
        confirm = input("\nProceed with deletion? (y/n): ")
        if confirm.lower() == 'y':
            for file in files_to_delete:
                try:
                    file_path = os.path.join(current_dir, file)
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    else:
                        os.remove(file_path)
                    print(f"Deleted: {file}")
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
            
            # Count files after cleanup
            remaining_files = os.listdir(current_dir)
            print(f"\nTotal files after cleanup: {len(remaining_files)}")
            print("✅ Cleanup complete!")
        else:
            print("Cleanup cancelled.")
    else:
        print("No files to delete. Project is already clean.")

if __name__ == "__main__":
    cleanup_project()