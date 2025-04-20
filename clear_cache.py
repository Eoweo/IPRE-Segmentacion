import os
import shutil
import datetime

# Paths
CACHE_DIR = "/mnt/workspace/cmorenor/.cache"
WANDB_CACHE_DIR = os.path.join(CACHE_DIR, "wandb")
LOG_FILE = "/home/cmorenor/Eoweo/IPRE-Segmentacion/cache_cleanup_log.txt"


def get_folder_size(folder_path):
    """Returns the total size of a folder in GB."""
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return round(total_size / (1024 ** 3), 2)  # Convert bytes to GB


def delete_folder(folder_path):
    """Deletes all files and subfolders inside a given directory and returns its size."""
    if not os.path.exists(folder_path):
        print(f"Directory '{folder_path}' does not exist.")
        return 0

    folder_size = get_folder_size(folder_path)

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Failed to delete {item_path}: {e}")

    print(f"Deleted '{folder_path}', Size: {folder_size} GB")
    return folder_size


def initialize_log():
    """Creates log file with a formatted header if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as log_file:
            log_file.write(f"{'Date':<12} | {'Cache Folder Size':<18} | {'Deleted Folder':<15} | {'Deleted Size (GB)':<18}\n")
            log_file.write("-" * 70 + "\n")


def log_cleanup(cache_size, deleted_folders):
    """Logs the cleanup details to a file with proper formatting."""
    initialize_log()
    timestamp = datetime.datetime.now().strftime("%d/%m/%y")

    for folder, size in deleted_folders.items():
        log_entry = f"{timestamp:<12} | {str(cache_size) + ' GB':<18} | {folder:<15} | {str(size) + ' GB':<18}\n"

        with open(LOG_FILE, "a") as log_file:
            log_file.write(log_entry)

    print("Cleanup logged successfully.")


def main():
    print("\nWandB Cache Cleaner")
    print("1. Delete ONLY WandB cache (`wandb/`)")
    print("2. Delete ALL cache inside `.cache/`")
    print("3. Exit")

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == "1":
        total_cache_size = get_folder_size(CACHE_DIR)
        wandb_size = delete_folder(WANDB_CACHE_DIR)
        log_cleanup(total_cache_size, {"wandb": wandb_size})

    elif choice == "2":
        total_cache_size = get_folder_size(CACHE_DIR)
        cache_size = delete_folder(CACHE_DIR)
        log_cleanup(total_cache_size, {"cache": cache_size})

    elif choice == "3":
        print("Exiting without deleting anything.")
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
