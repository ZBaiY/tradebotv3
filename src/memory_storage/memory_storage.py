import os
import psutil
from collections import defaultdict

class StorageMonitor:
    def __init__(self, base_dir):
        """
        Initialize the StorageMonitor with a base directory.
        """
        self.base_dir = base_dir

    def get_size(self, path):
        """
        Recursively calculate the size of a directory or file.
        """
        total_size = 0
        if os.path.isfile(path):
            total_size = os.path.getsize(path)
        else:
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
        return total_size

    def format_size(self, size_in_bytes):
        """
        Format the size from bytes to a human-readable format (KB, MB, GB, etc.).
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_in_bytes < 1024:
                return f"{size_in_bytes:.2f} {unit}"
            size_in_bytes /= 1024

    def storage_report(self):
        """
        Generate a storage usage report for each directory and file under the base_dir.
        """
        storage_data = defaultdict(dict)
        
        for root, dirs, files in os.walk(self.base_dir):
            total_size = sum(self.get_size(os.path.join(root, name)) for name in files)
            storage_data[root]['size'] = self.format_size(total_size)
            storage_data[root]['files'] = len(files)
        
        return storage_data

    def display_report(self):
        """
        Print out the storage usage report in a formatted way.
        """
        report = self.storage_report()
        
        print(f"\n{'Directory/File':<60} {'Size':<10} {'# Files'}")
        print("-" * 80)
        
        for dirpath, info in report.items():
            print(f"{dirpath:<60} {info['size']:<10} {info['files']}")


class MemoryMonitor:
    def __init__(self, limit_percent):
        """
        Initialize the MemoryMonitor to monitor system memory in real-time.
        :param limit_percent: The percentage of memory usage where action should be triggered.
        """
        self.limit_percent = limit_percent

    def get_memory_usage(self):
        """
        Get the current memory usage.
        """
        memory_usage = psutil.virtual_memory()
        return memory_usage.percent, memory_usage.used

    def free_memory(self):
        """
        Simulate freeing two-thirds of the memory.
        In reality, we will perform garbage collection to free up some memory.
        """
        print("Memory limit exceeded, freeing two-thirds of memory...")
        gc.collect()  # Perform garbage collection to free memory
        # Additional memory freeing logic can be implemented depending on the system's setup.

    def monitor_memory(self):
        """
        Check if the current memory usage exceeds the defined limit.
        If it does, free two-thirds of the memory.
        """
        current_percent, used_memory = self.get_memory_usage()
        print(f"Current memory usage: {current_percent}%")

        if current_percent > self.limit_percent:
            self.free_memory()
            current_percent, used_memory = self.get_memory_usage()
            print(f"Memory usage after cleanup: {current_percent}%")

    def display_memory_report(self):
        """
        Display the real-time memory usage report.
        """
        memory_report = self.get_memory_usage()

        print(f"\nReal-Time Memory Usage:")
        print("-" * 30)
        print(f"Total:      {memory_report['total']}")
        print(f"Available:  {memory_report['available']}")
        print(f"Used:       {memory_report['used']}")
        print(f"Free:       {memory_report['free']}")
        print(f"Percentage: {memory_report['percent']}")

