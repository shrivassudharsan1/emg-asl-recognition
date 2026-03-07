#!/usr/bin/env python3
"""
EMG UMAP Test Configuration Runner
Allows running umap_test.py with different preset or custom configurations
"""

import subprocess
import sys
import json
import argparse
from pathlib import Path

class UMAPRunner:
    def __init__(self, config_file='gesture_configs.json'):
        self.config_file = config_file
        self.configs = self._load_configs()
    
    def _load_configs(self):
        """Load configuration file"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Config file '{self.config_file}' not found")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in '{self.config_file}'")
            sys.exit(1)
    
    def list_configs(self):
        """List all available configurations"""
        print("\n" + "="*60)
        print("Available Configurations")
        print("="*60)
        for key, config in self.configs.items():
            print(f"\n📋 {key.upper()}")
            print(f"   Name: {config['name']}")
            print(f"   Description: {config['description']}")
            print(f"   Gestures:")
            for gesture in config['gestures']:
                print(f"      • {gesture['label']}: {gesture['file']}")
        print("\n" + "="*60)
    
    def run_default(self):
        """Run with default configuration"""
        print("🚀 Running with default configuration...")
        subprocess.run(['python', 'umap_test.py'], check=True)
    
    def run_config(self, config_key):
        """Run with a specific configuration from file"""
        if config_key not in self.configs:
            print(f"Error: Configuration '{config_key}' not found")
            print("Available configurations:")
            for key in self.configs.keys():
                print(f"  - {key}")
            sys.exit(1)
        
        config = self.configs[config_key]
        print(f"\n🚀 Running with configuration: {config_key}")
        print(f"   Name: {config['name']}")
        print(f"   Description: {config['description']}")
        
        cmd = ['python', 'umap_test.py', '--config', self.config_file, '--config-key', config_key]
        subprocess.run(cmd, check=True)
    
    def _find_csv_file(self, file_path):
        """
        Find a CSV file by searching in CSV-Files and its subdirectories.
        Returns the resolved path if found, otherwise returns the original path.
        """
        # Try exact path first
        candidate = Path(file_path)
        if candidate.exists():
            return str(candidate)
        
        # Try under CSV-Files directory
        csv_candidate = Path('CSV-Files') / file_path
        if csv_candidate.exists():
            return str(csv_candidate)
        
        # If just a filename (no directory separators), search subdirectories
        file_name = Path(file_path).name
        if file_name == file_path:  # It's just a filename, not a path
            csv_dir = Path('CSV-Files')
            if csv_dir.exists():
                # Search in subdirectories
                for subdir in csv_dir.iterdir():
                    if subdir.is_dir():
                        candidate_file = subdir / file_name
                        if candidate_file.exists():
                            return str(candidate_file)
        
        # Return original path if not found
        return file_path

    def _get_folder_files(self, folder_name):
        """
        Get all CSV files from a specific gesture folder in CSV-Files.
        Returns a sorted list of file paths.
        """
        folder_path = Path('CSV-Files') / folder_name
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"Error: Folder '{folder_name}' not found in CSV-Files/")
            return []
        
        csv_files = sorted(folder_path.glob('*.csv'))
        return [str(f) for f in csv_files]

    def test_folder_individual(self, folder_name):
        """Test all files in a folder individually (each file as separate label)"""
        files = self._get_folder_files(folder_name)
        if not files:
            sys.exit(1)
        
        # Use just the filename as the label
        labels = [Path(f).stem for f in files]
        
        print(f"\n🚀 Testing all files in folder: {folder_name}")
        print(f"   Files found: {len(files)}")
        for f, label in zip(files, labels):
            print(f"      • {label}: {f}")
        
        cmd = ['python', 'umap_test.py', '--files'] + files + ['--labels'] + labels
        subprocess.run(cmd, check=True)

    def compare_gestures(self, gesture1, gesture2):
        """Compare two gestures by combining all files in each folder"""
        files1 = self._get_folder_files(gesture1)
        files2 = self._get_folder_files(gesture2)
        
        if not files1 or not files2:
            sys.exit(1)
        
        all_files = files1 + files2
        all_labels = [gesture1.replace('-', ' ').title()] * len(files1) + \
                     [gesture2.replace('-', ' ').title()] * len(files2)
        
        print(f"\n🚀 Comparing gestures: {gesture1} vs {gesture2}")
        print(f"   {gesture1}: {len(files1)} files")
        for f in files1:
            print(f"      • {f}")
        print(f"   {gesture2}: {len(files2)} files")
        for f in files2:
            print(f"      • {f}")
        
        cmd = ['python', 'umap_test.py', '--files'] + all_files + ['--labels'] + all_labels
        subprocess.run(cmd, check=True)

    def run_custom(self, files, labels):
        """Run with custom files and labels"""
        if len(files) != len(labels):
            print("Error: Number of files must match number of labels")
            sys.exit(1)

        resolved_files = []
        for file_path in files:
            resolved_files.append(self._find_csv_file(file_path))
        
        print("\n🚀 Running with custom configuration")
        print(f"   Files: {resolved_files}")
        print(f"   Labels: {labels}")
        
        cmd = ['python', 'umap_test.py', '--files'] + resolved_files + ['--labels'] + labels
        subprocess.run(cmd, check=True)
    
    def show_presets(self):
        """Show all preset commands"""
        print("\n" + "="*60)
        print("Quick Run Presets")
        print("="*60)
        presets = {
            'default': 'Default (Open/Close Hand)',
            'ricardo': 'Ricardo only data',
            'all': 'All subjects (Ricardo + Sirisha)',
            'emc': 'EMG_ASL folder data',
            'random': 'Random Forest dataset',
            'fingers': 'Individual finger tests',
            'hand': 'Hand position variants',
            'all_gestures': 'Complete gesture set',
            'comparison': 'Ricardo vs Mohak (Closed Hand)',
        }
        for cmd, desc in presets.items():
            print(f"  python run_umap.py {cmd:15} → {desc}")
        print("\n" + "="*60)
        print("Gesture Folder Analysis")
        print("="*60)
        gestures = ['closed-hand', 'opened-hand', 'hang-loose', 'peace', 'spider-man']
        for gesture in gestures:
            print(f"  Test individual files: python run_umap.py test_folder {gesture}")
        print("\n  Compare two gestures: python run_umap.py compare_gesture <gesture1> <gesture2>")
        for i, gesture1 in enumerate(gestures):
            for gesture2 in gestures[i+1:]:
                print(f"    Example: python run_umap.py compare_gesture {gesture1} {gesture2}")
                if i < 2:  # Just show a few examples
                    pass
                else:
                    print(f"    ... and more combinations")
                    break
            if i >= 1:
                break
        print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description='EMG UMAP Test Configuration Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_umap.py default
  python run_umap.py list
  python run_umap.py test_folder closed-hand
  python run_umap.py compare_gesture spider-man peace
  python run_umap.py custom CSV-Files/Test-Ricardo.csv Ricardo EMG_ASL/CSV/Test-Ricardo_Open-Hand.csv "Open Hand"
        """
    )
    
    parser.add_argument(
        'action',
        nargs='?',
        default='default',
        help='Action: default, list, presets, custom, test_folder, compare_gesture, or preset name'
    )
    
    parser.add_argument(
        'extra_args',
        nargs='*',
        help='Additional arguments for test_folder (folder_name) or compare_gesture (gesture1 gesture2)'
    )
    
    parser.add_argument(
        '--files',
        nargs='+',
        help='File paths for custom configuration (space-separated)'
    )
    
    parser.add_argument(
        '--labels',
        nargs='+',
        help='Labels for custom configuration (space-separated, must match number of files)'
    )
    
    args = parser.parse_args()
    
    runner = UMAPRunner()
    
    if args.action == 'default':
        runner.run_default()
    elif args.action == 'list':
        runner.list_configs()
    elif args.action == 'presets':
        runner.show_presets()
    elif args.action == 'custom':
        if not args.files or not args.labels:
            print("Error: --files and --labels required for custom configuration")
            sys.exit(1)
        runner.run_custom(args.files, args.labels)
    elif args.action == 'test_folder':
        if not args.extra_args or len(args.extra_args) < 1:
            print("Error: test_folder requires a folder name")
            print("Available folders: closed-hand, opened-hand, hang-loose, peace, spider-man")
            print("Example: python run_umap.py test_folder closed-hand")
            sys.exit(1)
        runner.test_folder_individual(args.extra_args[0])
    elif args.action == 'compare_gesture':
        if not args.extra_args or len(args.extra_args) < 2:
            print("Error: compare_gesture requires two gesture names")
            print("Available gestures: closed-hand, opened-hand, hang-loose, peace, spider-man")
            print("Example: python run_umap.py compare_gesture spider-man peace")
            sys.exit(1)
        runner.compare_gestures(args.extra_args[0], args.extra_args[1])
    elif args.action in runner.configs:
        runner.run_config(args.action)
    elif args.action in ['ricardo', 'all', 'emc', 'random', 'fingers', 'hand', 'all_gestures', 'comparison']:
        # Handle preset shortcuts
        preset_mapping = {
            'ricardo': ['CSV-Files/Test-Ricardo.csv'], 
            'all': ['CSV-Files/Test-Ricardo.csv', 'CSV-Files/Test-Sirisha.csv'],
            'emc': ['EMG_ASL/CSV/Test-Ricardo_Open-Hand.csv', 'EMG_ASL/CSV/Test-Ricardo_Closed-Hand.csv'],
            'random': [
                'RandomForest/DATASET EMG MINDROVE/AYU/subjek_AYU1.csv',
                'RandomForest/DATASET EMG MINDROVE/DANIEL/subjek_DANIEL1.csv',
                'RandomForest/DATASET EMG MINDROVE/LINTANG/subjek_Lintang1.csv'
            ],
            'fingers': [
                'CSV-Files/index_finger.csv',
                'CSV-Files/ring_finger.csv',
                'CSV-Files/pinky_finger.csv'
            ],
            'hand': [
                'CSV-Files/closed_hand.csv',
                'CSV-Files/index_finger.csv',
                'CSV-Files/ring_finger.csv',
                'CSV-Files/pinky_finger.csv'
            ],
            'all_gestures': [
                'CSV-Files/Test-Ricardo_Open-Hand.csv',
                'CSV-Files/closed_hand.csv',
                'CSV-Files/index_finger.csv',
                'CSV-Files/ring_finger.csv',
                'CSV-Files/pinky_finger.csv'
            ],
            'comparison': [
                'CSV-Files/Test-Ricardo_Closed-Hand.csv',
                'CSV-Files/Test-Mohak_Closed-Hand.csv'
            ]
        }
        
        label_mapping = {
            'ricardo': ['Ricardo Mixed'],
            'all': ['Ricardo', 'Sirisha'],
            'emc': ['Open Hand', 'Close Hand'],
            'random': ['AYU-1', 'DANIEL-1', 'LINTANG-1'],
            'fingers': ['Index Finger', 'Ring Finger', 'Pinky Finger'],
            'hand': ['Closed Hand', 'Index Finger', 'Ring Finger', 'Pinky Finger'],
            'all_gestures': ['Open Hand', 'Closed Hand', 'Index Finger', 'Ring Finger', 'Pinky Finger'],
            'comparison': ['Ricardo - Closed Hand', 'Mohak - Closed Hand']
        }
        
        files = preset_mapping.get(args.action, [])
        labels = label_mapping.get(args.action, [])
        runner.run_custom(files, labels)
    else:
        print(f"Error: Unknown action '{args.action}'")
        runner.show_presets()
        print("\nRun 'python run_umap.py list' to see all configurations")
        sys.exit(1)

if __name__ == '__main__':
    main()
