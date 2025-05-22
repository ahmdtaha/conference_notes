#!/usr/bin/env python3
import re
import glob

output_filename = 'star_counts.txt'
markdown_files = glob.glob('**/*.md', recursive=True)

try:
    with open(output_filename, 'w') as outfile:
        for file_path in markdown_files:
            try:
                with open(file_path, 'r') as infile:
                    content = infile.read()
                    asterisk_count = content.count('*')
                    outfile.write(f"{file_path}: {asterisk_count}\n")
            except FileNotFoundError:
                print(f"Error: File not found at {file_path}")
            except IOError:
                print(f"Error: Could not read file at {file_path}")
    print(f"Results written to {output_filename}")
except IOError:
    print(f"Error: Could not write to output file {output_filename}")
