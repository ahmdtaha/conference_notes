#!/usr/bin/env python3
import glob
import re

import matplotlib.pyplot as plt

output_filename = 'star_counts.txt'
asterisk_counts_by_year = {}

try:
    with open(output_filename, 'w') as outfile:
        for year in range(2020, 2027):
            try:
                file_path = f'conf_{year}.md'
                with open(file_path, 'r') as infile:
                    content = infile.read()
                    asterisk_count = content.count('*')
                    asterisk_counts_by_year[year] = asterisk_count
                    outfile.write(f"{file_path}: {asterisk_count}\n")
            except FileNotFoundError:
                print(f"Error: File not found at {file_path}")
                asterisk_counts_by_year[year] = 0  # Store 0 if file not found
            except IOError:
                print(f"Error: Could not read file at {file_path}")
                asterisk_counts_by_year[year] = 0  # Store 0 if file cannot be read
    print(f"Results written to {output_filename}")
except IOError:
    print(f"Error: Could not write to output file {output_filename}")

# Later, this dictionary will be used for plotting
# For now, let's print it to verify
print("Asterisk counts by year:", asterisk_counts_by_year)

# --- Plotting Function ---
# import matplotlib.pyplot as plt # Already imported at the top


def plot_asterisk_counts(years_list, counts_list, plot_filename='asterisk_counts_plot.png'):
    """
    Generates and saves a bar plot of asterisk counts per year.

    Args:
        years_list (list): A list of years.
        counts_list (list): A list of corresponding asterisk counts.
        plot_filename (str): The filename for saving the plot image.
    """
    if not years_list or not counts_list or len(years_list) != len(counts_list):
        print(
            "Invalid data provided for plotting. Ensure years and counts are non-empty and of the same length."
        )
        return

    plt.figure(figsize=(10, 6))
    plt.bar(years_list, counts_list, color='skyblue')
    plt.xlabel("Year")
    plt.ylabel("Number of Asterisks (*)")
    plt.title("Asterisk Counts in Conference Files per Year")
    plt.xticks(years_list)  # Ensure all years are displayed as ticks
    plt.grid(axis='y', linestyle='--')

    try:
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")


# --- End of Plotting Function ---

# Prepare data for plotting
sorted_years = sorted(asterisk_counts_by_year.keys())
counts_for_plot = [asterisk_counts_by_year[year] for year in sorted_years]

# Call the plotting function after it's defined
plot_asterisk_counts(sorted_years, counts_for_plot)
