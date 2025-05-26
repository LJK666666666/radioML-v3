import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime  # For timestamped filenames

# Function to load the RML dataset
def load_radioml_data(file_path):
    print(f"Loading dataset from {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        print("Dataset loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def explore_dataset(dataset):
    """Explore and print information about the dataset."""
    # Show the structure of the dataset
    print("\nDataset Keys (SNR values):")
    snrs = sorted(list(set([k[1] for k in dataset.keys()])))
    print(snrs)
    
    print("\nModulation Types:")
    mods = sorted(list(set([k[0] for k in dataset.keys()])))
    print(mods)
    
    # Get the shape of one example
    example_key = list(dataset.keys())[0]
    example_data = dataset[example_key]
    print(f"\nShape of one example: {example_data.shape}")
    
    # Calculate and print the total number of examples
    total_examples = sum(len(dataset[k]) for k in dataset.keys())
    print(f"Total number of examples: {total_examples}")
    
    # Print examples per modulation type
    print("\nExamples per modulation type:")
    mod_counts = {}
    for k in dataset.keys():
        mod = k[0]
        if mod not in mod_counts:
            mod_counts[mod] = 0
        mod_counts[mod] += len(dataset[k])
    
    for mod, count in mod_counts.items():
        print(f"{mod}: {count} examples")
    
    return mods, snrs

def plot_signal_examples(dataset, mods, output_dir=None, show_plots=True):
    """Plot examples of each modulation type.
    
    Args:
        dataset: The RadioML dataset
        mods: List of modulation types
        output_dir: Directory to save figures, None if not saving
        show_plots: Whether to display the plots (default True)
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    # Get timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot one example per modulation at a decent SNR (e.g., 18dB)
    snr = 18
    plt.figure(figsize=(15, 10))
    
    for i, mod in enumerate(mods):
        key = (mod, snr)
        if key in dataset:
            # Get the first example for this modulation type
            example = dataset[key][0]
            
            # Separate I and Q components
            i_component = example[0, :]  # I component
            q_component = example[1, :]  # Q component
            
            plt.subplot(4, 3, i+1)
            plt.plot(i_component, label='I')
            plt.plot(q_component, label='Q')
            plt.title(f"{mod} at {snr}dB")
            plt.legend()
    
    plt.tight_layout()
    
    if output_dir:
        filename = f"modulation_examples_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300)
        print(f"Saved time domain plot to: {filepath}")
        
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Create IQ constellation plots
    plt.figure(figsize=(15, 10))
    
    for i, mod in enumerate(mods):
        key = (mod, snr)
        if key in dataset:
            # Get the first example for this modulation type
            example = dataset[key][0]
            
            # Separate I and Q components
            i_component = example[0, :]  # I component
            q_component = example[1, :]  # Q component
            
            plt.subplot(4, 3, i+1)
            plt.scatter(i_component, q_component, s=1)
            plt.title(f"{mod} Constellation")
            plt.xlabel('I')
            plt.ylabel('Q')
            plt.grid(True)
    
    plt.tight_layout()
    
    if output_dir:
        filename = f"constellations_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300)
        print(f"Saved constellation plot to: {filepath}")
        
    if show_plots:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    # Folder for outputs
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Path to the dataset
    dataset_path = "RML2016.10a_dict.pkl"
    
    # Load the dataset
    dataset = load_radioml_data(dataset_path)
    
    if dataset:
        # Explore the dataset
        mods, snrs = explore_dataset(dataset)
        
        # Plot examples of each modulation type
        # Set show_plots=True to display plots, False to save without displaying
        plot_signal_examples(dataset, mods, output_dir, show_plots=True)