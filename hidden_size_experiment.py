"""
Hidden Size Experiment for Guitar Amp Modeling
==============================================
This script runs a systematic experiment to test different hidden_size values
for LSTM-based guitar amp modeling and analyzes the results.

Usage:
    Run this script from your notebook with:
    %run hidden_size_experiment.py
    
    Or import and use the class:
    from hidden_size_experiment import HiddenSizeExperiment
    experiment = HiddenSizeExperiment(hidden_sizes=[16, 32, 48, 64])
    experiment.run_experiment()
"""

import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path
import shutil
import pandas as pd
from scipy.io import wavfile
import random

class HiddenSizeExperiment:
    def __init__(self, 
                 hidden_sizes=None,  # Different hidden sizes to test
                 epochs=200,         # Epochs per model
                 base_results_dir="Results",
                 experiment_dir="hidden_size_experiment",
                 config_path="./input/configuration.json",
                 seed=None):         # Optional fixed seed
        
        # Set default hidden sizes if none provided
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [16, 20, 32, 40, 64, 96, 128]
        self.epochs = epochs
        self.base_results_dir = base_results_dir
        self.experiment_dir = experiment_dir
        self.config_path = config_path
        
        # Generate a random seed for this experiment run if none provided
        self.seed = seed if seed is not None else random.randint(1, 100000)
        
        # Create experiment directory
        os.makedirs(os.path.join(self.base_results_dir, self.experiment_dir), exist_ok=True)
        
        # Results tracking
        self.results = {
            'hidden_size': [],
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'test_esr': [],
            'training_time': [],
            'model_size_kb': []
        }

    def get_num_parameters(self):
        """Get the number of parameters from configuration.json"""
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            return config["Number of Parameters"]
        except Exception as e:
            print(f"Error reading configuration: {e}")
            return None

    def train_model(self, hidden_size):
        """Train a model with the specified hidden size"""
        num_params = self.get_num_parameters()
        if num_params is None:
            return None
        
        input_size = 1 + num_params
        model_name = f"hs_{hidden_size}"
        
        # Start timing
        start_time = time.time()
        
        # Run training command
        cmd = [
            "python", "dist_model_recnet.py",
            "-eps", str(self.epochs),
            "--seed", str(self.seed),
            "-is", str(input_size),
            "-hs", str(hidden_size),
            "-ut", "LSTM",
            "-pf", "None",
            "-fn", "model",
            "-p", model_name
        ]
        
        print(f"\n{'='*80}\nTraining model with hidden_size={hidden_size}\n{'='*80}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Print output
        print(process.stdout)
        if process.stderr:
            print("Errors:", process.stderr)
        
        # Check if the command executed successfully
        if process.returncode != 0:
            print(f"Warning: Command exited with non-zero status: {process.returncode}")
        
        return {
            'model_name': model_name,
            'training_time': training_time
        }

    def extract_metrics(self, model_name):
        """Extract metrics from the training stats file"""
        stats_path = os.path.join(self.base_results_dir, model_name, "training_stats.json")
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            # Get model file size
            model_path = os.path.join(self.base_results_dir, model_name, "model_best.json")
            model_size_kb = os.path.getsize(model_path) / 1024
            
            return {
                'train_loss': stats['training_losses'][-1] if stats['training_losses'] else None,
                'val_loss': stats['best_val_loss'] if 'best_val_loss' in stats else None,
                'test_loss': stats.get('test_loss_best', None),
                'test_esr': stats.get('test_lossESR_best', None),
                'model_size_kb': model_size_kb
            }
        except Exception as e:
            print(f"Error extracting metrics for {model_name}: {e}")
            return {
                'train_loss': None,
                'val_loss': None,
                'test_loss': None,
                'test_esr': None,
                'model_size_kb': None
            }

    def analyze_audio_quality(self, model_name):
        """Analyze audio quality metrics for the model output"""
        try:
            # Load test output and target
            test_out_path = os.path.join(self.base_results_dir, model_name, "test_out_best.wav")
            
            # Copy the output to the experiment directory with a renamed file
            dest_path = os.path.join(self.base_results_dir, self.experiment_dir, f"{model_name}_output.wav")
            shutil.copy(test_out_path, dest_path)
            
            return True
        except Exception as e:
            print(f"Error analyzing audio for {model_name}: {e}")
            return False

    def run_experiment(self):
        """Run the full experiment with all hidden sizes"""
        print(f"Starting hidden size experiment with sizes: {self.hidden_sizes}")
        
        for hidden_size in self.hidden_sizes:
            # Train model
            train_result = self.train_model(hidden_size)
            if not train_result:
                continue
            
            model_name = train_result['model_name']
            
            # Extract metrics
            metrics = self.extract_metrics(model_name)
            
            # Analyze audio
            self.analyze_audio_quality(model_name)
            
            # Store results
            self.results['hidden_size'].append(hidden_size)
            self.results['train_loss'].append(metrics['train_loss'])
            self.results['val_loss'].append(metrics['val_loss'])
            self.results['test_loss'].append(metrics['test_loss'])
            self.results['test_esr'].append(metrics['test_esr'])
            self.results['training_time'].append(train_result['training_time'])
            self.results['model_size_kb'].append(metrics['model_size_kb'])
            
            # Save intermediate results
            self.save_results()
        
        # Create summary report
        summary_report = self.create_summary_report()
        
        # Display the final results plot
        plot_path = os.path.join(self.base_results_dir, self.experiment_dir, "hidden_size_results.png")
        if os.path.exists(plot_path):
            try:
                from IPython import display
                display.display(display.Image(filename=plot_path))
                print(f"Results plot displayed. You can also find it at: {plot_path}")
            except Exception as e:
                print(f"Could not display plot automatically: {e}")
                print(f"Results plot saved to: {plot_path}")
            print()
        
        # Display the summary report
        report_path = os.path.join(self.base_results_dir, self.experiment_dir, "summary_report.md")
        if os.path.exists(report_path):
            try:
                from IPython import display
                with open(report_path, 'r') as f:
                    report_content = f.read()
                display.display(display.Markdown(report_content))
                print(f"Summary report displayed. You can also find it at: {report_path}")
            except Exception as e:
                print(f"Could not display summary report automatically: {e}")
                print(f"Summary report saved to: {report_path}")
            print()
    
        return self.results

    def save_results(self):
        """Save experiment results to CSV and generate plots"""
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Save to CSV
        csv_path = os.path.join(self.base_results_dir, self.experiment_dir, "hidden_size_results.csv")
        df.to_csv(csv_path, index=False)
        
        # Generate plots
        self.plot_results(df)
        
        print(f"Results saved to {csv_path}")
        
        return df

    def plot_results(self, df):
        """Generate plots for the experiment results"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Loss vs Hidden Size
        plt.subplot(2, 2, 1)
        plt.plot(df['hidden_size'], df['train_loss'], 'o-', label='Training Loss')
        plt.plot(df['hidden_size'], df['val_loss'], 's-', label='Validation Loss')
        plt.plot(df['hidden_size'], df['test_loss'], '^-', label='Test Loss')
        plt.xlabel('Hidden Size')
        plt.ylabel('Loss')
        plt.title('Loss vs Hidden Size')
        plt.grid(True)
        plt.legend()
        
        # Plot 2: ESR vs Hidden Size
        plt.subplot(2, 2, 2)
        plt.plot(df['hidden_size'], df['test_esr'], 'o-', color='green')
        plt.xlabel('Hidden Size')
        plt.ylabel('Error-to-Signal Ratio (ESR)')
        plt.title('ESR vs Hidden Size')
        plt.grid(True)
        
        # Plot 3: Training Time vs Hidden Size
        plt.subplot(2, 2, 3)
        plt.plot(df['hidden_size'], df['training_time'] / 60, 'o-', color='orange')  # Convert to minutes
        plt.xlabel('Hidden Size')
        plt.ylabel('Training Time (minutes)')
        plt.title('Training Time vs Hidden Size')
        plt.grid(True)
        
        # Plot 4: Model Size vs Hidden Size
        plt.subplot(2, 2, 4)
        plt.plot(df['hidden_size'], df['model_size_kb'], 'o-', color='purple')
        plt.xlabel('Hidden Size')
        plt.ylabel('Model Size (KB)')
        plt.title('Model Size vs Hidden Size')
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.base_results_dir, self.experiment_dir, "hidden_size_results.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Try to display the plot in the notebook if running in one
        try:
            from IPython import display
            return display.Image(filename=plot_path)
        except:
            pass

    def create_summary_report(self):
        """Create a summary report with recommendations"""
        try:
            csv_path = os.path.join(self.base_results_dir, self.experiment_dir, "hidden_size_results.csv")
            df = pd.read_csv(csv_path)
            
            # Find best model based on test loss
            best_idx = df['test_loss'].idxmin()
            best_hs = df.iloc[best_idx]['hidden_size']
            
            # Calculate trade-offs
            smallest_viable_idx = df[df['test_loss'] <= df['test_loss'].min() * 1.1].iloc[0].name
            smallest_viable_hs = df.iloc[smallest_viable_idx]['hidden_size']
            
            # Generate report
            report = f"""
            # Hidden Size Experiment Results Summary
            
            ## Best Performing Model
            - **Optimal Hidden Size**: {best_hs}
            - **Test Loss**: {df.iloc[best_idx]['test_loss']:.6f}
            - **ESR**: {df.iloc[best_idx]['test_esr']:.6f}
            - **Training Time**: {df.iloc[best_idx]['training_time'] / 60:.2f} minutes
            - **Model Size**: {df.iloc[best_idx]['model_size_kb']:.2f} KB
            
            ## Efficient Alternative
            - **Smallest Viable Hidden Size**: {smallest_viable_hs}
            - **Test Loss**: {df.iloc[smallest_viable_idx]['test_loss']:.6f}
            - **ESR**: {df.iloc[smallest_viable_idx]['test_esr']:.6f}
            - **Training Time**: {df.iloc[smallest_viable_idx]['training_time'] / 60:.2f} minutes
            - **Model Size**: {df.iloc[smallest_viable_idx]['model_size_kb']:.2f} KB
            
            ## Recommendations
            
            1. **For Best Quality**: Use hidden_size = {best_hs}
            2. **For Efficiency**: Use hidden_size = {smallest_viable_hs}
            3. **For Real-time Applications**: Consider hidden_size = {min(smallest_viable_hs, 40)}
            
            ## Trade-offs
            
            - Increasing hidden_size beyond {best_hs} does not significantly improve model performance
            - Smaller hidden sizes ({', '.join(map(str, df[df['hidden_size'] < smallest_viable_hs]['hidden_size'].tolist()))}) 
              showed insufficient capacity to model the audio transformation accurately
            
            ## Next Steps
            
            1. Listen to the audio samples in the experiment directory to subjectively evaluate quality
            2. For your specific amp model, consider fine-tuning with the recommended hidden_size
            3. If modeling multiple parameters, you may need to increase hidden_size as parameter count increases
            """
            
            # Save report
            report_path = os.path.join(self.base_results_dir, self.experiment_dir, "summary_report.md")
            with open(report_path, 'w') as f:
                f.write(report)
            
            print(f"Summary report created at {report_path}")
            return report
        except Exception as e:
            print(f"Error creating summary report: {e}")
            return None

    def find_optimal_hidden_size(self):
        """Find the optimal hidden size based on test results"""
        try:
            df = pd.DataFrame(self.results)
            
            # Find best model based on test loss
            best_idx = df['test_loss'].idxmin()
            best_hs = df.iloc[best_idx]['hidden_size']
            
            # Find smallest viable model (within 10% of best test loss)
            threshold = df['test_loss'].min() * 1.1
            viable_models = df[df['test_loss'] <= threshold]
            smallest_viable_hs = viable_models['hidden_size'].min()
            
            print(f"\n{'='*80}")
            print(f"RESULTS SUMMARY:")
            print(f"{'='*80}")
            print(f"Best performing hidden size: {best_hs} (Test Loss: {df.iloc[best_idx]['test_loss']:.6f})")
            print(f"Most efficient viable hidden size: {smallest_viable_hs}")
            print(f"{'='*80}")
            
            return best_hs, smallest_viable_hs
        except Exception as e:
            print(f"Error finding optimal hidden size: {e}")
            return None, None

# If the script is run directly, execute with default parameters
if __name__ == "__main__":
    print("Starting Hidden Size Experiment with default parameters")
    experiment = HiddenSizeExperiment()
    experiment.run_experiment()
    print("Experiment completed!")
