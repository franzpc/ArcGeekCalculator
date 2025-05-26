"""
Classification Report Generator
-----------------------------------------
This module generates comprehensive HTML reports for image classification results
with scientifically rigorous accuracy assessment metrics.

Developed by ArcGeek, 2024
Based on standard remote sensing accuracy assessment protocols.
"""

import datetime
import base64
import io
import os
import numpy as np

# Check dependencies at module level
HAS_MATPLOTLIB = True
HAS_SEABORN = True
HAS_SKLEARN = True

try:
    import matplotlib
    matplotlib.use('Agg')  # Set Matplotlib to use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
except ImportError:
    HAS_SEABORN = False

try:
    from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score
    from sklearn.model_selection import cross_val_score
except ImportError:
    HAS_SKLEARN = False

class ScientificClassificationReportGenerator:
    """Generates comprehensive HTML reports for classification results with scientific rigor"""
    
    def __init__(self):
        self.has_dependencies = HAS_MATPLOTLIB and HAS_SEABORN and HAS_SKLEARN
        
    def check_dependencies(self):
        """Check if all required dependencies are available"""
        missing_libs = []
        if not HAS_MATPLOTLIB:
            missing_libs.append("matplotlib")
        if not HAS_SEABORN:
            missing_libs.append("seaborn")
        if not HAS_SKLEARN:
            missing_libs.append("scikit-learn")
        return missing_libs
    
    def calculate_scientific_metrics(self, y_test, y_pred, confusion_matrix, class_counts, n_bands):
        """
        Calculate additional scientific metrics important in remote sensing
        """
        metrics = {}
        
        # 1. Cohen's Kappa Coefficient (CRITICAL in remote sensing)
        if HAS_SKLEARN:
            metrics['kappa'] = cohen_kappa_score(y_test, y_pred)
            metrics['kappa_interpretation'] = self.interpret_kappa_coefficient(metrics['kappa'])
            
            # 2. Balanced Accuracy (important for imbalanced datasets)
            metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)
        else:
            metrics['kappa'] = self.calculate_kappa_manual(confusion_matrix)
            metrics['kappa_interpretation'] = self.interpret_kappa_coefficient(metrics['kappa'])
            metrics['balanced_accuracy'] = self.calculate_balanced_accuracy_manual(confusion_matrix)
        
        # 3. Producer's Accuracy and User's Accuracy (standard remote sensing terminology)
        n_classes = confusion_matrix.shape[0]
        
        # Handle division by zero
        row_sums = np.sum(confusion_matrix, axis=1)
        col_sums = np.sum(confusion_matrix, axis=0)
        
        producers_accuracy = np.where(row_sums > 0, 
                                    np.diag(confusion_matrix) / row_sums, 
                                    0.0)
        users_accuracy = np.where(col_sums > 0,
                                np.diag(confusion_matrix) / col_sums,
                                0.0)
        
        # 4. Omission and Commission Errors
        omission_error = 1 - producers_accuracy
        commission_error = 1 - users_accuracy
        
        metrics.update({
            'producers_accuracy': producers_accuracy,
            'users_accuracy': users_accuracy,
            'omission_error': omission_error,
            'commission_error': commission_error
        })
        
        # 5. Sample Size Adequacy Assessment
        metrics['sample_adequacy'] = self.assess_sample_size_adequacy(class_counts, n_bands)
        metrics['min_samples_per_class'] = min(class_counts)
        metrics['recommended_minimum'] = max(10 * n_bands, 30)
        
        return metrics
    
    def calculate_kappa_manual(self, confusion_matrix):
        """Manual calculation of Cohen's Kappa if sklearn is not available"""
        n = np.sum(confusion_matrix)
        observed_accuracy = np.sum(np.diag(confusion_matrix)) / n
        
        row_marginals = np.sum(confusion_matrix, axis=1) / n
        col_marginals = np.sum(confusion_matrix, axis=0) / n
        expected_accuracy = np.sum(row_marginals * col_marginals)
        
        if expected_accuracy == 1:
            return 1.0
        
        kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
        return kappa
    
    def calculate_balanced_accuracy_manual(self, confusion_matrix):
        """Manual calculation of balanced accuracy"""
        n_classes = confusion_matrix.shape[0]
        recall_per_class = []
        
        for i in range(n_classes):
            row_sum = np.sum(confusion_matrix[i, :])
            if row_sum > 0:
                recall = confusion_matrix[i, i] / row_sum
                recall_per_class.append(recall)
        
        return np.mean(recall_per_class) if recall_per_class else 0.0
    
    def interpret_kappa_coefficient(self, kappa):
        """
        Interpretation of Kappa coefficient according to Landis & Koch (1977)
        """
        if kappa < 0:
            return "Poor (worse than random)"
        elif kappa < 0.20:
            return "Slight"
        elif kappa < 0.40:
            return "Fair"
        elif kappa < 0.60:
            return "Moderate"
        elif kappa < 0.80:
            return "Substantial"
        else:
            return "Almost Perfect"
    
    def assess_sample_size_adequacy(self, class_counts, n_bands):
        """
        Assess sample size adequacy according to remote sensing best practices
        """
        min_samples_per_class = min(class_counts)
        recommended_minimum = max(10 * n_bands, 30)  # Common rule: 10 samples per band or minimum 30
        
        if min_samples_per_class >= recommended_minimum:
            return "Adequate"
        elif min_samples_per_class >= recommended_minimum * 0.7:
            return "Marginal"
        else:
            return "Insufficient"
    
    def figure_to_base64(self, figure):
        """Convert matplotlib figure to base64 encoded image for HTML embedding"""
        if not self.has_dependencies:
            return ""
            
        try:
            img_buf = io.BytesIO()
            figure.savefig(img_buf, format='png', bbox_inches='tight', dpi=150)
            img_buf.seek(0)
            img_str = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close(figure)
            return img_str
        except Exception as e:
            plt.close(figure)
            return ""
    
    def create_enhanced_confusion_matrix_plot(self, cm, class_names, producers_acc, users_acc):
        """Create enhanced confusion matrix plot with accuracy metrics"""
        if not self.has_dependencies:
            return None
            
        try:
            # Create figure with better styling and additional space for metrics
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Main confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'label': 'Number of Samples'},
                       square=True, linewidths=0.5, ax=ax1)
            
            ax1.set_ylabel('True Class (Reference)', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
            ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
            
            # Rotate labels for better readability
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
            ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
            
            # Accuracy metrics table
            ax2.axis('off')
            
            # Create accuracy table data
            table_data = []
            for i, class_name in enumerate(class_names):
                table_data.append([
                    class_name,
                    f"{producers_acc[i]:.3f}",
                    f"{users_acc[i]:.3f}",
                    f"{(1-producers_acc[i]):.3f}",
                    f"{(1-users_acc[i]):.3f}"
                ])
            
            # Create table
            table = ax2.table(cellText=table_data,
                            colLabels=['Class', "Producer's\nAccuracy", "User's\nAccuracy", 
                                     'Omission\nError', 'Commission\nError'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0.1, 0.3, 0.8, 0.4])
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # Style the table
            for i in range(len(class_names) + 1):
                for j in range(5):
                    cell = table[(i, j)]
                    if i == 0:  # Header row
                        cell.set_facecolor('#2E86AB')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
            
            ax2.set_title('Accuracy Assessment Summary', fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            return None
    
    def create_kappa_interpretation_plot(self, kappa_value):
        """Create visualization showing kappa interpretation scale"""
        if not self.has_dependencies:
            return None
            
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Kappa scale
            kappa_ranges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            labels = ['Slight', 'Fair', 'Moderate', 'Substantial', 'Almost Perfect']
            colors = ['#E63946', '#F77F00', '#FCBF49', '#06D6A0', '#2E86AB']
            
            # Create horizontal bar chart
            for i in range(len(labels)):
                ax.barh(0, 0.2, left=kappa_ranges[i], height=0.3, 
                       color=colors[i], alpha=0.7, edgecolor='black')
                
                # Add labels
                ax.text(kappa_ranges[i] + 0.1, 0, labels[i], 
                       ha='center', va='center', fontweight='bold', fontsize=10)
            
            # Add current kappa value
            ax.axvline(x=kappa_value, color='red', linewidth=4, linestyle='--')
            ax.text(kappa_value, 0.2, f'Your Kappa: {kappa_value:.3f}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.2, 0.4)
            ax.set_xlabel('Cohen\'s Kappa Coefficient', fontsize=12, fontweight='bold')
            ax.set_title('Kappa Coefficient Interpretation Scale\n(Landis & Koch, 1977)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_yticks([])
            
            # Add reference line at 0
            ax.axvline(x=0, color='black', linewidth=1, alpha=0.5)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            return None
    
    def create_sample_adequacy_plot(self, class_names, class_counts, n_bands):
        """Create plot showing sample size adequacy assessment"""
        if not self.has_dependencies:
            return None
            
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            recommended_min = max(10 * n_bands, 30)
            marginal_threshold = recommended_min * 0.7
            
            # Create color coding based on adequacy
            colors = []
            for count in class_counts:
                if count >= recommended_min:
                    colors.append('#2E86AB')  # Adequate - Blue
                elif count >= marginal_threshold:
                    colors.append('#F77F00')  # Marginal - Orange
                else:
                    colors.append('#E63946')  # Insufficient - Red
            
            bars = ax.bar(class_names, class_counts, color=colors, edgecolor='black', linewidth=1.2)
            
            # Add horizontal lines for thresholds
            ax.axhline(y=recommended_min, color='green', linestyle='--', linewidth=2, 
                      label=f'Recommended Minimum ({recommended_min})')
            ax.axhline(y=marginal_threshold, color='orange', linestyle='--', linewidth=2,
                      label=f'Marginal Threshold ({marginal_threshold:.0f})')
            
            # Add value labels on bars
            for bar, count in zip(bars, class_counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(count)}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f'Sample Size Adequacy Assessment\nRule: Minimum {max(10, 10 * n_bands)} samples per class for {n_bands} bands', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Class', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Training Samples', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            return fig
        except Exception as e:
            return None
    
    def create_class_distribution_plot(self, classes, counts):
        """Create enhanced class distribution plot"""
        if not self.has_dependencies:
            return None
            
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create color palette with vibrant colors
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#1B998B', '#E63946', '#F77F00', '#06D6A0']
            while len(colors) < len(classes):
                colors.extend(colors)
            colors = colors[:len(classes)]
            
            bars = plt.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.2)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(count)}', ha='center', va='bottom', fontweight='bold')
            
            plt.title('Training Data Distribution', fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('Class', fontsize=12, fontweight='bold')
            plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            # Add percentage labels
            total_samples = sum(counts)
            for i, (bar, count) in enumerate(zip(bars, counts)):
                percentage = (count / total_samples) * 100
                plt.text(bar.get_x() + bar.get_width()/2., height/2,
                        f'{percentage:.1f}%', ha='center', va='center', 
                        fontweight='bold', color='white', fontsize=10)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            return None
    
    def create_feature_importance_plot(self, importances, feature_names):
        """Create enhanced feature importance plot for RF classifier"""
        if not self.has_dependencies:
            return None
            
        try:
            # Sort by importance
            indices = np.argsort(importances)[::-1]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create color gradient
            colors = plt.cm.viridis(np.linspace(0, 1, len(importances)))
            
            bars = plt.bar(range(len(indices)), importances[indices], 
                          color=colors[indices], edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for i, (bar, idx) in enumerate(zip(bars, indices)):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                        f'{importances[idx]:.3f}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=10)
            
            plt.xlabel('Spectral Bands', fontsize=12, fontweight='bold')
            plt.ylabel('Relative Importance', fontsize=12, fontweight='bold')
            plt.title('Feature Importance Analysis (Random Forest)\nGini-based Importance Scores', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            return fig
        except Exception as e:
            return None
    
    def create_accuracy_metrics_plot(self, precision, recall, f1, class_names):
        """Create grouped bar chart for precision, recall, and F1-score"""
        if not self.has_dependencies:
            return None
            
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            x = np.arange(len(class_names))
            width = 0.25
            
            # Updated colors - vibrant and modern
            bars1 = plt.bar(x - width, precision, width, label='Precision (User\'s Accuracy)', 
                           color='#2E86AB', edgecolor='black', linewidth=1)
            bars2 = plt.bar(x, recall, width, label='Recall (Producer\'s Accuracy)', 
                           color='#F18F01', edgecolor='black', linewidth=1)
            bars3 = plt.bar(x + width, f1, width, label='F1-Score', 
                           color='#1B998B', edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            def add_labels(bars, values):
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', 
                            fontweight='bold', fontsize=9)
            
            add_labels(bars1, precision)
            add_labels(bars2, recall)
            add_labels(bars3, f1)
            
            plt.xlabel('Classes', fontsize=12, fontweight='bold')
            plt.ylabel('Score', fontsize=12, fontweight='bold')
            plt.title('Classification Metrics by Class\n(Remote Sensing Terminology)', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.xticks(x, class_names, rotation=45, ha='right')
            plt.legend(fontsize=11)
            plt.grid(axis='y', alpha=0.3)
            plt.ylim(0, 1.1)
            plt.tight_layout()
            return fig
        except Exception as e:
            return None
    
    def generate_scientific_html_report(self, parameters, context, classifier, classes, X_test, y_test, 
                                      class_counts, confusion_mat, overall_accuracy, feature_importance=None,
                                      algorithm_type=0, source_image=None, output_image=None, n_bands=1):
        """Generate comprehensive HTML classification report with scientific rigor"""
        try:
            if not self.has_dependencies:
                return self._create_no_dependencies_report()
            
            from sklearn.metrics import precision_recall_fscore_support
            
            # Calculate scientific metrics
            y_pred = classifier.predict(X_test)
            scientific_metrics = self.calculate_scientific_metrics(y_test, y_pred, confusion_mat, class_counts, n_bands)
            
            # Enhanced HTML template with scientific rigor
            html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Classification Report - ArcGeek Calculator</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6; 
            color: #333;
            background-color: #f5f5f5;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{ 
            background: linear-gradient(135deg, #2E86AB 0%, #1B998B 100%);
            color: white; 
            padding: 30px; 
            border-radius: 10px; 
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .section {{ 
            background: white;
            margin-bottom: 30px; 
            padding: 25px; 
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }}
        
        .section:hover {{
            transform: translateY(-2px);
        }}
        
        .section h2 {{
            color: #4a5568;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 3px solid #2E86AB;
            padding-bottom: 10px;
        }}
        
        .metrics {{ 
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .metric-card {{ 
            background: linear-gradient(135deg, #F18F01 0%, #E63946 100%);
            color: white;
            padding: 25px; 
            border-radius: 10px; 
            text-align: center; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }}
        
        .metric-card:hover {{
            transform: scale(1.05);
        }}
        
        .metric-card h3 {{
            font-size: 1.2em;
            margin-bottom: 10px;
            opacity: 0.9;
        }}
        
        .metric-card p {{
            font-size: 2em;
            font-weight: bold;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        }}
        
        .scientific-metric {{
            background: linear-gradient(135deg, #2E86AB 0%, #1B998B 100%);
        }}
        
        table {{ 
            width: 100%;
            border-collapse: collapse; 
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        th, td {{ 
            padding: 12px 15px; 
            text-align: left;
            border: none;
        }}
        
        th {{ 
            background: linear-gradient(135deg, #2E86AB 0%, #1B998B 100%);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        tr:nth-child(even) {{ 
            background-color: #f8f9fa; 
        }}
        
        tr:hover {{
            background-color: #e3f2fd;
            transition: background-color 0.2s ease;
        }}
        
        .plot-container {{ 
            margin: 25px 0; 
            text-align: center;
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        .info-box {{
            background: linear-gradient(135deg, #e3f2fd 0%, #f0f7ff 100%);
            border-left: 5px solid #2E86AB;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        
        .info-box h4 {{
            color: #2E86AB;
            margin-bottom: 15px;
        }}
        
        .info-box ul {{
            margin-left: 20px;
        }}
        
        .info-box li {{
            margin-bottom: 8px;
        }}
        
        .footer {{ 
            text-align: center; 
            margin-top: 40px; 
            padding: 20px;
            background: #2d3748;
            color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .footer p {{
            margin-bottom: 10px;
        }}
        
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .stat-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #2E86AB;
        }}
        
        .stat-label {{
            font-weight: 600;
            color: #4a5568;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .stat-value {{
            font-size: 1.3em;
            font-weight: bold;
            color: #2d3748;
            margin-top: 5px;
        }}
        
        .warning {{
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            border-left: 5px solid #F77F00;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        
        .success {{
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border-left: 5px solid #1B998B;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .metrics {{
                grid-template-columns: 1fr;
            }}
            
            table {{
                font-size: 0.9em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Image Classification Report</h1>
            <p>Experimental version in development. If you find an error, please report it <a href="https://github.com/franzpc/ArcGeekCalculator/issues">here</a>.</p>
            <p>Generated on {date_time} | ArcGeek Calculator by <a href="https://x.com/franzpc/">franzpc</a>.</p>
            <p><em>Following remote sensing accuracy assessment standards.</em></p>
        </div>
        
        <div class="section">
            <h2>📊 Classification Summary</h2>
            <div class="metrics">
                <div class="metric-card">
                    <h3>Algorithm Used</h3>
                    <p>{algorithm}</p>
                </div>
                <div class="metric-card">
                    <h3>Overall Accuracy</h3>
                    <p>{accuracy:.2f}%</p>
                </div>
                <div class="metric-card scientific-metric">
                    <h3>Cohen's Kappa</h3>
                    <p>{kappa:.3f}</p>
                </div>
                <div class="metric-card scientific-metric">
                    <h3>Balanced Accuracy</h3>
                    <p>{balanced_accuracy:.2f}%</p>
                </div>
                <div class="metric-card">
                    <h3>Number of Classes</h3>
                    <p>{num_classes}</p>
                </div>
                <div class="metric-card">
                    <h3>Training Samples</h3>
                    <p>{total_samples}</p>
                </div>
            </div>
            
            <div class="summary-stats">
                <div class="stat-item">
                    <div class="stat-label">Kappa Interpretation</div>
                    <div class="stat-value">{kappa_interpretation}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Sample Size Adequacy</div>
                    <div class="stat-value">{sample_adequacy}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Spectral Bands Used</div>
                    <div class="stat-value">{n_bands} bands</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Classification Quality</div>
                    <div class="stat-value">{quality_assessment}</div>
                </div>
            </div>
            
            {sample_adequacy_warning}
        </div>
        
        <div class="section">
            <h2>🔬 Accuracy Assessment</h2>
            <div class="info-box">
                <h4>📖 Remote Sensing Accuracy Assessment Standards:</h4>
                <ul>
                    <li><strong>Overall Accuracy:</strong> Proportion of correctly classified pixels (diagonal sum / total)</li>
                    <li><strong>Cohen's Kappa (κ):</strong> Agreement measure accounting for chance agreement (Landis & Koch, 1977)</li>
                    <li><strong>Producer's Accuracy:</strong> Map accuracy from the point of view of the map maker (1 - Omission Error)</li>
                    <li><strong>User's Accuracy:</strong> Map accuracy from the point of view of the map user (1 - Commission Error)</li>
                    <li><strong>Balanced Accuracy:</strong> Average of recall scores per class, better for imbalanced datasets</li>
                </ul>
            </div>
            
            <div class="plot-container">
                <img src="data:image/png;base64,{kappa_plot}" alt="Kappa Interpretation Scale">
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Sample Size Adequacy Assessment</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{sample_adequacy_plot}" alt="Sample Size Adequacy">
            </div>
            <div class="info-box">
                <h4>📏 Sample Size Guidelines (Congalton & Green, 2019):</h4>
                <ul>
                    <li><strong>Minimum recommended:</strong> 10 samples per spectral band per class</li>
                    <li><strong>For {n_bands} bands:</strong> At least {recommended_minimum} samples per class</li>
                    <li><strong>Current minimum:</strong> {min_samples_per_class} samples in smallest class</li>
                    <li><strong>Assessment:</strong> {sample_adequacy_detailed}</li>
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Training Data Distribution</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{class_distribution_plot}" alt="Class Distribution">
            </div>
        </div>
        
        <div class="section">
            <h2>🎯 Enhanced Confusion Matrix & Accuracy Assessment</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{confusion_matrix_plot}" alt="Enhanced Confusion Matrix">
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Performance Metrics by Class</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{metrics_plot}" alt="Performance Metrics">
            </div>
        </div>
        
        {feature_importance_section}
        
        <div class="section">
            <h2>📋 Detailed Scientific Metrics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Class</th>
                        <th>Producer's Accuracy</th>
                        <th>User's Accuracy</th>
                        <th>F1-Score</th>
                        <th>Omission Error</th>
                        <th>Commission Error</th>
                        <th>Support</th>
                    </tr>
                </thead>
                <tbody>
                    {scientific_metrics_rows}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>📚 References</h2>
            <div class="info-box">
                <h4>📖 Key References for Classification Accuracy Assessment:</h4>
                <ul>
                    <li><strong>Congalton, R.G. & Green, K. (2019).</strong> <em>Assessing the Accuracy of Remotely Sensed Data: Principles and Practices.</em> CRC Press.</li>
                    <li><strong>Landis, J.R. & Koch, G.G. (1977).</strong> The measurement of observer agreement for categorical data. <em>Biometrics, 33</em>(1), 159-174.</li>
                    <li><strong>Foody, G.M. (2002).</strong> Status of land cover classification accuracy assessment. <em>Remote Sensing of Environment, 80</em>(1), 185-201.</li>
                    <li><strong>Stehman, S.V. (1997).</strong> Selecting and interpreting measures of thematic classification accuracy. <em>Remote Sensing of Environment, 62</em>(1), 77-89.</li>
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>🚀 Generated by ArcGeek Calculator Enhanced Classification Tool</strong></p>
            <p>Accuracy assessment following remote sensing standards</p>
            <p>For more tools and resources, visit <strong>www.arcgeek.com</strong></p>
            <p><em>Empowering GIS professionals with scientifically rigorous analytical capabilities</em></p>
        </div>
    </div>
</body>
</html>"""
            
            # Format class names and inverse mapping
            class_names = []
            inverse_classes = {}
            for class_title, class_id in classes.items():
                class_names.append(str(class_title))
                inverse_classes[class_id] = str(class_title)
                
            # Calculate per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
            
            # Find best performing class and calculate averages
            best_f1_idx = np.argmax(f1)
            best_class = class_names[best_f1_idx] if len(class_names) > best_f1_idx else "Unknown"
            avg_f1 = np.mean(f1)
            
            # Quality assessment based on both OA and Kappa
            kappa_value = scientific_metrics['kappa']
            if overall_accuracy >= 0.9 and kappa_value >= 0.8:
                quality_assessment = "Excellent"
            elif overall_accuracy >= 0.8 and kappa_value >= 0.6:
                quality_assessment = "Very Good"
            elif overall_accuracy >= 0.7 and kappa_value >= 0.4:
                quality_assessment = "Good"
            elif overall_accuracy >= 0.6 and kappa_value >= 0.2:
                quality_assessment = "Fair"
            else:
                quality_assessment = "Poor - Consider More Training Data"
            
            # Sample adequacy warning
            sample_adequacy_warning = ""
            if scientific_metrics['sample_adequacy'] == "Insufficient":
                sample_adequacy_warning = f"""
                <div class="warning">
                    <h4>⚠️ Sample Size Warning</h4>
                    <p>The training dataset may be insufficient for reliable classification. Consider collecting more training samples.</p>
                    <p><strong>Current:</strong> {scientific_metrics['min_samples_per_class']} samples in smallest class</p>
                    <p><strong>Recommended:</strong> At least {scientific_metrics['recommended_minimum']} samples per class</p>
                </div>
                """
            elif scientific_metrics['sample_adequacy'] == "Marginal":
                sample_adequacy_warning = f"""
                <div class="warning">
                    <h4>⚠️ Sample Size Caution</h4>
                    <p>The training dataset is marginally adequate. Results should be interpreted carefully.</p>
                    <p><strong>Current:</strong> {scientific_metrics['min_samples_per_class']} samples in smallest class</p>
                    <p><strong>Recommended:</strong> At least {scientific_metrics['recommended_minimum']} samples per class</p>
                </div>
                """
            else:
                sample_adequacy_warning = """
                <div class="success">
                    <h4>✅ Sample Size Adequate</h4>
                    <p>The training dataset meets recommended sample size guidelines for reliable classification.</p>
                </div>
                """
            
            # Sample adequacy detailed description
            sample_adequacy_detailed = {
                "Adequate": "Meets or exceeds recommended guidelines",
                "Marginal": "Below recommended but may be acceptable",
                "Insufficient": "Below minimum standards - collect more samples"
            }[scientific_metrics['sample_adequacy']]
            
            # Generate scientific metrics table rows
            scientific_metrics_rows = ""
            producers_acc = scientific_metrics['producers_accuracy']
            users_acc = scientific_metrics['users_accuracy']
            omission_err = scientific_metrics['omission_error']
            commission_err = scientific_metrics['commission_error']
            
            for i, class_id in enumerate(sorted(classes.values())):
                class_name = inverse_classes[class_id]
                class_index = list(classes.values()).index(class_id)
                
                if class_index < len(producers_acc):
                    scientific_metrics_rows += f"""
                    <tr>
                        <td><strong>{class_name}</strong></td>
                        <td>{producers_acc[class_index]:.3f}</td>
                        <td>{users_acc[class_index]:.3f}</td>
                        <td>{f1[class_index]:.3f}</td>
                        <td>{omission_err[class_index]:.3f}</td>
                        <td>{commission_err[class_index]:.3f}</td>
                        <td>{support[class_index]}</td>
                    </tr>
                    """
            
            # Create plots
            cm_fig = self.create_enhanced_confusion_matrix_plot(confusion_mat, class_names, producers_acc, users_acc)
            cm_base64 = self.figure_to_base64(cm_fig) if cm_fig else ""
            
            cls_fig = self.create_class_distribution_plot(class_names, class_counts)
            cls_base64 = self.figure_to_base64(cls_fig) if cls_fig else ""
            
            metrics_fig = self.create_accuracy_metrics_plot(precision, recall, f1, class_names)
            metrics_base64 = self.figure_to_base64(metrics_fig) if metrics_fig else ""
            
            kappa_fig = self.create_kappa_interpretation_plot(kappa_value)
            kappa_base64 = self.figure_to_base64(kappa_fig) if kappa_fig else ""
            
            sample_fig = self.create_sample_adequacy_plot(class_names, class_counts, n_bands)
            sample_base64 = self.figure_to_base64(sample_fig) if sample_fig else ""
            
            # Feature importance section (only for Random Forest)
            feature_importance_section = ""
            if feature_importance is not None and algorithm_type == 0:
                feature_names = [f"Band {i+1}" for i in range(len(feature_importance))]
                fi_fig = self.create_feature_importance_plot(feature_importance, feature_names)
                fi_base64 = self.figure_to_base64(fi_fig) if fi_fig else ""
                
                # Find most important feature
                most_important_idx = np.argmax(feature_importance)
                most_important_feature = feature_names[most_important_idx]
                
                feature_importance_section = f"""
                <div class="section">
                    <h2>🔍 Spectral Band Importance Analysis</h2>
                    <div class="summary-stats">
                        <div class="stat-item">
                            <div class="stat-label">Most Important Band</div>
                            <div class="stat-value">{most_important_feature}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Importance Score</div>
                            <div class="stat-value">{feature_importance[most_important_idx]:.3f}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Least Important Band</div>
                            <div class="stat-value">{feature_names[np.argmin(feature_importance)]}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Importance Range</div>
                            <div class="stat-value">{np.max(feature_importance) - np.min(feature_importance):.3f}</div>
                        </div>
                    </div>
                    <div class="plot-container">
                        <img src="data:image/png;base64,{fi_base64}" alt="Feature Importance">
                    </div>
                    <div class="info-box">
                        <h4>🔍 Feature Importance Interpretation:</h4>
                        <ul>
                            <li><strong>Gini-based importance:</strong> Measures how much each band contributes to decreasing node impurity</li>
                            <li><strong>Higher values:</strong> Indicate bands that are more useful for classification</li>
                            <li><strong>Lower values:</strong> May indicate redundant or noisy bands</li>
                            <li><strong>Band selection:</strong> Consider focusing on high-importance bands for future classifications</li>
                        </ul>
                    </div>
                </div>
                """
            
            # Get algorithm name
            algorithm_names = ["Random Forest", "Gaussian Mixture Model", "Support Vector Machine", "K-Nearest Neighbors"]
            algorithm_name = algorithm_names[algorithm_type]
            
            # Fill in the template
            html_content = html_template.format(
                date_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                algorithm=algorithm_name,
                accuracy=overall_accuracy * 100,
                kappa=kappa_value,
                balanced_accuracy=scientific_metrics['balanced_accuracy'] * 100,
                num_classes=len(classes),
                total_samples=sum(class_counts),
                kappa_interpretation=scientific_metrics['kappa_interpretation'],
                sample_adequacy=scientific_metrics['sample_adequacy'],
                n_bands=n_bands,
                quality_assessment=quality_assessment,
                sample_adequacy_warning=sample_adequacy_warning,
                recommended_minimum=scientific_metrics['recommended_minimum'],
                min_samples_per_class=scientific_metrics['min_samples_per_class'],
                sample_adequacy_detailed=sample_adequacy_detailed,
                class_distribution_plot=cls_base64,
                confusion_matrix_plot=cm_base64,
                metrics_plot=metrics_base64,
                kappa_plot=kappa_base64,
                sample_adequacy_plot=sample_base64,
                feature_importance_section=feature_importance_section,
                scientific_metrics_rows=scientific_metrics_rows
            )
            
            return html_content
            
        except Exception as e:
            return self._create_error_report(str(e))
    
    def _create_no_dependencies_report(self):
        """Create a simple report when plotting dependencies are missing"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Classification Report - Limited</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 20px; max-width: 800px; margin: 0 auto; }
                .warning { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .info { background-color: #d1ecf1; border: 1px solid #bee5eb; padding: 15px; border-radius: 5px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Scientific Image Classification Report</h1>
            <div class="warning">
                <h3>⚠️ Limited Report</h3>
                <p>This is a simplified report because matplotlib, seaborn, and/or scikit-learn are not fully available.</p>
                <p>To get the full scientific report with charts, statistical analysis, and visualizations, please install:</p>
                <code>pip install matplotlib seaborn scikit-learn</code>
            </div>
            <div class="info">
                <h3>✅ Classification Completed Successfully</h3>
                <p>Your image classification has been completed successfully. The classified image has been saved to your specified location.</p>
                <p>Install the missing dependencies to get detailed scientific metrics, confusion matrices, Cohen's Kappa analysis, and feature importance charts.</p>
            </div>
        </body>
        </html>
        """
    
    def _create_error_report(self, error_message):
        """Create an error report with detailed information"""
        import traceback
        error_details = traceback.format_exc()
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Classification Report Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; max-width: 1000px; margin: 0 auto; }}
                .error {{ color: #d32f2f; background-color: #ffebee; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .details {{ background-color: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; overflow-x: auto; }}
                pre {{ white-space: pre-wrap; word-wrap: break-word; }}
                .success {{ color: #2e7d32; background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Scientific Classification Report Generation Error</h1>
            <div class="success">
                <h3>✅ Classification Completed Successfully</h3>
                <p>Your image classification has been completed successfully and the classified image has been saved.</p>
            </div>
            <div class="error">
                <h3>❌ Report Generation Failed</h3>
                <p><strong>Error:</strong> {error_message}</p>
                <p>The classification process completed successfully, but there was an issue generating the detailed scientific report.</p>
            </div>
            <div class="details">
                <h3>Technical Details:</h3>
                <pre>{error_details}</pre>
            </div>
            <p><em>Generated by ArcGeek Calculator Enhanced Classification Tool</em></p>
        </body>
        </html>
        """

def generate_scientific_classification_report(parameters, context, classifier, classes, X_test, y_test, 
                                             class_counts, confusion_mat, overall_accuracy, feature_importance=None,
                                             algorithm_type=0, source_image=None, output_image=None, n_bands=1):
    """
    Convenience function to generate scientific classification report
    
    Parameters:
    -----------
    parameters : dict
        Processing parameters
    context : QgsProcessingContext
        Processing context
    classifier : sklearn classifier
        Trained classifier object
    classes : dict
        Class mapping dictionary
    X_test, y_test : array-like
        Test data and labels
    class_counts : array-like
        Count of samples per class
    confusion_mat : array-like
        Confusion matrix
    overall_accuracy : float
        Overall accuracy score
    feature_importance : array-like, optional
        Feature importance scores (for Random Forest)
    algorithm_type : int
        Algorithm type index (0=RF, 1=GMM, 2=SVM, 3=KNN)
    source_image : str, optional
        Path to source image
    output_image : str, optional
        Path to output image
    n_bands : int
        Number of spectral bands in the source image
        
    Returns:
    --------
    str
        HTML content for the scientific classification report
    """
    generator = ScientificClassificationReportGenerator()
    return generator.generate_scientific_html_report(
        parameters, context, classifier, classes, X_test, y_test,
        class_counts, confusion_mat, overall_accuracy, feature_importance,
        algorithm_type, source_image, output_image, n_bands
    )