import os
import json
import pandas as pd
from pathlib import Path
import dataframe_image as dfi

def find_method_dirs(root_dir):
    """Recursively locate method directories (egnn/seed_gnn)"""
    method_dirs = []
    for path in Path(root_dir).rglob('*'):
        if path.is_dir() and path.name in ['egnn', 'seed_gnn']:
            method_dirs.append(path)
    return method_dirs

def collect_results(root_dir):
    method_dirs = find_method_dirs(root_dir)
    if not method_dirs:
        return pd.DataFrame()
    
    models = ['gat', 'gcn', 'gin', 'sage']
    datasets = ['amazoncomputers', 'amazonphoto', 'arxiv', 'coauthorcs', 'cora']
    
    # First pass: Discover all metric columns
    metric_columns = set()
    for method_dir in method_dirs:
        for model in models:
            for dataset in datasets:
                json_path = method_dir / model / dataset / 'output_config.json'
                if json_path.exists():
                    try:
                        with open(json_path) as f:
                            results = json.load(f).get('eval_results', {})
                            metric_columns.update(results.keys())
                    except:
                        pass
    metric_columns = sorted(metric_columns)
    
    # Second pass: Collect data for all combinations
    data = []
    for method_dir in method_dirs:
        method_name = method_dir.name.upper()
        
        for model in models:
            for dataset in datasets:
                json_path = method_dir / model / dataset / 'output_config.json'
                row = {
                    'Method': method_name,
                    'Model': model.upper(),
                    'Dataset': dataset
                }
                metrics = {col: 'OOM' for col in metric_columns}
                
                if json_path.exists():
                    try:
                        with open(json_path) as f:
                            results = json.load(f).get('eval_results', {})
                            metrics.update({k: v for k, v in results.items() if k in metric_columns})
                    except:
                        pass  # Keep OOM values if JSON is corrupted
                
                row.update(metrics)
                data.append(row)
    
    return pd.DataFrame(data)

def export_table_image(df, output_path="results.png"):
    # Style DataFrame with OOM highlighting
    styled_df = df.style.applymap(
        lambda x: 'background-color: #ffcccc' if x == 'OOM' else ''
    ).set_properties(**{'text-align': 'center'})
    
    # Export as image
    dfi.export(
        styled_df,
        output_path,
        dpi=300,
        fontsize=10,
        table_conversion='matplotlib'
    )
    print(f"Table image saved to {output_path}")

if __name__ == "__main__":
    root_folder = "output_edit_reg_class_aware_beta"
    df = collect_results(root_folder)
    
    if not df.empty:
        output_filename = root_folder.replace("output_edit_","")
        output_path = os.path.join("visualizations", f"{output_filename}.png")

        export_table_image(df, output_path)
        print("\nData preview:")
        print(df.head().to_string(index=False))
    else:
        print("No valid results found in directory structure")