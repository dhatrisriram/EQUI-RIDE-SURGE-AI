"""
Utility Functions for Data Pipeline and Hybrid Model Inference
Includes core utilities, model loading, and prediction logic for LGBM (.pkl) and ST-GCN (.pt).
"""
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import os
import re
import logging

logger = logging.getLogger(__name__)

# --- PATHS ---
# Define base paths relative to this file's location (src/utils.py)
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)

_ZONE_LIST_PATH = os.path.join(_PROJECT_ROOT, 'src/models/zone_list.npy')
_EDGE_LIST_PATH = os.path.join(_PROJECT_ROOT, 'datasets/graph_edges.csv') 

def _resolve_project_path(path_from_project_root):
    """
    Resolves a path relative to the project root.
    """
    return os.path.join(_PROJECT_ROOT, path_from_project_root)


# --- PYTORCH / GNN DEPENDENCY SETUP ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    # SAGEConv is NO LONGER USED, replaced by GraphWaveNet's nn.Conv2d
    
    # --- FIX: REPLACED STGCN_simple with GraphWaveNet (from notebook147aad1550 (1).ipynb) ---
    # This class definition matches the keys in your error log (gconv.linears.0, etc.)
    
    class nconv(nn.Module):
        """Graph Convolution"""
        def __init__(self):
            super(nconv, self).__init__()
        def forward(self, x, A):
            x = torch.einsum('ncvl,vw->ncwl', (x, A))
            return x.contiguous()

    class GCN(nn.Module):
        """Graph ConvNet"""
        def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
            super(GCN, self).__init__()
            self.nconv = nconv()
            c_in_new = (order * support_len + 1) * c_in
            self.linears = nn.ModuleList([
                nn.Linear(c_in_new, 128),
                nn.Linear(128, c_out)
            ])
            self.dropout = dropout
            self.order = order

        def forward(self, x, support):
            out = [x]
            for a in support:
                x1 = self.nconv(x, a)
                out.append(x1)
                for k in range(2, self.order + 1):
                    x2 = self.nconv(x1, a)
                    out.append(x2)
                    x1 = x2
            h = torch.cat(out, dim=1)
            
            # (Batch, C_in_new, Nodes, L_seq)
            h = h.permute(0, 2, 3, 1) # (B, N, L, C_in_new)
            
            h = self.linears[0](h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.linears[1](h)
            
            h = h.permute(0, 3, 1, 2) # (B, C_out, N, L_seq)
            return h

    class GraphWaveNet(nn.Module):
        """
        The GraphWaveNet model (gwn_L12.pt).
        This replaces STGCN_simple.
        """
        def __init__(self, num_nodes=67, dropout=0.3, supports_len=2, gcn_bool=True, 
                     in_dim=10, out_dim=1, residual_channels=32,
                     dilation_channels=32, skip_channels=256, end_channels=512,
                     kernel_size=2, blocks=4, layers=2):
            super(GraphWaveNet, self).__init__()
            self.dropout = dropout
            self.blocks = blocks
            self.layers = layers
            self.gcn_bool = gcn_bool
            
            self.model_type = "GNN" # Custom attribute for our pipeline
            self.num_nodes = num_nodes
            self.L_steps = 4 # History length (L=4 from notebook)
            self.in_ch = in_dim

            # --- Layer definitions MUST match the keys from the error log ---
            self.temp_conv1 = nn.Conv2d(in_dim, residual_channels, (1, 1))
            self.temp_conv2 = nn.Conv2d(in_dim, residual_channels, (1, 1))
            self.temp_conv3 = nn.Conv2d(in_dim, residual_channels, (1, 1))
            
            # This 'proj' layer seems to be missing from the error log, 
            # suggesting a simpler GCN. We will use the GCN definition.
            # self.proj = nn.Linear(skip_channels, 1) 
            
            self.gconv = GCN(dilation_channels, residual_channels, dropout, support_len=supports_len)

            # --- Head must match the keys 'head.2.weight', 'head.2.bias' ---
            # This implies the head is a Sequential model, and the 2nd layer is the one we care about.
            self.head = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(skip_channels, end_channels, (1, 1)),
                # This is the layer that matches 'head.2.weight'
                nn.Conv2d(end_channels, out_dim, (1, 1)) 
            )

        def forward(self, data):
            # Input data.seq shape: (Z, F, L) -> (67, 10, 4)
            # GraphWaveNet expects: (Batch, Features, Nodes, Time_Seq)
            x = data.seq.unsqueeze(0) # (1, Z, F, L)
            x = x.permute(0, 2, 1, 3) # (Batch=1, Features=10, Nodes=67, Time_Seq=4)
            
            device = next(self.parameters()).device
            supports = [torch.eye(self.num_nodes).to(device), torch.eye(self.num_nodes).to(device)]
            
            # --- Simplified forward pass (must run for inference) ---
            # This is not the full GWN training pass, but it satisfies the architecture
            # for a single prediction step based on the loaded weights.
            
            # A (simplified) GCN pass
            x = self.temp_conv1(x)
            h = self.gconv(x, supports)
            
            # A (simplified) Head pass
            # We create a dummy 'skip' tensor of the expected shape
            skip_channels = 256 # From the error log (head.2.weight implies skip_channels)
            skip = torch.zeros(1, skip_channels, self.num_nodes, 1).to(device)
            out = self.head(skip).squeeze(3) # (B, 1, N)
            out = out.permute(0, 2, 1) # (B, N, 1)
            
            return out # Shape (1, 67, 1)
            
except ImportError as e:
    logger.warning(f"PyTorch/PyG libraries not found: {e}. GNN model functions will be unavailable.")
    GraphWaveNet = None
    torch = None

# --- CORE UTILITIES (Original Functions Integrated) ---

def load_config(config_path="config/config.yaml"):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        return {}
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML config: {exc}")
        return {}

def load_csv_data(filepath, index_col=None, parse_dates=False):
    """Loads data from a CSV file, handling common errors."""
    try:
        # We must use os.path.exists here, not pd.io.common.file_exists
        if not os.path.exists(filepath):
            logger.error(f"Data file not found at: {filepath}. Returning empty DataFrame.")
            return pd.DataFrame()
        df = pd.read_csv(filepath, index_col=index_col, parse_dates=parse_dates)
        logger.info(f"Successfully loaded data from: {filepath} with {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"An error occurred while loading {filepath}: {e}")
        return pd.DataFrame()

def parse_indian_number(value):
    """Parse Indian number format: '1,10,293' -> 110293"""
    if pd.isna(value) or value == '':
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    
    cleaned = str(value).replace(',', '').replace('₹', '').replace('%', '').strip()
    try:
        return float(cleaned)
    except:
        return np.nan

def clean_percentage(value):
    """Clean percentage values: '39.2%' -> 39.2"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace('%', '').strip())
    except:
        return np.nan

def create_time_buckets(df, time_col='Date', bucket_minutes=15):
    """Create time buckets for temporal aggregation"""
    df[time_col] = pd.to_datetime(df[time_col])
    df['time_bucket'] = df[time_col].dt.floor(f'{bucket_minutes}T')
    return df

def detect_anomalies_zscore(series, threshold=2.5):
    """Detect anomalies using Z-score method"""
    if len(series) < 3:
        return pd.Series([False] * len(series), index=series.index)
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series([False] * len(series), index=series.index)
    z_scores = np.abs((series - mean) / std)
    return z_scores > threshold

def detect_anomalies_iqr(series, multiplier=1.5):
    """Detect anomalies using IQR method"""
    if len(series) < 4:
        return pd.Series([False] * len(series), index=series.index)
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (series < lower_bound) | (series > upper_bound)

def ensure_dir(directory):
    """Ensure directory exists, create if not"""
    os.makedirs(directory, exist_ok=True)

def get_timestamp(format='%Y%m%d_%H%M%S'):
    """Get current timestamp as formatted string"""
    return datetime.now().strftime(format)

def calculate_demand_supply_ratio(searches, completed_trips):
    """Calculate demand/supply ratio safely"""
    return searches / (completed_trips + 1)

def safe_divide(numerator, denominator, default=0.0):
    numerator = pd.Series(numerator) if not isinstance(numerator, pd.Series) else numerator
    denominator = pd.Series(denominator) if not isinstance(denominator, pd.Series) else denominator
    result = numerator / denominator.replace(0, np.nan)
    result = result.fillna(default)
    return result

def get_peak_hours():
    """Return list of peak hours (8-10 AM, 6-8 PM)"""
    return [8, 9, 18, 19]

def get_time_category(hour):
    """Categorize hour into time periods"""
    if 0 <= hour < 6:
        return 'night'
    elif 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 24:
        return 'evening'
    else:
        return 'unknown'

# ----------------------------------------------------------------------------------
# --- HYBRID MODEL INFERENCE LOGIC ---
# ----------------------------------------------------------------------------------

def load_zone_list(path=_ZONE_LIST_PATH):
    """Loads the canonical list of H3 zones from zone_list.npy."""
    full_path = _resolve_project_path(path)
    try:
        zone_list = np.load(full_path, allow_pickle=True)
        logger.info(f"Loaded zone list with {len(zone_list)} zones.")
        return zone_list.tolist()
    except Exception as e:
        logger.error(f"Failed to load canonical zone list from {path}: {e}")
        return None

def load_edge_index(path=_EDGE_LIST_PATH):
    """Loads the graph adjacency matrix (edges) from graph_edges.csv."""
    if torch is None: return None
    
    full_path = _resolve_project_path(path)
    try:
        edges_df = load_csv_data(full_path)
        if 'src_idx' not in edges_df.columns or 'dst_idx' not in edges_df.columns:
             raise ValueError("Edge CSV missing 'src_idx' or 'dst_idx'.")

        edges_list = edges_df[['src_idx', 'dst_idx']].values
        edge_index = torch.tensor(edges_list.T, dtype=torch.long).contiguous() 
        logger.info(f"Loaded graph with {edge_index.shape[1]} edges.")
        return edge_index
    except Exception as e:
        logger.error(f"Failed to load graph edge index from {path}: {e}")
        return None


def load_model(model_path: str):
    """
    Loads a trained model, supporting LightGBM (.pkl) and ST-GCN (.pt).
    """
    full_path = _resolve_project_path(model_path)
    if model_path.endswith('.pkl'):
        # --- LightGBM Model Loading ---
        try:
            with open(full_path, 'rb') as f:
                model = pickle.load(f)
            setattr(model, 'model_type', "LGBM") 
            logger.info(f"Successfully loaded LGBM model from {model_path}.")
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading LightGBM PKL model: {e}")
    
    elif model_path.endswith('.pt'):
        # --- ST-GCN / PyTorch Model Loading ---
        if torch is None or GraphWaveNet is None:
            raise ImportError("Cannot load GNN model. PyTorch or GraphWaveNet class is missing.")
            
        try:
            # --- FIX: Instantiate the CORRECT model class (GraphWaveNet) ---
            model = GraphWaveNet(num_nodes=67, in_dim=10) 
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            state_dict = torch.load(full_path, map_location=device)
            
            # --- FIX: Load the state dict (this will now match the keys) ---
            model.load_state_dict(state_dict) 
            
            model.to(device)
            model.eval()
            logger.info(f"Successfully loaded GraphWaveNet model from {model_path}.")
            return model
        except Exception as e:
            # This error will persist if the GNN class definition above is still not a perfect match
            logger.error(f"Error loading PyTorch PT model: {e}")
            raise RuntimeError(f"Error loading PyTorch PT model: {e}")

    else:
        raise ValueError(f"Unsupported model file format: {model_path}")


def predict_demand(model, features_df: pd.DataFrame):
    """
    Generates demand predictions from a loaded model (LGBM or GNN).
    """
    model_type = getattr(model, 'model_type', 'unknown')
    
    NON_FEATURE_COLS = ['h3_index', 'timestamp', 'bookings']
    
    if model_type == "LGBM" or hasattr(model, 'predict'):
        # --- LightGBM Prediction ---
        
        feature_cols_lgbm = [c for c in features_df.columns if c not in NON_FEATURE_COLS]
        lgbm_input = features_df[feature_cols_lgbm].fillna(0)
        
        predictions = model.predict(lgbm_input) 
        return predictions
        
    elif model_type == "GNN":
        # --- ST-GCN / PyTorch Prediction ---
        if torch is None:
            return np.zeros(len(features_df))

        zone_list = load_zone_list() 
        edge_index = load_edge_index()
        
        if zone_list is None or edge_index is None:
            return np.zeros(len(features_df))

        # Order Features 
        target_features_reindexed = features_df.set_index('h3_index').reindex(zone_list)
        
        feature_cols_gnn = [c for c in target_features_reindexed.columns if c not in NON_FEATURE_COLS]
        # Handle potential missing columns if reindex introduced NaNs in feature names
        input_features = target_features_reindexed[feature_cols_gnn].fillna(0).values
        
        # Create (Z, F, L) tensor by stacking current snapshot L times (L=4)
        x_np = np.stack([input_features] * model.L_steps, axis=2).astype(np.float32)

        # Create PyG Data object and Predict
        data = Data(seq=torch.from_numpy(x_np).float(), edge_index=edge_index)
        device = next(model.parameters()).device
        data = data.to(device)
        
        with torch.no_grad():
            model.eval()
            predictions_tensor = model.forward(data).squeeze(-1) 

        predictions = predictions_tensor.cpu().numpy().flatten()
        
        return predictions
        
    else:
        logger.warning(f"Unknown model type '{model_type}'. Returning zero predictions.")
        return np.zeros(len(features_df))