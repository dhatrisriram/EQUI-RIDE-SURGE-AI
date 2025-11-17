"""
Utility Functions for Data Pipeline and Hybrid Model Inference
Includes core utilities, model loading, and prediction logic for LGBM (.pkl) and ST-GCN (.pt).
"""
import pickle
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import os
import re
import logging
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# --- PATHS ---
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)

_ZONE_LIST_PATH = os.path.join(_PROJECT_ROOT, 'src/models/zone_list.npy')
_EDGE_LIST_PATH = os.path.join(_PROJECT_ROOT, 'datasets/graph_edges.csv') 

def _resolve_project_path(path_from_project_root):
    return os.path.join(_PROJECT_ROOT, path_from_project_root)

# --- PYTORCH / GNN DEPENDENCY SETUP ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data

    class DiffusionGraphConvVec(nn.Module):
        """Graph Convolution Layer for GraphWaveNet (Vectorized)."""
        def __init__(self, in_ch, out_ch, K=2):
            super(DiffusionGraphConvVec, self).__init__()
            self.K = K
            self.linears = nn.ModuleList([nn.Linear(in_ch, out_ch) for _ in range(K+1)])
            
        def forward(self, x, A_norm):
            B, Z, C = x.shape
            out = torch.zeros(B, Z, self.linears[0].out_features, device=x.device)
            Xk = x
            for k in range(self.K + 1):
                t = self.linears[k](Xk.reshape(B*Z, C)).reshape(B, Z, -1)
                out = out + t
                if k < self.K:
                    Xk = torch.einsum('ij,bjk->bik', A_norm, Xk) 
            return out

    class GraphWaveNet(nn.Module):
        """GraphWaveNet model definition matching the saved checkpoint."""
        def __init__(self, num_nodes=67, in_dim=10, residual_channels=64, end_channels=128, dropout=0.2, L_steps=4):
            super(GraphWaveNet, self).__init__()
            self.num_nodes = num_nodes
            self.in_dim = in_dim
            self.temp_conv1 = nn.Conv1d(in_dim, residual_channels, kernel_size=3, padding=1, dilation=1)
            self.temp_conv2 = nn.Conv1d(residual_channels, residual_channels, kernel_size=3, padding=2, dilation=2)
            self.temp_conv3 = nn.Conv1d(residual_channels, residual_channels, kernel_size=3, padding=4, dilation=4)
            self.proj = nn.Linear(residual_channels, end_channels)
            self.gconv = DiffusionGraphConvVec(in_ch=end_channels, out_ch=end_channels, K=2)
            self.head = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(end_channels, 1)
            )
            self.register_buffer('A_norm', self._build_adjacency())

        def _build_adjacency(self):
            try:
                if not os.path.exists(_EDGE_LIST_PATH):
                    logger.warning(f"Edge list not found at {_EDGE_LIST_PATH}. Using Identity matrix.")
                    return torch.eye(self.num_nodes)
                edges_df = pd.read_csv(_EDGE_LIST_PATH)
                if 'src_idx' in edges_df.columns:
                     src = edges_df['src_idx'].values
                     dst = edges_df['dst_idx'].values
                else:
                     src = edges_df.iloc[:, 0].values
                     dst = edges_df.iloc[:, 1].values
                Z = self.num_nodes
                A = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(Z, Z))
                A = A + A.T
                A.data = np.ones_like(A.data)
                A.setdiag(0)
                A.eliminate_zeros()
                deg = np.array(A.sum(axis=1)).flatten()
                deg_inv_sqrt = np.power(deg, -0.5, where=deg>0)
                deg_inv_sqrt[~np.isfinite(deg_inv_sqrt)] = 0.0
                D_inv_sqrt = sp.diags(deg_inv_sqrt)
                A_norm = D_inv_sqrt.dot(A).dot(D_inv_sqrt).tocoo()
                return torch.from_numpy(A_norm.toarray()).float()
            except Exception as e:
                logger.error(f"Failed to build adjacency matrix: {e}")
                return torch.eye(self.num_nodes)

        def forward(self, data):
            if not hasattr(data, 'seq'): return None
            x = data.seq.unsqueeze(0) 
            B, Z, C, L = x.shape
            x = x.reshape(B*Z, C, L) 
            x = self.temp_conv1(x)
            x = F.relu(x)
            x = self.temp_conv2(x)
            x = F.relu(x)
            x = self.temp_conv3(x)
            x = F.relu(x)
            x = x.mean(dim=2) 
            x = x.reshape(B, Z, -1)
            x = self.proj(x)
            x = self.gconv(x, self.A_norm)
            out = self.head(x).squeeze(-1)
            return out.squeeze(0)

except ImportError as e:
    logger.warning(f"PyTorch/PyG libraries not found: {e}. GNN model functions will be unavailable.")
    GraphWaveNet = None
    torch = None

# --- CORE UTILITIES ---

def load_config(config_path="config/config.yaml"):
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
    try:
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
    if pd.isna(value) or value == '': return np.nan
    if isinstance(value, (int, float)): return float(value)
    cleaned = str(value).replace(',', '').replace('₹', '').replace('%', '').strip()
    try: return float(cleaned)
    except: return np.nan

def clean_percentage(value):
    if pd.isna(value): return np.nan
    if isinstance(value, (int, float)): return float(value)
    try: return float(str(value).replace('%', '').strip())
    except: return np.nan

def create_time_buckets(df, time_col='Date', bucket_minutes=15):
    df[time_col] = pd.to_datetime(df[time_col])
    df['time_bucket'] = df[time_col].dt.floor(f'{bucket_minutes}T')
    return df

def detect_anomalies_zscore(series, threshold=2.5):
    if len(series) < 3: return pd.Series([False] * len(series), index=series.index)
    mean = series.mean(); std = series.std()
    if std == 0: return pd.Series([False] * len(series), index=series.index)
    z_scores = np.abs((series - mean) / std)
    return z_scores > threshold

def detect_anomalies_iqr(series, multiplier=1.5):
    if len(series) < 4: return pd.Series([False] * len(series), index=series.index)
    Q1 = series.quantile(0.25); Q3 = series.quantile(0.75); IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR; upper = Q3 + multiplier * IQR
    return (series < lower) | (series > upper)

def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)

def get_timestamp(format='%Y%m%d_%H%M%S'):
    return datetime.now().strftime(format)

def calculate_demand_supply_ratio(searches, completed_trips):
    return searches / (completed_trips + 1)

def safe_divide(numerator, denominator, default=0.0):
    numerator = pd.Series(numerator) if not isinstance(numerator, pd.Series) else numerator
    denominator = pd.Series(denominator) if not isinstance(denominator, pd.Series) else denominator
    result = numerator / denominator.replace(0, np.nan)
    result = result.fillna(default)
    return result

def get_peak_hours(): return [8, 9, 18, 19]

def get_time_category(hour):
    if 0 <= hour < 6: return 'night'
    elif 6 <= hour < 12: return 'morning'
    elif 12 <= hour < 18: return 'afternoon'
    elif 18 <= hour < 24: return 'evening'
    else: return 'unknown'

# --- MODEL INFERENCE LOGIC ---

def load_zone_list(path=_ZONE_LIST_PATH):
    full_path = _resolve_project_path(path)
    try:
        zone_list = np.load(full_path, allow_pickle=True)
        logger.info(f"Loaded zone list with {len(zone_list)} zones.")
        return zone_list.tolist()
    except Exception as e:
        logger.error(f"Failed to load canonical zone list from {path}: {e}")
        return None

def load_edge_index(path=_EDGE_LIST_PATH):
    if torch is None: return None
    full_path = _resolve_project_path(path)
    try:
        edges_df = load_csv_data(full_path)
        if 'src_idx' not in edges_df.columns or 'dst_idx' not in edges_df.columns:
             return None
        edges_list = edges_df[['src_idx', 'dst_idx']].values
        edge_index = torch.tensor(edges_list.T, dtype=torch.long).contiguous() 
        return edge_index
    except Exception as e:
        logger.error(f"Failed to load graph edge index: {e}")
        return None

def load_model(model_path: str):
    """
    Loads a trained model, supporting LightGBM (.pkl) and ST-GCN (.pt).
    Safely handles potential tuple-wrapped models.
    """
    full_path = _resolve_project_path(model_path)
    config = load_config(os.path.join(_PROJECT_ROOT, 'config/config.yaml'))
    
    if model_path.endswith('.pkl'):
        # --- LightGBM Model Loading ---
        model = None
        try:
            model = joblib.load(full_path)
            logger.info(f"Loaded LGBM model via joblib from {model_path}")
        except Exception as e1:
            try:
                with open(full_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Loaded LGBM model via pickle from {model_path}")
            except Exception as e2:
                raise RuntimeError(f"Failed to load LGBM model: {e1}, {e2}")
        
        # Unpack if tuple
        if isinstance(model, tuple):
            logger.info("Loaded object is a tuple, extracting first element as model.")
            model = model[0]
            
        setattr(model, 'model_type', "LGBM")
        return model
    
    elif model_path.endswith('.pt'):
        # --- ST-GCN / PyTorch Model Loading ---
        if torch is None or GraphWaveNet is None:
            raise ImportError("Cannot load GNN model. PyTorch or GraphWaveNet class is missing.")
            
        try:
            gnn_config = config.get('model', {}).get('gnn', {})
            
            model = GraphWaveNet(
                num_nodes=67, 
                in_dim=10,
                residual_channels=gnn_config.get('residual_channels', 64),
                end_channels=gnn_config.get('end_channels', 128),
                dropout=gnn_config.get('dropout', 0.2),
                L_steps=4
            )
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            state_dict = torch.load(full_path, map_location=device)
            model.load_state_dict(state_dict, strict=False) 
            
            model.to(device)
            model.eval()
            setattr(model, 'model_type', 'GNN')
            logger.info(f"Successfully loaded GraphWaveNet model from {model_path}.")
            return model
        except Exception as e:
            logger.error(f"Error loading PyTorch PT model: {e}")
            raise RuntimeError(f"Error loading PyTorch PT model: {e}")

    else:
        raise ValueError(f"Unsupported model file format: {model_path}")


def predict_demand(model, features_df: pd.DataFrame):
    """
    Generates demand predictions from a loaded model (LGBM or GNN).
    Features passed in 'features_df' should already be filtered to match model expectations.
    """
    model_type = getattr(model, 'model_type', 'unknown')
    
    # Columns that are NOT features but metadata
    NON_FEATURE_COLS = ['h3_index', 'timestamp', 'bookings', 'datetime', 'date', 'time']
    
    if model_type == "LGBM" or hasattr(model, 'predict'):
        # --- LightGBM Prediction ---
        # Expects input dataframe to contain only feature columns + metadata
        # Extract only numeric feature columns for safety if not pre-filtered
        cols_to_use = [c for c in features_df.columns if c not in NON_FEATURE_COLS]
        
        lgbm_input = features_df[cols_to_use].fillna(0)
        try:
            predictions = model.predict(lgbm_input)
            return predictions
        except Exception as e:
            logger.error(f"LGBM prediction execution failed: {e}")
            return np.zeros(len(features_df))
        
    elif model_type == "GNN":
        # --- ST-GCN / PyTorch Prediction ---
        if torch is None:
            return np.zeros(len(features_df))

        zone_list = load_zone_list() 
        if zone_list is None:
            return np.zeros(len(features_df))

        # Re-index to ensure order matches adjacency matrix
        target_features_reindexed = features_df.set_index('h3_index').reindex(zone_list)
        
        cols_to_use = [c for c in target_features_reindexed.columns if c not in NON_FEATURE_COLS]
        input_features = target_features_reindexed[cols_to_use].fillna(0).values
        
        # GraphWaveNet expects (Batch, Nodes, Features, TimeSteps)
        # We replicate the current single snapshot to fill the time window (L_steps=4)
        L_steps = 4 
        x_np = np.stack([input_features] * L_steps, axis=2).astype(np.float32)

        data = Data(seq=torch.from_numpy(x_np).float())
        device = next(model.parameters()).device
        data = data.to(device)
        
        with torch.no_grad():
            model.eval()
            predictions_tensor = model(data)

        predictions = predictions_tensor.cpu().numpy().flatten()
        return predictions
        
    else:
        logger.warning(f"Unknown model type '{model_type}'. Returning zero predictions.")
        return np.zeros(len(features_df))