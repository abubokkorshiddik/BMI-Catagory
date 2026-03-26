import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Load data
file_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
mydata = pd.read_excel(file_path, sheet_name="data")

# BMI categories
bmi_categories = ["obesity", "overweight", "underweight"]

# Predictor columns (assuming X1-X22 are predictors)
predictors = [f"X{i}" for i in range(1, 23)]

# Function to calculate VIF and remove perfect collinearity
def calculate_vif(df, predictors):
    # Start with all predictors
    X = df[predictors].copy()
    
    # Iteratively remove columns with inf VIF
    while True:
        X_const = add_constant(X)
        vif_values = pd.Series(
            [variance_inflation_factor(X_const.values, i+1) for i in range(X.shape[1])],
            index=X.columns
        )
        if (vif_values == float('inf')).any():
            # Remove predictor with highest VIF
            drop_col = vif_values.idxmax()
            print(f"Dropping perfectly collinear predictor: {drop_col}")
            X = X.drop(columns=[drop_col])
        else:
            break
    
    vif_df = pd.DataFrame({
        "Variable": X.columns,
        "VIF": vif_values.values,
        "High_VIF": vif_values.values > 10
    })
    return vif_df

# Create Excel writer
output_file = "J:/Research/Research/WorldBMI/ABSDATA/BMI_VIF_PY.xlsx"
writer = pd.ExcelWriter(output_file, engine="openpyxl")

# Calculate VIF for each BMI category
for bmi in bmi_categories:
    # Prepare dataset
    df = mydata[predictors + [bmi]].copy()
    df = df.dropna(subset=[bmi])  # remove rows with missing BMI
    
    # Calculate VIF
    vif_df = calculate_vif(df, predictors)
    
    # Save to Excel sheet
    vif_df.to_excel(writer, sheet_name=bmi.capitalize(), index=False)

writer.close()
print(f"VIF results saved to {output_file}")



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

features = [f'X{i}' for i in range(1,23)]  # X1-X22
target = 'obesity'

# -----------------------
# Impute missing years with country mean
# -----------------------
all_years = list(range(1990, 2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country'] == country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            for f in features:
                if f in df_country.columns:
                    row[f] = df_country[f].mean()
                else:
                    row[f] = 0
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)

df = df.sort_values(['Country','Year']).reset_index(drop=True)

# -----------------------
# Feature standardization and log-transform target
# -----------------------
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
df[target] = np.log1p(df[target])

# -----------------------
# Create PyG data
# -----------------------
def create_graph_data(df_subset, year_range):
    df_year = df_subset[df_subset['Year'].isin(year_range)].copy()
    subset_countries = df_year['Country'].unique()
    df_year = df_year.set_index('Country').loc[subset_countries].reset_index()
    x = torch.tensor(df_year[features].values, dtype=torch.float)
    y = torch.tensor(df_year[target].values, dtype=torch.float).unsqueeze(1)
    N_subset = len(subset_countries)
    adj_matrix_subset = np.ones((N_subset, N_subset)) - np.eye(N_subset)
    edge_index = torch.tensor(np.array(np.nonzero(adj_matrix_subset)), dtype=torch.long)
    return Data(x=x, y=y, edge_index=edge_index)

# -----------------------
# Define STGNN
# -----------------------
class STGNN(nn.Module):
    def __init__(self, in_channels, hidden=64):
        super(STGNN, self).__init__()
        self.gcn1 = nn.Linear(in_channels, hidden)
        self.gcn2 = nn.Linear(hidden, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden,1)
        
    def forward(self, x, edge_index=None):
        x = F.relu(self.gcn1(x))
        x = F.relu(self.gcn2(x))
        x = x.unsqueeze(1)
        x,_ = self.lstm(x)
        x = x.squeeze(1)
        out = self.fc(x)
        return out

# -----------------------
# Evaluation function
# -----------------------
def evaluate(y_true, y_pred, n_features):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2)*(n-1)/(n - n_features -1)
    
    se_rmse = stats.sem(np.abs(y_true - y_pred))
    se_mae = stats.sem(np.abs(y_true - y_pred))
    
    rmse_ci = f"{rmse:.4f} ({rmse - 1.96*se_rmse:.4f}-{rmse + 1.96*se_rmse:.4f})"
    mae_ci = f"{mae:.4f} ({mae - 1.96*se_mae:.4f}-{mae + 1.96*se_mae:.4f})"
    
    return {'RMSE (95% CI)': rmse_ci, 'MAE (95% CI)': mae_ci,
            'R2': round(r2,4), 'Adj_R2': round(adj_r2,4)}

# -----------------------
# Training with 5-fold CV
# -----------------------
def train_evaluate(data):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all, y_pred_all = [], []
    for train_idx, val_idx in kf.split(data.x):
        model = STGNN(in_channels=data.x.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(300):
            model.train()
            optimizer.zero_grad()
            out = model(data.x)
            loss = F.mse_loss(out, data.y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            y_pred_all.append(model(data.x).numpy())
            y_true_all.append(data.y.numpy())
    y_true_all = np.concatenate(y_true_all).flatten()
    y_pred_all = np.concatenate(y_pred_all).flatten()
    return y_true_all, y_pred_all

# -----------------------
# Temporal split
# -----------------------
train_years = list(range(1990,2019))
test_years = list(range(2019,2022))
data_train_temp = create_graph_data(df, train_years)
data_test_temp = create_graph_data(df, test_years)

y_true_train, y_pred_train = train_evaluate(data_train_temp)
metrics_train_temp = evaluate(y_true_train, y_pred_train, n_features=data_train_temp.x.shape[1])

# Retrain final model for test
model = STGNN(in_channels=data_train_temp.x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    out = model(data_train_temp.x)
    loss = F.mse_loss(out, data_train_temp.y)
    loss.backward()
    optimizer.step()
model.eval()
with torch.no_grad():
    y_pred_test = model(data_test_temp.x).numpy().flatten()
    y_true_test = data_test_temp.y.numpy().flatten()
metrics_test_temp = evaluate(y_true_test, y_pred_test, n_features=data_test_temp.x.shape[1])

# -----------------------
# Save metrics to Excel
# -----------------------
with pd.ExcelWriter("J:/Research/Research/WorldBMI/ABSDATA/GeoAI_STGNN_Evaluation.xlsx", engine='openpyxl') as writer:
    df_temp = pd.DataFrame([metrics_train_temp], index=['Train'])
    df_temp_test = pd.DataFrame([metrics_test_temp], index=['Test'])
    df_temp = pd.concat([df_temp, df_temp_test])
    df_temp.to_excel(writer, sheet_name="TemporalSplit", index=True)

print("Done. Excel saved with train and test metrics.")


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

features = [f'X{i}' for i in range(1,23)]
target = 'obesity'

# -----------------------
# Impute missing years with country mean
# -----------------------
all_years = list(range(1990,2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country']==country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            for f in features:
                row[f] = df_country[f].mean() if f in df_country.columns else 0
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)
df = df.sort_values(['Country','Year']).reset_index(drop=True)

# -----------------------
# Standardize features and log-transform target
# -----------------------
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
df[target] = np.log1p(df[target])

# -----------------------
# ConvLSTM model
# -----------------------
class ConvLSTM(nn.Module):
    def __init__(self, n_features, hidden=64, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=hidden, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=1)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden,1)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0,2,1)  # -> (batch, features, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0,2,1)  # -> (batch, seq_len, hidden)
        x,_ = self.lstm(x)
        x = x[:,-1,:]  # last time step
        out = self.fc(x)
        return out

# -----------------------
# Create sequences
# -----------------------
def create_sequences(df_subset, years, seq_len=5):
    df_seq = df_subset[df_subset['Year'].isin(years)].copy()
    df_seq = df_seq.sort_values(['Country','Year']).reset_index(drop=True)
    X, y = [], []
    for country in df_seq['Country'].unique():
        df_c = df_seq[df_seq['Country']==country].sort_values('Year')
        vals = df_c[features].values
        target_vals = df_c[target].values
        for i in range(len(vals)-seq_len+1):
            X.append(vals[i:i+seq_len])
            y.append(target_vals[i+seq_len-1])
    if len(X)==0:
        return torch.empty(0, seq_len, len(features)), torch.empty(0,1)
    return torch.tensor(np.array(X), dtype=torch.float), torch.tensor(np.array(y), dtype=torch.float).unsqueeze(1)

# -----------------------
# Evaluation function
# -----------------------
def evaluate(y_true, y_pred, n_features):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2)*(n-1)/(n - n_features -1)
    
    se_rmse = stats.sem(np.abs(y_true - y_pred))
    se_mae = stats.sem(np.abs(y_true - y_pred))
    
    rmse_ci = f"{rmse:.4f} ({rmse - 1.96*se_rmse:.4f}-{rmse + 1.96*se_rmse:.4f})"
    mae_ci = f"{mae:.4f} ({mae - 1.96*se_mae:.4f}-{mae + 1.96*se_mae:.4f})"
    
    return {'RMSE (95% CI)': rmse_ci, 'MAE (95% CI)': mae_ci,
            'R2': round(r2,4), 'Adj_R2': round(adj_r2,4)}

# -----------------------
# Temporal split
# -----------------------
train_years = list(range(1990,2019))
test_years = list(range(2019,2022))
seq_len_train = 5
seq_len_test = min(seq_len_train, len(test_years))  # safe sequence length for test

X_train, y_train = create_sequences(df, train_years, seq_len=seq_len_train)
X_test, y_test = create_sequences(df, test_years, seq_len=seq_len_test)

if X_test.shape[0] == 0:
    raise ValueError(f"Test sequence length {seq_len_test} is too long for test years {test_years}. Reduce seq_len.")

# -----------------------
# Train model
# -----------------------
model = ConvLSTM(n_features=X_train.shape[2])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 200  # adjust for speed

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(X_train)
    loss = F.mse_loss(out, y_train)
    loss.backward()
    optimizer.step()

# -----------------------
# Predictions & evaluation
# -----------------------
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train).detach().numpy()
    y_true_train = y_train.numpy()
    y_pred_test = model(X_test).detach().numpy()
    y_true_test = y_test.numpy()

metrics_train = evaluate(y_true_train, y_pred_train, n_features=X_train.shape[2])
metrics_test = evaluate(y_true_test, y_pred_test, n_features=X_test.shape[2])

# -----------------------
# Save to Excel
# -----------------------
with pd.ExcelWriter("J:/Research/Research/WorldBMI/ABSDATA/GeoAI_ConvLSTM_Evaluation.xlsx", engine='openpyxl') as writer:
    df_train = pd.DataFrame([metrics_train], index=['Train'])
    df_test = pd.DataFrame([metrics_test], index=['Test'])
    df_all = pd.concat([df_train, df_test])
    df_all.to_excel(writer, sheet_name="TemporalSplit", index=True)

print("Done. Excel saved with train and test metrics for ConvLSTM.")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

features = [f'X{i}' for i in range(1,23)]
target = 'obesity'

# -----------------------
# Impute missing years with country mean
# -----------------------
all_years = list(range(1990,2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country']==country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            for f in features:
                row[f] = df_country[f].mean() if f in df_country.columns else 0
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)
df = df.sort_values(['Country','Year']).reset_index(drop=True)

# -----------------------
# Standardize features and log-transform target
# -----------------------
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
df[target] = np.log1p(df[target])

# -----------------------
# Create PyG graph data
# -----------------------
def create_graph_data(df_subset, years):
    df_year = df_subset[df_subset['Year'].isin(years)].copy()
    countries = df_year['Country'].unique()
    df_year = df_year.set_index('Country').loc[countries].reset_index()
    
    x = torch.tensor(df_year[features].values, dtype=torch.float)
    y = torch.tensor(df_year[target].values, dtype=torch.float).unsqueeze(1)
    
    N = len(countries)
    adj = np.ones((N,N)) - np.eye(N)  # fully connected minus self-loops
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(adj.sum(axis=1) + 1e-10))
    adj_norm = deg_inv_sqrt @ adj @ deg_inv_sqrt
    edge_index = torch.tensor(np.array(np.nonzero(adj_norm)), dtype=torch.long)
    edge_weight = torch.tensor(adj_norm[np.nonzero(adj_norm)], dtype=torch.float)
    
    return Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weight)

# -----------------------
# Define GWNN layer with residual projection
# -----------------------
class GWNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.res_proj = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index, edge_weight):
        # Project x first
        x_proj = self.lin(x)  # [num_nodes, out_channels]
        row, col = edge_index
        agg = torch.zeros_like(x_proj)  # [num_nodes, out_channels]
        # Aggregate projected neighbors
        agg.index_add_(0, row, x_proj[col] * edge_weight.unsqueeze(1))
        out = agg + self.res_proj(x)  # residual connection
        return F.relu(out)
# -----------------------
# GWNN model
# -----------------------
class GWNN(nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.gwnn1 = GWNNLayer(in_channels, hidden)
        self.gwnn2 = GWNNLayer(hidden, hidden)
        self.fc = nn.Linear(hidden,1)
        
    def forward(self, x, edge_index, edge_weight):
        x = self.gwnn1(x, edge_index, edge_weight)
        x = self.gwnn2(x, edge_index, edge_weight)
        out = self.fc(x)
        return out

# -----------------------
# Evaluation function
# -----------------------
def evaluate(y_true, y_pred, n_features):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2)*(n-1)/(n - n_features -1)
    
    se_rmse = stats.sem(np.abs(y_true - y_pred))
    se_mae = stats.sem(np.abs(y_true - y_pred))
    
    rmse_ci = f"{rmse:.4f} ({rmse - 1.96*se_rmse:.4f}-{rmse + 1.96*se_rmse:.4f})"
    mae_ci = f"{mae:.4f} ({mae - 1.96*se_mae:.4f}-{mae + 1.96*se_mae:.4f})"
    
    return {'RMSE (95% CI)': rmse_ci, 'MAE (95% CI)': mae_ci,
            'R2': round(r2,4), 'Adj_R2': round(adj_r2,4)}

# -----------------------
# Temporal split
# -----------------------
train_years = list(range(1990,2019))
test_years = list(range(2019,2022))

data_train = create_graph_data(df, train_years)
data_test = create_graph_data(df, test_years)

# -----------------------
# Train GWNN
# -----------------------
model = GWNN(in_channels=data_train.x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 300

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data_train.x, data_train.edge_index, data_train.edge_weight)
    loss = F.mse_loss(out, data_train.y)
    loss.backward()
    optimizer.step()

# -----------------------
# Predictions & evaluation
# -----------------------
model.eval()
with torch.no_grad():
    y_pred_train = model(data_train.x, data_train.edge_index, data_train.edge_weight).numpy()
    y_true_train = data_train.y.numpy()
    y_pred_test = model(data_test.x, data_test.edge_index, data_test.edge_weight).numpy()
    y_true_test = data_test.y.numpy()

metrics_train = evaluate(y_true_train, y_pred_train, n_features=data_train.x.shape[1])
metrics_test = evaluate(y_true_test, y_pred_test, n_features=data_test.x.shape[1])

# -----------------------
# Save to Excel
# -----------------------
with pd.ExcelWriter("J:/Research/Research/WorldBMI/ABSDATA/GeoAI_GWNN_Evaluation.xlsx", engine='openpyxl') as writer:
    df_train = pd.DataFrame([metrics_train], index=['Train'])
    df_test = pd.DataFrame([metrics_test], index=['Test'])
    df_all = pd.concat([df_train, df_test])
    df_all.to_excel(writer, sheet_name="TemporalSplit", index=True)

print("Done. Excel saved with GWNN metrics (residual projection).")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

features = [f'X{i}' for i in range(1,23)]
target = 'obesity'

# -----------------------
# Impute missing years with country mean
# -----------------------
all_years = list(range(1990,2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country']==country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            for f in features:
                row[f] = df_country[f].mean() if f in df_country.columns else 0
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)
df = df.sort_values(['Country','Year']).reset_index(drop=True)

# -----------------------
# Standardize features and log-transform target
# -----------------------
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
df[target] = np.log1p(df[target])

# -----------------------
# Create PyG graph data
# -----------------------
def create_graph_data(df_subset, years):
    df_year = df_subset[df_subset['Year'].isin(years)].copy()
    countries = df_year['Country'].unique()
    df_year = df_year.set_index('Country').loc[countries].reset_index()
    
    x = torch.tensor(df_year[features].values, dtype=torch.float)
    y = torch.tensor(df_year[target].values, dtype=torch.float).unsqueeze(1)
    
    N = len(countries)
    adj = np.ones((N,N)) - np.eye(N)  # fully connected minus self-loops
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(adj.sum(axis=1) + 1e-10))
    adj_norm = deg_inv_sqrt @ adj @ deg_inv_sqrt
    edge_index = torch.tensor(np.array(np.nonzero(adj_norm)), dtype=torch.long)
    edge_weight = torch.tensor(adj_norm[np.nonzero(adj_norm)], dtype=torch.float)
    
    return Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weight)

# -----------------------
# Define GAT model
# -----------------------
class GAT(nn.Module):
    def __init__(self, in_channels, hidden=64, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden, heads=heads, concat=True)
        self.gat2 = GATConv(hidden*heads, hidden, heads=1, concat=False)
        self.fc = nn.Linear(hidden,1)
        
    def forward(self, x, edge_index, edge_weight=None):
        x = F.elu(self.gat1(x, edge_index, edge_weight))
        x = F.elu(self.gat2(x, edge_index, edge_weight))
        out = self.fc(x)
        return out

# -----------------------
# Evaluation function
# -----------------------
def evaluate(y_true, y_pred, n_features):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2)*(n-1)/(n - n_features -1)
    
    se_rmse = stats.sem(np.abs(y_true - y_pred))
    se_mae = stats.sem(np.abs(y_true - y_pred))
    
    rmse_ci = f"{rmse:.4f} ({rmse - 1.96*se_rmse:.4f}-{rmse + 1.96*se_rmse:.4f})"
    mae_ci = f"{mae:.4f} ({mae - 1.96*se_mae:.4f}-{mae + 1.96*se_mae:.4f})"
    
    return {'RMSE (95% CI)': rmse_ci, 'MAE (95% CI)': mae_ci,
            'R2': round(r2,4), 'Adj_R2': round(adj_r2,4)}

# -----------------------
# Temporal split
# -----------------------
train_years = list(range(1990,2019))
test_years = list(range(2019,2022))

data_train = create_graph_data(df, train_years)
data_test = create_graph_data(df, test_years)

# -----------------------
# Train GAT
# -----------------------
model = GAT(in_channels=data_train.x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 300

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data_train.x, data_train.edge_index, data_train.edge_weight)
    loss = F.mse_loss(out, data_train.y)
    loss.backward()
    optimizer.step()

# -----------------------
# Predictions & evaluation
# -----------------------
model.eval()
with torch.no_grad():
    y_pred_train = model(data_train.x, data_train.edge_index, data_train.edge_weight).numpy()
    y_true_train = data_train.y.numpy()
    y_pred_test = model(data_test.x, data_test.edge_index, data_test.edge_weight).numpy()
    y_true_test = data_test.y.numpy()

metrics_train = evaluate(y_true_train, y_pred_train, n_features=data_train.x.shape[1])
metrics_test = evaluate(y_true_test, y_pred_test, n_features=data_test.x.shape[1])

# -----------------------
# Save to Excel
# -----------------------
with pd.ExcelWriter("J:/Research/Research/WorldBMI/ABSDATA/GeoAI_GAT_Evaluation.xlsx", engine='openpyxl') as writer:
    df_train = pd.DataFrame([metrics_train], index=['Train'])
    df_test = pd.DataFrame([metrics_test], index=['Test'])
    df_all = pd.concat([df_train, df_test])
    df_all.to_excel(writer, sheet_name="TemporalSplit", index=True)

print("Done. Excel saved with GAT metrics.")



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

features = [f'X{i}' for i in range(1,23)]  # X1-X22
target = 'overweight'

# -----------------------
# Impute missing years with country mean
# -----------------------
all_years = list(range(1990, 2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country'] == country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            for f in features:
                if f in df_country.columns:
                    row[f] = df_country[f].mean()
                else:
                    row[f] = 0
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)

df = df.sort_values(['Country','Year']).reset_index(drop=True)

# -----------------------
# Feature standardization and log-transform target
# -----------------------
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
df[target] = np.log1p(df[target])

# -----------------------
# Create PyG data
# -----------------------
def create_graph_data(df_subset, year_range):
    df_year = df_subset[df_subset['Year'].isin(year_range)].copy()
    subset_countries = df_year['Country'].unique()
    df_year = df_year.set_index('Country').loc[subset_countries].reset_index()
    x = torch.tensor(df_year[features].values, dtype=torch.float)
    y = torch.tensor(df_year[target].values, dtype=torch.float).unsqueeze(1)
    N_subset = len(subset_countries)
    adj_matrix_subset = np.ones((N_subset, N_subset)) - np.eye(N_subset)
    edge_index = torch.tensor(np.array(np.nonzero(adj_matrix_subset)), dtype=torch.long)
    return Data(x=x, y=y, edge_index=edge_index)

# -----------------------
# Define STGNN
# -----------------------
class STGNN(nn.Module):
    def __init__(self, in_channels, hidden=64):
        super(STGNN, self).__init__()
        self.gcn1 = nn.Linear(in_channels, hidden)
        self.gcn2 = nn.Linear(hidden, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden,1)
        
    def forward(self, x, edge_index=None):
        x = F.relu(self.gcn1(x))
        x = F.relu(self.gcn2(x))
        x = x.unsqueeze(1)
        x,_ = self.lstm(x)
        x = x.squeeze(1)
        out = self.fc(x)
        return out

# -----------------------
# Evaluation function
# -----------------------
def evaluate(y_true, y_pred, n_features):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2)*(n-1)/(n - n_features -1)
    
    se_rmse = stats.sem(np.abs(y_true - y_pred))
    se_mae = stats.sem(np.abs(y_true - y_pred))
    
    rmse_ci = f"{rmse:.4f} ({rmse - 1.96*se_rmse:.4f}-{rmse + 1.96*se_rmse:.4f})"
    mae_ci = f"{mae:.4f} ({mae - 1.96*se_mae:.4f}-{mae + 1.96*se_mae:.4f})"
    
    return {'RMSE (95% CI)': rmse_ci, 'MAE (95% CI)': mae_ci,
            'R2': round(r2,4), 'Adj_R2': round(adj_r2,4)}

# -----------------------
# Training with 5-fold CV
# -----------------------
def train_evaluate(data):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all, y_pred_all = [], []
    for train_idx, val_idx in kf.split(data.x):
        model = STGNN(in_channels=data.x.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(400):
            model.train()
            optimizer.zero_grad()
            out = model(data.x)
            loss = F.mse_loss(out, data.y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            y_pred_all.append(model(data.x).numpy())
            y_true_all.append(data.y.numpy())
    y_true_all = np.concatenate(y_true_all).flatten()
    y_pred_all = np.concatenate(y_pred_all).flatten()
    return y_true_all, y_pred_all

# -----------------------
# Temporal split
# -----------------------
train_years = list(range(1990,2019))
test_years = list(range(2019,2022))
data_train_temp = create_graph_data(df, train_years)
data_test_temp = create_graph_data(df, test_years)

y_true_train, y_pred_train = train_evaluate(data_train_temp)
metrics_train_temp = evaluate(y_true_train, y_pred_train, n_features=data_train_temp.x.shape[1])

# Retrain final model for test
model = STGNN(in_channels=data_train_temp.x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(450):
    model.train()
    optimizer.zero_grad()
    out = model(data_train_temp.x)
    loss = F.mse_loss(out, data_train_temp.y)
    loss.backward()
    optimizer.step()
model.eval()
with torch.no_grad():
    y_pred_test = model(data_test_temp.x).numpy().flatten()
    y_true_test = data_test_temp.y.numpy().flatten()
metrics_test_temp = evaluate(y_true_test, y_pred_test, n_features=data_test_temp.x.shape[1])

# -----------------------
# Save metrics to Excel
# -----------------------
with pd.ExcelWriter("J:/Research/Research/WorldBMI/ABSDATA/overweight_GeoAI_STGNN_Evaluation.xlsx", engine='openpyxl') as writer:
    df_temp = pd.DataFrame([metrics_train_temp], index=['Train'])
    df_temp_test = pd.DataFrame([metrics_test_temp], index=['Test'])
    df_temp = pd.concat([df_temp, df_temp_test])
    df_temp.to_excel(writer, sheet_name="TemporalSplit", index=True)

print("Done. Excel saved with train and test metrics.")



import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

features = [f'X{i}' for i in range(1,23)]
target = 'overweight'

# -----------------------
# Impute missing years with country mean
# -----------------------
all_years = list(range(1990,2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country']==country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            for f in features:
                row[f] = df_country[f].mean() if f in df_country.columns else 0
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)
df = df.sort_values(['Country','Year']).reset_index(drop=True)

# -----------------------
# Standardize features and log-transform target
# -----------------------
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
df[target] = np.log1p(df[target])

# -----------------------
# ConvLSTM model
# -----------------------
class ConvLSTM(nn.Module):
    def __init__(self, n_features, hidden=64, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=hidden, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=1)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden,1)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0,2,1)  # -> (batch, features, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0,2,1)  # -> (batch, seq_len, hidden)
        x,_ = self.lstm(x)
        x = x[:,-1,:]  # last time step
        out = self.fc(x)
        return out

# -----------------------
# Create sequences
# -----------------------
def create_sequences(df_subset, years, seq_len=5):
    df_seq = df_subset[df_subset['Year'].isin(years)].copy()
    df_seq = df_seq.sort_values(['Country','Year']).reset_index(drop=True)
    X, y = [], []
    for country in df_seq['Country'].unique():
        df_c = df_seq[df_seq['Country']==country].sort_values('Year')
        vals = df_c[features].values
        target_vals = df_c[target].values
        for i in range(len(vals)-seq_len+1):
            X.append(vals[i:i+seq_len])
            y.append(target_vals[i+seq_len-1])
    if len(X)==0:
        return torch.empty(0, seq_len, len(features)), torch.empty(0,1)
    return torch.tensor(np.array(X), dtype=torch.float), torch.tensor(np.array(y), dtype=torch.float).unsqueeze(1)

# -----------------------
# Evaluation function
# -----------------------
def evaluate(y_true, y_pred, n_features):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2)*(n-1)/(n - n_features -1)
    
    se_rmse = stats.sem(np.abs(y_true - y_pred))
    se_mae = stats.sem(np.abs(y_true - y_pred))
    
    rmse_ci = f"{rmse:.4f} ({rmse - 1.96*se_rmse:.4f}-{rmse + 1.96*se_rmse:.4f})"
    mae_ci = f"{mae:.4f} ({mae - 1.96*se_mae:.4f}-{mae + 1.96*se_mae:.4f})"
    
    return {'RMSE (95% CI)': rmse_ci, 'MAE (95% CI)': mae_ci,
            'R2': round(r2,4), 'Adj_R2': round(adj_r2,4)}

# -----------------------
# Temporal split
# -----------------------
train_years = list(range(1990,2019))
test_years = list(range(2019,2022))
seq_len_train = 5
seq_len_test = min(seq_len_train, len(test_years))  # safe sequence length for test

X_train, y_train = create_sequences(df, train_years, seq_len=seq_len_train)
X_test, y_test = create_sequences(df, test_years, seq_len=seq_len_test)

if X_test.shape[0] == 0:
    raise ValueError(f"Test sequence length {seq_len_test} is too long for test years {test_years}. Reduce seq_len.")

# -----------------------
# Train model
# -----------------------
model = ConvLSTM(n_features=X_train.shape[2])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 200  # adjust for speed

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(X_train)
    loss = F.mse_loss(out, y_train)
    loss.backward()
    optimizer.step()

# -----------------------
# Predictions & evaluation
# -----------------------
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train).detach().numpy()
    y_true_train = y_train.numpy()
    y_pred_test = model(X_test).detach().numpy()
    y_true_test = y_test.numpy()

metrics_train = evaluate(y_true_train, y_pred_train, n_features=X_train.shape[2])
metrics_test = evaluate(y_true_test, y_pred_test, n_features=X_test.shape[2])

# -----------------------
# Save to Excel
# -----------------------
with pd.ExcelWriter("J:/Research/Research/WorldBMI/ABSDATA/overweight_GeoAI_ConvLSTM_Evaluation.xlsx", engine='openpyxl') as writer:
    df_train = pd.DataFrame([metrics_train], index=['Train'])
    df_test = pd.DataFrame([metrics_test], index=['Test'])
    df_all = pd.concat([df_train, df_test])
    df_all.to_excel(writer, sheet_name="TemporalSplit", index=True)

print("Done. Excel saved with train and test metrics for ConvLSTM.")



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

features = [f'X{i}' for i in range(1,23)]
target = 'overweight'

# -----------------------
# Impute missing years with country mean
# -----------------------
all_years = list(range(1990,2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country']==country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            for f in features:
                row[f] = df_country[f].mean() if f in df_country.columns else 0
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)
df = df.sort_values(['Country','Year']).reset_index(drop=True)

# -----------------------
# Standardize features and log-transform target
# -----------------------
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
df[target] = np.log1p(df[target])

# -----------------------
# Create PyG graph data
# -----------------------
def create_graph_data(df_subset, years):
    df_year = df_subset[df_subset['Year'].isin(years)].copy()
    countries = df_year['Country'].unique()
    df_year = df_year.set_index('Country').loc[countries].reset_index()
    
    x = torch.tensor(df_year[features].values, dtype=torch.float)
    y = torch.tensor(df_year[target].values, dtype=torch.float).unsqueeze(1)
    
    N = len(countries)
    adj = np.ones((N,N)) - np.eye(N)  # fully connected minus self-loops
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(adj.sum(axis=1) + 1e-10))
    adj_norm = deg_inv_sqrt @ adj @ deg_inv_sqrt
    edge_index = torch.tensor(np.array(np.nonzero(adj_norm)), dtype=torch.long)
    edge_weight = torch.tensor(adj_norm[np.nonzero(adj_norm)], dtype=torch.float)
    
    return Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weight)

# -----------------------
# Define GWNN layer with residual projection
# -----------------------
class GWNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.res_proj = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index, edge_weight):
        # Project x first
        x_proj = self.lin(x)  # [num_nodes, out_channels]
        row, col = edge_index
        agg = torch.zeros_like(x_proj)  # [num_nodes, out_channels]
        # Aggregate projected neighbors
        agg.index_add_(0, row, x_proj[col] * edge_weight.unsqueeze(1))
        out = agg + self.res_proj(x)  # residual connection
        return F.relu(out)
# -----------------------
# GWNN model
# -----------------------
class GWNN(nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.gwnn1 = GWNNLayer(in_channels, hidden)
        self.gwnn2 = GWNNLayer(hidden, hidden)
        self.fc = nn.Linear(hidden,1)
        
    def forward(self, x, edge_index, edge_weight):
        x = self.gwnn1(x, edge_index, edge_weight)
        x = self.gwnn2(x, edge_index, edge_weight)
        out = self.fc(x)
        return out

# -----------------------
# Evaluation function
# -----------------------
def evaluate(y_true, y_pred, n_features):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2)*(n-1)/(n - n_features -1)
    
    se_rmse = stats.sem(np.abs(y_true - y_pred))
    se_mae = stats.sem(np.abs(y_true - y_pred))
    
    rmse_ci = f"{rmse:.4f} ({rmse - 1.96*se_rmse:.4f}-{rmse + 1.96*se_rmse:.4f})"
    mae_ci = f"{mae:.4f} ({mae - 1.96*se_mae:.4f}-{mae + 1.96*se_mae:.4f})"
    
    return {'RMSE (95% CI)': rmse_ci, 'MAE (95% CI)': mae_ci,
            'R2': round(r2,4), 'Adj_R2': round(adj_r2,4)}

# -----------------------
# Temporal split
# -----------------------
train_years = list(range(1990,2019))
test_years = list(range(2019,2022))

data_train = create_graph_data(df, train_years)
data_test = create_graph_data(df, test_years)

# -----------------------
# Train GWNN
# -----------------------
model = GWNN(in_channels=data_train.x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 300

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data_train.x, data_train.edge_index, data_train.edge_weight)
    loss = F.mse_loss(out, data_train.y)
    loss.backward()
    optimizer.step()

# -----------------------
# Predictions & evaluation
# -----------------------
model.eval()
with torch.no_grad():
    y_pred_train = model(data_train.x, data_train.edge_index, data_train.edge_weight).numpy()
    y_true_train = data_train.y.numpy()
    y_pred_test = model(data_test.x, data_test.edge_index, data_test.edge_weight).numpy()
    y_true_test = data_test.y.numpy()

metrics_train = evaluate(y_true_train, y_pred_train, n_features=data_train.x.shape[1])
metrics_test = evaluate(y_true_test, y_pred_test, n_features=data_test.x.shape[1])

# -----------------------
# Save to Excel
# -----------------------
with pd.ExcelWriter("J:/Research/Research/WorldBMI/ABSDATA/overweight_GeoAI_GWNN_Evaluation.xlsx", engine='openpyxl') as writer:
    df_train = pd.DataFrame([metrics_train], index=['Train'])
    df_test = pd.DataFrame([metrics_test], index=['Test'])
    df_all = pd.concat([df_train, df_test])
    df_all.to_excel(writer, sheet_name="TemporalSplit", index=True)

print("Done. Excel saved with GWNN metrics (residual projection).")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

features = [f'X{i}' for i in range(1,23)]
target = 'overweight'

# -----------------------
# Impute missing years with country mean
# -----------------------
all_years = list(range(1990,2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country']==country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            for f in features:
                row[f] = df_country[f].mean() if f in df_country.columns else 0
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)
df = df.sort_values(['Country','Year']).reset_index(drop=True)

# -----------------------
# Standardize features and log-transform target
# -----------------------
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
df[target] = np.log1p(df[target])

# -----------------------
# Create PyG graph data
# -----------------------
def create_graph_data(df_subset, years):
    df_year = df_subset[df_subset['Year'].isin(years)].copy()
    countries = df_year['Country'].unique()
    df_year = df_year.set_index('Country').loc[countries].reset_index()
    
    x = torch.tensor(df_year[features].values, dtype=torch.float)
    y = torch.tensor(df_year[target].values, dtype=torch.float).unsqueeze(1)
    
    N = len(countries)
    adj = np.ones((N,N)) - np.eye(N)  # fully connected minus self-loops
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(adj.sum(axis=1) + 1e-10))
    adj_norm = deg_inv_sqrt @ adj @ deg_inv_sqrt
    edge_index = torch.tensor(np.array(np.nonzero(adj_norm)), dtype=torch.long)
    edge_weight = torch.tensor(adj_norm[np.nonzero(adj_norm)], dtype=torch.float)
    
    return Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weight)

# -----------------------
# Define GAT model
# -----------------------
class GAT(nn.Module):
    def __init__(self, in_channels, hidden=64, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden, heads=heads, concat=True)
        self.gat2 = GATConv(hidden*heads, hidden, heads=1, concat=False)
        self.fc = nn.Linear(hidden,1)
        
    def forward(self, x, edge_index, edge_weight=None):
        x = F.elu(self.gat1(x, edge_index, edge_weight))
        x = F.elu(self.gat2(x, edge_index, edge_weight))
        out = self.fc(x)
        return out

# -----------------------
# Evaluation function
# -----------------------
def evaluate(y_true, y_pred, n_features):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2)*(n-1)/(n - n_features -1)
    
    se_rmse = stats.sem(np.abs(y_true - y_pred))
    se_mae = stats.sem(np.abs(y_true - y_pred))
    
    rmse_ci = f"{rmse:.4f} ({rmse - 1.96*se_rmse:.4f}-{rmse + 1.96*se_rmse:.4f})"
    mae_ci = f"{mae:.4f} ({mae - 1.96*se_mae:.4f}-{mae + 1.96*se_mae:.4f})"
    
    return {'RMSE (95% CI)': rmse_ci, 'MAE (95% CI)': mae_ci,
            'R2': round(r2,4), 'Adj_R2': round(adj_r2,4)}

# -----------------------
# Temporal split
# -----------------------
train_years = list(range(1990,2019))
test_years = list(range(2019,2022))

data_train = create_graph_data(df, train_years)
data_test = create_graph_data(df, test_years)

# -----------------------
# Train GAT
# -----------------------
model = GAT(in_channels=data_train.x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 300

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data_train.x, data_train.edge_index, data_train.edge_weight)
    loss = F.mse_loss(out, data_train.y)
    loss.backward()
    optimizer.step()

# -----------------------
# Predictions & evaluation
# -----------------------
model.eval()
with torch.no_grad():
    y_pred_train = model(data_train.x, data_train.edge_index, data_train.edge_weight).numpy()
    y_true_train = data_train.y.numpy()
    y_pred_test = model(data_test.x, data_test.edge_index, data_test.edge_weight).numpy()
    y_true_test = data_test.y.numpy()

metrics_train = evaluate(y_true_train, y_pred_train, n_features=data_train.x.shape[1])
metrics_test = evaluate(y_true_test, y_pred_test, n_features=data_test.x.shape[1])

# -----------------------
# Save to Excel
# -----------------------
with pd.ExcelWriter("J:/Research/Research/WorldBMI/ABSDATA/overweight_GeoAI_GAT_Evaluation.xlsx", engine='openpyxl') as writer:
    df_train = pd.DataFrame([metrics_train], index=['Train'])
    df_test = pd.DataFrame([metrics_test], index=['Test'])
    df_all = pd.concat([df_train, df_test])
    df_all.to_excel(writer, sheet_name="TemporalSplit", index=True)

print("Done. Excel saved with GAT metrics.")



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

features = [f'X{i}' for i in range(1,23)]  # X1-X22
target = 'underweight'

# -----------------------
# Impute missing years with country mean
# -----------------------
all_years = list(range(1990, 2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country'] == country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            for f in features:
                if f in df_country.columns:
                    row[f] = df_country[f].mean()
                else:
                    row[f] = 0
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)

df = df.sort_values(['Country','Year']).reset_index(drop=True)

# -----------------------
# Feature standardization and log-transform target
# -----------------------
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
df[target] = np.log1p(df[target])

# -----------------------
# Create PyG data
# -----------------------
def create_graph_data(df_subset, year_range):
    df_year = df_subset[df_subset['Year'].isin(year_range)].copy()
    subset_countries = df_year['Country'].unique()
    df_year = df_year.set_index('Country').loc[subset_countries].reset_index()
    x = torch.tensor(df_year[features].values, dtype=torch.float)
    y = torch.tensor(df_year[target].values, dtype=torch.float).unsqueeze(1)
    N_subset = len(subset_countries)
    adj_matrix_subset = np.ones((N_subset, N_subset)) - np.eye(N_subset)
    edge_index = torch.tensor(np.array(np.nonzero(adj_matrix_subset)), dtype=torch.long)
    return Data(x=x, y=y, edge_index=edge_index)

# -----------------------
# Define STGNN
# -----------------------
class STGNN(nn.Module):
    def __init__(self, in_channels, hidden=64):
        super(STGNN, self).__init__()
        self.gcn1 = nn.Linear(in_channels, hidden)
        self.gcn2 = nn.Linear(hidden, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden,1)
        
    def forward(self, x, edge_index=None):
        x = F.relu(self.gcn1(x))
        x = F.relu(self.gcn2(x))
        x = x.unsqueeze(1)
        x,_ = self.lstm(x)
        x = x.squeeze(1)
        out = self.fc(x)
        return out

# -----------------------
# Evaluation function
# -----------------------
def evaluate(y_true, y_pred, n_features):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2)*(n-1)/(n - n_features -1)
    
    se_rmse = stats.sem(np.abs(y_true - y_pred))
    se_mae = stats.sem(np.abs(y_true - y_pred))
    
    rmse_ci = f"{rmse:.4f} ({rmse - 1.96*se_rmse:.4f}-{rmse + 1.96*se_rmse:.4f})"
    mae_ci = f"{mae:.4f} ({mae - 1.96*se_mae:.4f}-{mae + 1.96*se_mae:.4f})"
    
    return {'RMSE (95% CI)': rmse_ci, 'MAE (95% CI)': mae_ci,
            'R2': round(r2,4), 'Adj_R2': round(adj_r2,4)}

# -----------------------
# Training with 5-fold CV
# -----------------------
def train_evaluate(data):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all, y_pred_all = [], []
    for train_idx, val_idx in kf.split(data.x):
        model = STGNN(in_channels=data.x.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(300):
            model.train()
            optimizer.zero_grad()
            out = model(data.x)
            loss = F.mse_loss(out, data.y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            y_pred_all.append(model(data.x).numpy())
            y_true_all.append(data.y.numpy())
    y_true_all = np.concatenate(y_true_all).flatten()
    y_pred_all = np.concatenate(y_pred_all).flatten()
    return y_true_all, y_pred_all

# -----------------------
# Temporal split
# -----------------------
train_years = list(range(1990,2019))
test_years = list(range(2019,2022))
data_train_temp = create_graph_data(df, train_years)
data_test_temp = create_graph_data(df, test_years)

y_true_train, y_pred_train = train_evaluate(data_train_temp)
metrics_train_temp = evaluate(y_true_train, y_pred_train, n_features=data_train_temp.x.shape[1])

# Retrain final model for test
model = STGNN(in_channels=data_train_temp.x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    out = model(data_train_temp.x)
    loss = F.mse_loss(out, data_train_temp.y)
    loss.backward()
    optimizer.step()
model.eval()
with torch.no_grad():
    y_pred_test = model(data_test_temp.x).numpy().flatten()
    y_true_test = data_test_temp.y.numpy().flatten()
metrics_test_temp = evaluate(y_true_test, y_pred_test, n_features=data_test_temp.x.shape[1])

# -----------------------
# Save metrics to Excel
# -----------------------
with pd.ExcelWriter("J:/Research/Research/WorldBMI/ABSDATA/underweight_GeoAI_STGNN_Evaluation.xlsx", engine='openpyxl') as writer:
    df_temp = pd.DataFrame([metrics_train_temp], index=['Train'])
    df_temp_test = pd.DataFrame([metrics_test_temp], index=['Test'])
    df_temp = pd.concat([df_temp, df_temp_test])
    df_temp.to_excel(writer, sheet_name="TemporalSplit", index=True)

print("Done. Excel saved with train and test metrics.")


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

features = [f'X{i}' for i in range(1,23)]
target = 'underweight'

# -----------------------
# Impute missing years with country mean
# -----------------------
all_years = list(range(1990,2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country']==country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            for f in features:
                row[f] = df_country[f].mean() if f in df_country.columns else 0
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)
df = df.sort_values(['Country','Year']).reset_index(drop=True)

# -----------------------
# Standardize features and log-transform target
# -----------------------
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
df[target] = np.log1p(df[target])

# -----------------------
# ConvLSTM model
# -----------------------
class ConvLSTM(nn.Module):
    def __init__(self, n_features, hidden=64, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=hidden, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=1)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden,1)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0,2,1)  # -> (batch, features, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0,2,1)  # -> (batch, seq_len, hidden)
        x,_ = self.lstm(x)
        x = x[:,-1,:]  # last time step
        out = self.fc(x)
        return out

# -----------------------
# Create sequences
# -----------------------
def create_sequences(df_subset, years, seq_len=5):
    df_seq = df_subset[df_subset['Year'].isin(years)].copy()
    df_seq = df_seq.sort_values(['Country','Year']).reset_index(drop=True)
    X, y = [], []
    for country in df_seq['Country'].unique():
        df_c = df_seq[df_seq['Country']==country].sort_values('Year')
        vals = df_c[features].values
        target_vals = df_c[target].values
        for i in range(len(vals)-seq_len+1):
            X.append(vals[i:i+seq_len])
            y.append(target_vals[i+seq_len-1])
    if len(X)==0:
        return torch.empty(0, seq_len, len(features)), torch.empty(0,1)
    return torch.tensor(np.array(X), dtype=torch.float), torch.tensor(np.array(y), dtype=torch.float).unsqueeze(1)

# -----------------------
# Evaluation function
# -----------------------
def evaluate(y_true, y_pred, n_features):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2)*(n-1)/(n - n_features -1)
    
    se_rmse = stats.sem(np.abs(y_true - y_pred))
    se_mae = stats.sem(np.abs(y_true - y_pred))
    
    rmse_ci = f"{rmse:.4f} ({rmse - 1.96*se_rmse:.4f}-{rmse + 1.96*se_rmse:.4f})"
    mae_ci = f"{mae:.4f} ({mae - 1.96*se_mae:.4f}-{mae + 1.96*se_mae:.4f})"
    
    return {'RMSE (95% CI)': rmse_ci, 'MAE (95% CI)': mae_ci,
            'R2': round(r2,4), 'Adj_R2': round(adj_r2,4)}

# -----------------------
# Temporal split
# -----------------------
train_years = list(range(1990,2019))
test_years = list(range(2019,2022))
seq_len_train = 5
seq_len_test = min(seq_len_train, len(test_years))  # safe sequence length for test

X_train, y_train = create_sequences(df, train_years, seq_len=seq_len_train)
X_test, y_test = create_sequences(df, test_years, seq_len=seq_len_test)

if X_test.shape[0] == 0:
    raise ValueError(f"Test sequence length {seq_len_test} is too long for test years {test_years}. Reduce seq_len.")

# -----------------------
# Train model
# -----------------------
model = ConvLSTM(n_features=X_train.shape[2])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 200  # adjust for speed

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(X_train)
    loss = F.mse_loss(out, y_train)
    loss.backward()
    optimizer.step()

# -----------------------
# Predictions & evaluation
# -----------------------
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train).detach().numpy()
    y_true_train = y_train.numpy()
    y_pred_test = model(X_test).detach().numpy()
    y_true_test = y_test.numpy()

metrics_train = evaluate(y_true_train, y_pred_train, n_features=X_train.shape[2])
metrics_test = evaluate(y_true_test, y_pred_test, n_features=X_test.shape[2])

# -----------------------
# Save to Excel
# -----------------------
with pd.ExcelWriter("J:/Research/Research/WorldBMI/ABSDATA/underweight_GeoAI_ConvLSTM_Evaluation.xlsx", engine='openpyxl') as writer:
    df_train = pd.DataFrame([metrics_train], index=['Train'])
    df_test = pd.DataFrame([metrics_test], index=['Test'])
    df_all = pd.concat([df_train, df_test])
    df_all.to_excel(writer, sheet_name="TemporalSplit", index=True)

print("Done. Excel saved with train and test metrics for ConvLSTM.")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

features = [f'X{i}' for i in range(1,23)]
target = 'underweight'

# -----------------------
# Impute missing years with country mean
# -----------------------
all_years = list(range(1990,2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country']==country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            for f in features:
                row[f] = df_country[f].mean() if f in df_country.columns else 0
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)
df = df.sort_values(['Country','Year']).reset_index(drop=True)

# -----------------------
# Standardize features and log-transform target
# -----------------------
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
df[target] = np.log1p(df[target])

# -----------------------
# Create PyG graph data
# -----------------------
def create_graph_data(df_subset, years):
    df_year = df_subset[df_subset['Year'].isin(years)].copy()
    countries = df_year['Country'].unique()
    df_year = df_year.set_index('Country').loc[countries].reset_index()
    
    x = torch.tensor(df_year[features].values, dtype=torch.float)
    y = torch.tensor(df_year[target].values, dtype=torch.float).unsqueeze(1)
    
    N = len(countries)
    adj = np.ones((N,N)) - np.eye(N)  # fully connected minus self-loops
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(adj.sum(axis=1) + 1e-10))
    adj_norm = deg_inv_sqrt @ adj @ deg_inv_sqrt
    edge_index = torch.tensor(np.array(np.nonzero(adj_norm)), dtype=torch.long)
    edge_weight = torch.tensor(adj_norm[np.nonzero(adj_norm)], dtype=torch.float)
    
    return Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weight)

# -----------------------
# Define GWNN layer with residual projection
# -----------------------
class GWNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.res_proj = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index, edge_weight):
        # Project x first
        x_proj = self.lin(x)  # [num_nodes, out_channels]
        row, col = edge_index
        agg = torch.zeros_like(x_proj)  # [num_nodes, out_channels]
        # Aggregate projected neighbors
        agg.index_add_(0, row, x_proj[col] * edge_weight.unsqueeze(1))
        out = agg + self.res_proj(x)  # residual connection
        return F.relu(out)
# -----------------------
# GWNN model
# -----------------------
class GWNN(nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.gwnn1 = GWNNLayer(in_channels, hidden)
        self.gwnn2 = GWNNLayer(hidden, hidden)
        self.fc = nn.Linear(hidden,1)
        
    def forward(self, x, edge_index, edge_weight):
        x = self.gwnn1(x, edge_index, edge_weight)
        x = self.gwnn2(x, edge_index, edge_weight)
        out = self.fc(x)
        return out

# -----------------------
# Evaluation function
# -----------------------
def evaluate(y_true, y_pred, n_features):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2)*(n-1)/(n - n_features -1)
    
    se_rmse = stats.sem(np.abs(y_true - y_pred))
    se_mae = stats.sem(np.abs(y_true - y_pred))
    
    rmse_ci = f"{rmse:.4f} ({rmse - 1.96*se_rmse:.4f}-{rmse + 1.96*se_rmse:.4f})"
    mae_ci = f"{mae:.4f} ({mae - 1.96*se_mae:.4f}-{mae + 1.96*se_mae:.4f})"
    
    return {'RMSE (95% CI)': rmse_ci, 'MAE (95% CI)': mae_ci,
            'R2': round(r2,4), 'Adj_R2': round(adj_r2,4)}

# -----------------------
# Temporal split
# -----------------------
train_years = list(range(1990,2019))
test_years = list(range(2019,2022))

data_train = create_graph_data(df, train_years)
data_test = create_graph_data(df, test_years)

# -----------------------
# Train GWNN
# -----------------------
model = GWNN(in_channels=data_train.x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 300

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data_train.x, data_train.edge_index, data_train.edge_weight)
    loss = F.mse_loss(out, data_train.y)
    loss.backward()
    optimizer.step()

# -----------------------
# Predictions & evaluation
# -----------------------
model.eval()
with torch.no_grad():
    y_pred_train = model(data_train.x, data_train.edge_index, data_train.edge_weight).numpy()
    y_true_train = data_train.y.numpy()
    y_pred_test = model(data_test.x, data_test.edge_index, data_test.edge_weight).numpy()
    y_true_test = data_test.y.numpy()

metrics_train = evaluate(y_true_train, y_pred_train, n_features=data_train.x.shape[1])
metrics_test = evaluate(y_true_test, y_pred_test, n_features=data_test.x.shape[1])

# -----------------------
# Save to Excel
# -----------------------
with pd.ExcelWriter("J:/Research/Research/WorldBMI/ABSDATA/underweight_GeoAI_GWNN_Evaluation.xlsx", engine='openpyxl') as writer:
    df_train = pd.DataFrame([metrics_train], index=['Train'])
    df_test = pd.DataFrame([metrics_test], index=['Test'])
    df_all = pd.concat([df_train, df_test])
    df_all.to_excel(writer, sheet_name="TemporalSplit", index=True)

print("Done. Excel saved with GWNN metrics (residual projection).")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

features = [f'X{i}' for i in range(1,23)]
target = 'underweight'

# -----------------------
# Impute missing years with country mean
# -----------------------
all_years = list(range(1990,2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country']==country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            for f in features:
                row[f] = df_country[f].mean() if f in df_country.columns else 0
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)
df = df.sort_values(['Country','Year']).reset_index(drop=True)

# -----------------------
# Standardize features and log-transform target
# -----------------------
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
df[target] = np.log1p(df[target])

# -----------------------
# Create PyG graph data
# -----------------------
def create_graph_data(df_subset, years):
    df_year = df_subset[df_subset['Year'].isin(years)].copy()
    countries = df_year['Country'].unique()
    df_year = df_year.set_index('Country').loc[countries].reset_index()
    
    x = torch.tensor(df_year[features].values, dtype=torch.float)
    y = torch.tensor(df_year[target].values, dtype=torch.float).unsqueeze(1)
    
    N = len(countries)
    adj = np.ones((N,N)) - np.eye(N)  # fully connected minus self-loops
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(adj.sum(axis=1) + 1e-10))
    adj_norm = deg_inv_sqrt @ adj @ deg_inv_sqrt
    edge_index = torch.tensor(np.array(np.nonzero(adj_norm)), dtype=torch.long)
    edge_weight = torch.tensor(adj_norm[np.nonzero(adj_norm)], dtype=torch.float)
    
    return Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weight)

# -----------------------
# Define GAT model
# -----------------------
class GAT(nn.Module):
    def __init__(self, in_channels, hidden=64, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden, heads=heads, concat=True)
        self.gat2 = GATConv(hidden*heads, hidden, heads=1, concat=False)
        self.fc = nn.Linear(hidden,1)
        
    def forward(self, x, edge_index, edge_weight=None):
        x = F.elu(self.gat1(x, edge_index, edge_weight))
        x = F.elu(self.gat2(x, edge_index, edge_weight))
        out = self.fc(x)
        return out

# -----------------------
# Evaluation function
# -----------------------
def evaluate(y_true, y_pred, n_features):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2)*(n-1)/(n - n_features -1)
    
    se_rmse = stats.sem(np.abs(y_true - y_pred))
    se_mae = stats.sem(np.abs(y_true - y_pred))
    
    rmse_ci = f"{rmse:.4f} ({rmse - 1.96*se_rmse:.4f}-{rmse + 1.96*se_rmse:.4f})"
    mae_ci = f"{mae:.4f} ({mae - 1.96*se_mae:.4f}-{mae + 1.96*se_mae:.4f})"
    
    return {'RMSE (95% CI)': rmse_ci, 'MAE (95% CI)': mae_ci,
            'R2': round(r2,4), 'Adj_R2': round(adj_r2,4)}

# -----------------------
# Temporal split
# -----------------------
train_years = list(range(1990,2019))
test_years = list(range(2019,2022))

data_train = create_graph_data(df, train_years)
data_test = create_graph_data(df, test_years)

# -----------------------
# Train GAT
# -----------------------
model = GAT(in_channels=data_train.x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 300

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data_train.x, data_train.edge_index, data_train.edge_weight)
    loss = F.mse_loss(out, data_train.y)
    loss.backward()
    optimizer.step()

# -----------------------
# Predictions & evaluation
# -----------------------
model.eval()
with torch.no_grad():
    y_pred_train = model(data_train.x, data_train.edge_index, data_train.edge_weight).numpy()
    y_true_train = data_train.y.numpy()
    y_pred_test = model(data_test.x, data_test.edge_index, data_test.edge_weight).numpy()
    y_true_test = data_test.y.numpy()

metrics_train = evaluate(y_true_train, y_pred_train, n_features=data_train.x.shape[1])
metrics_test = evaluate(y_true_test, y_pred_test, n_features=data_test.x.shape[1])

# -----------------------
# Save to Excel
# -----------------------
with pd.ExcelWriter("J:/Research/Research/WorldBMI/ABSDATA/underweight_GeoAI_GAT_Evaluation.xlsx", engine='openpyxl') as writer:
    df_train = pd.DataFrame([metrics_train], index=['Train'])
    df_test = pd.DataFrame([metrics_test], index=['Test'])
    df_all = pd.concat([df_train, df_test])
    df_all.to_excel(writer, sheet_name="TemporalSplit", index=True)

print("Done. Excel saved with GAT metrics.")



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pycountry_convert as pc

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df["Year"] = df["Year"].astype(int)

features = [f"X{i}" for i in range(1,23)]
target = "obesity"

# -----------------------
# Impute missing years
# -----------------------
all_years = list(range(1990,2022))
countries = df["Country"].unique()
imputed_rows = []

for c in countries:
    df_c = df[df["Country"] == c].copy()
    mean_target = df_c[target].mean()
    for y in all_years:
        if y not in df_c["Year"].values:
            row = {"Country":c,"Year":y,target:mean_target}
            for f in features:
                if f in df_c.columns:
                    row[f] = df_c[f].mean()
                else:
                    row[f] = 0
            imputed_rows.append(row)

if len(imputed_rows) > 0:
    df = pd.concat([df,pd.DataFrame(imputed_rows)],ignore_index=True)

df = df.sort_values(["Country","Year"]).reset_index(drop=True)

# -----------------------
# Standardize features
# -----------------------
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
df[target] = np.log1p(df[target])

# -----------------------
# Create graph data
# -----------------------
def create_graph(df_subset):
    x = torch.tensor(df_subset[features].values,dtype=torch.float)
    y = torch.tensor(df_subset[target].values,dtype=torch.float).unsqueeze(1)
    N = len(df_subset)
    adj = np.ones((N,N)) - np.eye(N)
    edge_index = torch.tensor(np.array(np.nonzero(adj)),dtype=torch.long)
    return Data(x=x,y=y,edge_index=edge_index)

data = create_graph(df)

# -----------------------
# Define STGNN
# -----------------------
class STGNN(nn.Module):
    def __init__(self,in_channels,hidden=64):
        super(STGNN,self).__init__()
        self.fc1 = nn.Linear(in_channels,hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        self.lstm = nn.LSTM(hidden,hidden,batch_first=True)
        self.fc_out = nn.Linear(hidden,1)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.unsqueeze(1)
        x,_ = self.lstm(x)
        x = x.squeeze(1)
        out = self.fc_out(x)
        return out

# -----------------------
# Train STGNN model
# -----------------------
model = STGNN(in_channels=len(features))
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    pred = model(data.x)
    loss = F.mse_loss(pred,data.y)
    loss.backward()
    optimizer.step()

# -----------------------
# Integrated gradients
# -----------------------
def integrated_gradients(model,x,steps=50):
    baseline = torch.zeros_like(x)
    scaled_inputs = [baseline + (float(i)/steps)*(x-baseline) for i in range(steps+1)]
    grads = []
    for s in scaled_inputs:
        s.requires_grad = True
        out = model(s)
        out.sum().backward()
        grads.append(s.grad.detach().numpy())
    grads = np.array(grads)
    avg_grads = grads.mean(axis=0)
    ig = (x.detach().numpy() - baseline.detach().numpy()) * avg_grads
    return ig

X_tensor = torch.tensor(df[features].values,dtype=torch.float)
ig_values = integrated_gradients(model,X_tensor)

ig_df = pd.DataFrame(ig_values,columns=features)
ig_df["Country"] = df["Country"].values
ig_df["Year"] = df["Year"].values

# -----------------------
# Global IG with 95% CI
# -----------------------
global_list = []
for f in features:
    vals = np.abs(ig_df[f].values)
    mean_val = np.mean(vals)
    se = stats.sem(vals)
    lower = mean_val - 1.96*se
    upper = mean_val + 1.96*se
    global_list.append({
        "Feature":f,
        "Mean_IG":mean_val,
        "CI_Lower":lower,
        "CI_Upper":upper
    })
global_df = pd.DataFrame(global_list)
global_df = global_df.sort_values("Mean_IG",ascending=False)

# -----------------------
# Map countries to regions automatically
# -----------------------
def country_to_region(country):
    try:
        code = pc.country_name_to_country_alpha2(country)
        continent = pc.country_alpha2_to_continent_code(code)
        mapping = {
            "AF":"Africa",
            "AS":"Asia",
            "EU":"Europe",
            "NA":"North America",
            "SA":"South America",
            "OC":"Oceania"
        }
        return mapping.get(continent,"Other")
    except:
        return "Other"

ig_df["Region"] = ig_df["Country"].apply(country_to_region)

# -----------------------
# Regional IG
# -----------------------
regional_df = ig_df.groupby("Region")[features].mean().abs().reset_index()

# -----------------------
# Country IG
# -----------------------
country_df = ig_df.groupby("Country")[features].mean().abs().reset_index()

# -----------------------
# Save results to Excel
# -----------------------
output_file = "J:/Research/Research/WorldBMI/ABSDATA/IG_GeoAI_Results.xlsx"

with pd.ExcelWriter(output_file,engine="openpyxl") as writer:
    global_df.to_excel(writer,sheet_name="Global_IG",index=False)
    regional_df.to_excel(writer,sheet_name="Regional_IG",index=False)
    country_df.to_excel(writer,sheet_name="Country_IG",index=False)

print("Finished. Excel file saved successfully.")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pycountry_convert as pc

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df["Year"] = df["Year"].astype(int)

features = [f"X{i}" for i in range(1,23)]
target = "overweight"

# -----------------------
# Impute missing years
# -----------------------
all_years = list(range(1990,2022))
countries = df["Country"].unique()
imputed_rows = []

for c in countries:
    df_c = df[df["Country"] == c].copy()
    mean_target = df_c[target].mean()
    for y in all_years:
        if y not in df_c["Year"].values:
            row = {"Country":c,"Year":y,target:mean_target}
            for f in features:
                if f in df_c.columns:
                    row[f] = df_c[f].mean()
                else:
                    row[f] = 0
            imputed_rows.append(row)

if len(imputed_rows) > 0:
    df = pd.concat([df,pd.DataFrame(imputed_rows)],ignore_index=True)

df = df.sort_values(["Country","Year"]).reset_index(drop=True)

# -----------------------
# Standardize features
# -----------------------
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
df[target] = np.log1p(df[target])

# -----------------------
# Create graph data
# -----------------------
def create_graph(df_subset):
    x = torch.tensor(df_subset[features].values,dtype=torch.float)
    y = torch.tensor(df_subset[target].values,dtype=torch.float).unsqueeze(1)
    N = len(df_subset)
    adj = np.ones((N,N)) - np.eye(N)
    edge_index = torch.tensor(np.array(np.nonzero(adj)),dtype=torch.long)
    return Data(x=x,y=y,edge_index=edge_index)

data = create_graph(df)

# -----------------------
# Define STGNN
# -----------------------
class STGNN(nn.Module):
    def __init__(self,in_channels,hidden=64):
        super(STGNN,self).__init__()
        self.fc1 = nn.Linear(in_channels,hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        self.lstm = nn.LSTM(hidden,hidden,batch_first=True)
        self.fc_out = nn.Linear(hidden,1)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.unsqueeze(1)
        x,_ = self.lstm(x)
        x = x.squeeze(1)
        out = self.fc_out(x)
        return out

# -----------------------
# Train STGNN model
# -----------------------
model = STGNN(in_channels=len(features))
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    pred = model(data.x)
    loss = F.mse_loss(pred,data.y)
    loss.backward()
    optimizer.step()

# -----------------------
# Integrated gradients
# -----------------------
def integrated_gradients(model,x,steps=50):
    baseline = torch.zeros_like(x)
    scaled_inputs = [baseline + (float(i)/steps)*(x-baseline) for i in range(steps+1)]
    grads = []
    for s in scaled_inputs:
        s.requires_grad = True
        out = model(s)
        out.sum().backward()
        grads.append(s.grad.detach().numpy())
    grads = np.array(grads)
    avg_grads = grads.mean(axis=0)
    ig = (x.detach().numpy() - baseline.detach().numpy()) * avg_grads
    return ig

X_tensor = torch.tensor(df[features].values,dtype=torch.float)
ig_values = integrated_gradients(model,X_tensor)

ig_df = pd.DataFrame(ig_values,columns=features)
ig_df["Country"] = df["Country"].values
ig_df["Year"] = df["Year"].values

# -----------------------
# Global IG with 95% CI
# -----------------------
global_list = []
for f in features:
    vals = np.abs(ig_df[f].values)
    mean_val = np.mean(vals)
    se = stats.sem(vals)
    lower = mean_val - 1.96*se
    upper = mean_val + 1.96*se
    global_list.append({
        "Feature":f,
        "Mean_IG":mean_val,
        "CI_Lower":lower,
        "CI_Upper":upper
    })
global_df = pd.DataFrame(global_list)
global_df = global_df.sort_values("Mean_IG",ascending=False)

# -----------------------
# Map countries to regions automatically
# -----------------------
def country_to_region(country):
    try:
        code = pc.country_name_to_country_alpha2(country)
        continent = pc.country_alpha2_to_continent_code(code)
        mapping = {
            "AF":"Africa",
            "AS":"Asia",
            "EU":"Europe",
            "NA":"North America",
            "SA":"South America",
            "OC":"Oceania"
        }
        return mapping.get(continent,"Other")
    except:
        return "Other"

ig_df["Region"] = ig_df["Country"].apply(country_to_region)

# -----------------------
# Regional IG
# -----------------------
regional_df = ig_df.groupby("Region")[features].mean().abs().reset_index()

# -----------------------
# Country IG
# -----------------------
country_df = ig_df.groupby("Country")[features].mean().abs().reset_index()

# -----------------------
# Save results to Excel
# -----------------------
output_file = "J:/Research/Research/WorldBMI/ABSDATA/overweight_IG_GeoAI_Results.xlsx"

with pd.ExcelWriter(output_file,engine="openpyxl") as writer:
    global_df.to_excel(writer,sheet_name="Global_IG",index=False)
    regional_df.to_excel(writer,sheet_name="Regional_IG",index=False)
    country_df.to_excel(writer,sheet_name="Country_IG",index=False)

print("Finished. Excel file saved successfully.")



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pycountry_convert as pc

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df["Year"] = df["Year"].astype(int)

features = [f"X{i}" for i in range(1,23)]
target = "underweight"

# -----------------------
# Impute missing years
# -----------------------
all_years = list(range(1990,2022))
countries = df["Country"].unique()
imputed_rows = []

for c in countries:
    df_c = df[df["Country"] == c].copy()
    mean_target = df_c[target].mean()
    for y in all_years:
        if y not in df_c["Year"].values:
            row = {"Country":c,"Year":y,target:mean_target}
            for f in features:
                if f in df_c.columns:
                    row[f] = df_c[f].mean()
                else:
                    row[f] = 0
            imputed_rows.append(row)

if len(imputed_rows) > 0:
    df = pd.concat([df,pd.DataFrame(imputed_rows)],ignore_index=True)

df = df.sort_values(["Country","Year"]).reset_index(drop=True)

# -----------------------
# Standardize features
# -----------------------
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
df[target] = np.log1p(df[target])

# -----------------------
# Create graph data
# -----------------------
def create_graph(df_subset):
    x = torch.tensor(df_subset[features].values,dtype=torch.float)
    y = torch.tensor(df_subset[target].values,dtype=torch.float).unsqueeze(1)
    N = len(df_subset)
    adj = np.ones((N,N)) - np.eye(N)
    edge_index = torch.tensor(np.array(np.nonzero(adj)),dtype=torch.long)
    return Data(x=x,y=y,edge_index=edge_index)

data = create_graph(df)

# -----------------------
# Define STGNN
# -----------------------
class STGNN(nn.Module):
    def __init__(self,in_channels,hidden=64):
        super(STGNN,self).__init__()
        self.fc1 = nn.Linear(in_channels,hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        self.lstm = nn.LSTM(hidden,hidden,batch_first=True)
        self.fc_out = nn.Linear(hidden,1)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.unsqueeze(1)
        x,_ = self.lstm(x)
        x = x.squeeze(1)
        out = self.fc_out(x)
        return out

# -----------------------
# Train STGNN model
# -----------------------
model = STGNN(in_channels=len(features))
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    pred = model(data.x)
    loss = F.mse_loss(pred,data.y)
    loss.backward()
    optimizer.step()

# -----------------------
# Integrated gradients
# -----------------------
def integrated_gradients(model,x,steps=50):
    baseline = torch.zeros_like(x)
    scaled_inputs = [baseline + (float(i)/steps)*(x-baseline) for i in range(steps+1)]
    grads = []
    for s in scaled_inputs:
        s.requires_grad = True
        out = model(s)
        out.sum().backward()
        grads.append(s.grad.detach().numpy())
    grads = np.array(grads)
    avg_grads = grads.mean(axis=0)
    ig = (x.detach().numpy() - baseline.detach().numpy()) * avg_grads
    return ig

X_tensor = torch.tensor(df[features].values,dtype=torch.float)
ig_values = integrated_gradients(model,X_tensor)

ig_df = pd.DataFrame(ig_values,columns=features)
ig_df["Country"] = df["Country"].values
ig_df["Year"] = df["Year"].values

# -----------------------
# Global IG with 95% CI
# -----------------------
global_list = []
for f in features:
    vals = np.abs(ig_df[f].values)
    mean_val = np.mean(vals)
    se = stats.sem(vals)
    lower = mean_val - 1.96*se
    upper = mean_val + 1.96*se
    global_list.append({
        "Feature":f,
        "Mean_IG":mean_val,
        "CI_Lower":lower,
        "CI_Upper":upper
    })
global_df = pd.DataFrame(global_list)
global_df = global_df.sort_values("Mean_IG",ascending=False)

# -----------------------
# Map countries to regions automatically
# -----------------------
def country_to_region(country):
    try:
        code = pc.country_name_to_country_alpha2(country)
        continent = pc.country_alpha2_to_continent_code(code)
        mapping = {
            "AF":"Africa",
            "AS":"Asia",
            "EU":"Europe",
            "NA":"North America",
            "SA":"South America",
            "OC":"Oceania"
        }
        return mapping.get(continent,"Other")
    except:
        return "Other"

ig_df["Region"] = ig_df["Country"].apply(country_to_region)

# -----------------------
# Regional IG
# -----------------------
regional_df = ig_df.groupby("Region")[features].mean().abs().reset_index()

# -----------------------
# Country IG
# -----------------------
country_df = ig_df.groupby("Country")[features].mean().abs().reset_index()

# -----------------------
# Save results to Excel
# -----------------------
output_file = "J:/Research/Research/WorldBMI/ABSDATA/underweight_IG_GeoAI_Results.xlsx"

with pd.ExcelWriter(output_file,engine="openpyxl") as writer:
    global_df.to_excel(writer,sheet_name="Global_IG",index=False)
    regional_df.to_excel(writer,sheet_name="Regional_IG",index=False)
    country_df.to_excel(writer,sheet_name="Country_IG",index=False)

print("Finished. Excel file saved successfully.")



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

features = [f'X{i}' for i in range(1,23)]
target = 'obesity'

# -----------------------
# Impute missing years with country mean
# -----------------------
all_years = list(range(1990,2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country']==country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            for f in features:
                row[f] = df_country[f].mean() if f in df_country.columns else 0
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)
df = df.sort_values(['Country','Year']).reset_index(drop=True)

# -----------------------
# Standardize features & log-transform target
# -----------------------
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
df[target] = np.log1p(df[target])

# -----------------------
# Function: weighted correlation with bootstrap CI & p-value
# -----------------------
def weighted_corr_bootstrap(x, y, weights=None, n_boot=1000, random_state=42):
    """
    Compute weighted correlation between vectors x and y, with bootstrap CI and p-value.
    weights: vector of weights, or None
    Returns: r, CI_lower, CI_upper, p_value
    """
    rng = np.random.default_rng(random_state)
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    if weights is None:
        weights = np.ones(n)
    weights = weights / np.sum(weights)
    
    # weighted mean
    xm = np.sum(weights * x)
    ym = np.sum(weights * y)
    
    # weighted correlation
    cov = np.sum(weights * (x - xm) * (y - ym))
    var_x = np.sum(weights * (x - xm)**2)
    var_y = np.sum(weights * (y - ym)**2)
    if var_x <= 0 or var_y <= 0:
        return np.nan, np.nan, np.nan, np.nan
    r = cov / np.sqrt(var_x * var_y)
    
    # bootstrap CI
    boot_r = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True, p=weights)
        x_b = x[idx]
        y_b = y[idx]
        w_b = weights[idx]
        w_b = w_b / np.sum(w_b)
        xm_b = np.sum(w_b * x_b)
        ym_b = np.sum(w_b * y_b)
        cov_b = np.sum(w_b * (x_b - xm_b) * (y_b - ym_b))
        var_x_b = np.sum(w_b * (x_b - xm_b)**2)
        var_y_b = np.sum(w_b * (y_b - ym_b)**2)
        if var_x_b > 0 and var_y_b > 0:
            boot_r.append(cov_b / np.sqrt(var_x_b * var_y_b))
    if len(boot_r) == 0:
        return np.nan, np.nan, np.nan, np.nan
    ci_lower = np.percentile(boot_r, 2.5)
    ci_upper = np.percentile(boot_r, 97.5)
    
    # p-value (two-sided)
    t_stat = r * np.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else 0
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2)) if abs(r) < 1 else 0
    
    return r, ci_lower, ci_upper, p_val

# -----------------------
# Compute spatio-temporal weighted correlation
# -----------------------
results = []

for feat in features:
    # Spatial weighting: weights within year (countries equally weighted)
    r_list = []
    for yr in all_years:
        df_year = df[df['Year']==yr]
        x = df_year[feat].values
        y = df_year[target].values
        w = np.ones(len(x)) / len(x)
        r, _, _, _ = weighted_corr_bootstrap(x, y, weights=w, n_boot=200)
        r_list.append(r)
    # Temporal weighting: weights within country (years equally weighted)
    r_temp_list = []
    for country in all_countries:
        df_country = df[df['Country']==country]
        x = df_country[feat].values
        y = df_country[target].values
        w = np.ones(len(x)) / len(x)
        r_c, _, _, _ = weighted_corr_bootstrap(x, y, weights=w, n_boot=200)
        r_temp_list.append(r_c)
    
    # Combine: simple average of spatial and temporal correlation
    r_final = np.nanmean(r_list + r_temp_list)
    
    # Bootstrap CI and p-value on full vector
    x_full = df[feat].values
    y_full = df[target].values
    r_boot, ci_low, ci_high, p_val = weighted_corr_bootstrap(x_full, y_full, n_boot=1000)
    
    results.append({
        'Feature': feat,
        'WeightedCorr': r_final,
        'CI_lower': ci_low,
        'CI_upper': ci_high,
        'p_value': p_val
    })

# -----------------------
# Save to Excel
# -----------------------
df_res = pd.DataFrame(results)
df_res.to_excel("J:/Research/Research/WorldBMI/ABSDATA/SpatioTemporalWeightedCorr_Bootstrap.xlsx", index=False)

print("Done. Spatio-temporal weighted correlations with 95% CI and p-values saved to Excel.")



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

features = [f'X{i}' for i in range(1,23)]
target = 'overweight'

# -----------------------
# Impute missing years with country mean
# -----------------------
all_years = list(range(1990,2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country']==country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            for f in features:
                row[f] = df_country[f].mean() if f in df_country.columns else 0
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)
df = df.sort_values(['Country','Year']).reset_index(drop=True)

# -----------------------
# Standardize features & log-transform target
# -----------------------
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
df[target] = np.log1p(df[target])

# -----------------------
# Function: weighted correlation with bootstrap CI & p-value
# -----------------------
def weighted_corr_bootstrap(x, y, weights=None, n_boot=1000, random_state=42):
    """
    Compute weighted correlation between vectors x and y, with bootstrap CI and p-value.
    weights: vector of weights, or None
    Returns: r, CI_lower, CI_upper, p_value
    """
    rng = np.random.default_rng(random_state)
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    if weights is None:
        weights = np.ones(n)
    weights = weights / np.sum(weights)
    
    # weighted mean
    xm = np.sum(weights * x)
    ym = np.sum(weights * y)
    
    # weighted correlation
    cov = np.sum(weights * (x - xm) * (y - ym))
    var_x = np.sum(weights * (x - xm)**2)
    var_y = np.sum(weights * (y - ym)**2)
    if var_x <= 0 or var_y <= 0:
        return np.nan, np.nan, np.nan, np.nan
    r = cov / np.sqrt(var_x * var_y)
    
    # bootstrap CI
    boot_r = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True, p=weights)
        x_b = x[idx]
        y_b = y[idx]
        w_b = weights[idx]
        w_b = w_b / np.sum(w_b)
        xm_b = np.sum(w_b * x_b)
        ym_b = np.sum(w_b * y_b)
        cov_b = np.sum(w_b * (x_b - xm_b) * (y_b - ym_b))
        var_x_b = np.sum(w_b * (x_b - xm_b)**2)
        var_y_b = np.sum(w_b * (y_b - ym_b)**2)
        if var_x_b > 0 and var_y_b > 0:
            boot_r.append(cov_b / np.sqrt(var_x_b * var_y_b))
    if len(boot_r) == 0:
        return np.nan, np.nan, np.nan, np.nan
    ci_lower = np.percentile(boot_r, 2.5)
    ci_upper = np.percentile(boot_r, 97.5)
    
    # p-value (two-sided)
    t_stat = r * np.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else 0
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2)) if abs(r) < 1 else 0
    
    return r, ci_lower, ci_upper, p_val

# -----------------------
# Compute spatio-temporal weighted correlation
# -----------------------
results = []

for feat in features:
    # Spatial weighting: weights within year (countries equally weighted)
    r_list = []
    for yr in all_years:
        df_year = df[df['Year']==yr]
        x = df_year[feat].values
        y = df_year[target].values
        w = np.ones(len(x)) / len(x)
        r, _, _, _ = weighted_corr_bootstrap(x, y, weights=w, n_boot=200)
        r_list.append(r)
    # Temporal weighting: weights within country (years equally weighted)
    r_temp_list = []
    for country in all_countries:
        df_country = df[df['Country']==country]
        x = df_country[feat].values
        y = df_country[target].values
        w = np.ones(len(x)) / len(x)
        r_c, _, _, _ = weighted_corr_bootstrap(x, y, weights=w, n_boot=200)
        r_temp_list.append(r_c)
    
    # Combine: simple average of spatial and temporal correlation
    r_final = np.nanmean(r_list + r_temp_list)
    
    # Bootstrap CI and p-value on full vector
    x_full = df[feat].values
    y_full = df[target].values
    r_boot, ci_low, ci_high, p_val = weighted_corr_bootstrap(x_full, y_full, n_boot=1000)
    
    results.append({
        'Feature': feat,
        'WeightedCorr': r_final,
        'CI_lower': ci_low,
        'CI_upper': ci_high,
        'p_value': p_val
    })

# -----------------------
# Save to Excel
# -----------------------
df_res = pd.DataFrame(results)
df_res.to_excel("J:/Research/Research/WorldBMI/ABSDATA/overweight_SpatioTemporalWeightedCorr_Bootstrap.xlsx", index=False)

print("Done. Spatio-temporal weighted correlations with 95% CI and p-values saved to Excel.")



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

features = [f'X{i}' for i in range(1,23)]
target = 'underweight'

# -----------------------
# Impute missing years with country mean
# -----------------------
all_years = list(range(1990,2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country']==country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            for f in features:
                row[f] = df_country[f].mean() if f in df_country.columns else 0
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)
df = df.sort_values(['Country','Year']).reset_index(drop=True)

# -----------------------
# Standardize features & log-transform target
# -----------------------
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
df[target] = np.log1p(df[target])

# -----------------------
# Function: weighted correlation with bootstrap CI & p-value
# -----------------------
def weighted_corr_bootstrap(x, y, weights=None, n_boot=1000, random_state=42):
    """
    Compute weighted correlation between vectors x and y, with bootstrap CI and p-value.
    weights: vector of weights, or None
    Returns: r, CI_lower, CI_upper, p_value
    """
    rng = np.random.default_rng(random_state)
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    if weights is None:
        weights = np.ones(n)
    weights = weights / np.sum(weights)
    
    # weighted mean
    xm = np.sum(weights * x)
    ym = np.sum(weights * y)
    
    # weighted correlation
    cov = np.sum(weights * (x - xm) * (y - ym))
    var_x = np.sum(weights * (x - xm)**2)
    var_y = np.sum(weights * (y - ym)**2)
    if var_x <= 0 or var_y <= 0:
        return np.nan, np.nan, np.nan, np.nan
    r = cov / np.sqrt(var_x * var_y)
    
    # bootstrap CI
    boot_r = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True, p=weights)
        x_b = x[idx]
        y_b = y[idx]
        w_b = weights[idx]
        w_b = w_b / np.sum(w_b)
        xm_b = np.sum(w_b * x_b)
        ym_b = np.sum(w_b * y_b)
        cov_b = np.sum(w_b * (x_b - xm_b) * (y_b - ym_b))
        var_x_b = np.sum(w_b * (x_b - xm_b)**2)
        var_y_b = np.sum(w_b * (y_b - ym_b)**2)
        if var_x_b > 0 and var_y_b > 0:
            boot_r.append(cov_b / np.sqrt(var_x_b * var_y_b))
    if len(boot_r) == 0:
        return np.nan, np.nan, np.nan, np.nan
    ci_lower = np.percentile(boot_r, 2.5)
    ci_upper = np.percentile(boot_r, 97.5)
    
    # p-value (two-sided)
    t_stat = r * np.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else 0
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2)) if abs(r) < 1 else 0
    
    return r, ci_lower, ci_upper, p_val

# -----------------------
# Compute spatio-temporal weighted correlation
# -----------------------
results = []

for feat in features:
    # Spatial weighting: weights within year (countries equally weighted)
    r_list = []
    for yr in all_years:
        df_year = df[df['Year']==yr]
        x = df_year[feat].values
        y = df_year[target].values
        w = np.ones(len(x)) / len(x)
        r, _, _, _ = weighted_corr_bootstrap(x, y, weights=w, n_boot=200)
        r_list.append(r)
    # Temporal weighting: weights within country (years equally weighted)
    r_temp_list = []
    for country in all_countries:
        df_country = df[df['Country']==country]
        x = df_country[feat].values
        y = df_country[target].values
        w = np.ones(len(x)) / len(x)
        r_c, _, _, _ = weighted_corr_bootstrap(x, y, weights=w, n_boot=200)
        r_temp_list.append(r_c)
    
    # Combine: simple average of spatial and temporal correlation
    r_final = np.nanmean(r_list + r_temp_list)
    
    # Bootstrap CI and p-value on full vector
    x_full = df[feat].values
    y_full = df[target].values
    r_boot, ci_low, ci_high, p_val = weighted_corr_bootstrap(x_full, y_full, n_boot=1000)
    
    results.append({
        'Feature': feat,
        'WeightedCorr': r_final,
        'CI_lower': ci_low,
        'CI_upper': ci_high,
        'p_value': p_val
    })

# -----------------------
# Save to Excel
# -----------------------
df_res = pd.DataFrame(results)
df_res.to_excel("J:/Research/Research/WorldBMI/ABSDATA/underweightt_SpatioTemporalWeightedCorr_Bootstrap.xlsx", index=False)

print("Done. Spatio-temporal weighted correlations with 95% CI and p-values saved to Excel.")



import pandas as pd
import numpy as np
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

target = 'obesity'

# -----------------------
# Impute missing years
# -----------------------
all_years = list(range(1990, 2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country'] == country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)

df = df.sort_values(['Year', 'Country']).reset_index(drop=True)

# -----------------------
# Spatial weights (fully connected minus self)
# -----------------------
N = len(all_countries)
W = np.ones((N, N)) - np.eye(N)

# -----------------------
# Gi* calculation
# -----------------------
def gi_star(x, W):
    x = np.array(x)
    W = np.array(W)
    x_mean = x.mean()
    x_var = x.var(ddof=0)
    n = len(x)
    sum_w = W.sum(axis=1)
    
    num = W @ x - x_mean * sum_w
    denom = np.sqrt(x_var * ((n * (W**2).sum(axis=1) - sum_w**2) / (n - 1)))
    denom[denom == 0] = 1e-10
    z = num / denom
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return z, p

# -----------------------
# Hotspot/coldspot label function for each CI
# -----------------------
def label_hot_cold(z, alpha):
    threshold = stats.norm.ppf(1 - alpha/2)
    labels = []
    for zi in z:
        if zi > threshold:
            labels.append("Hotspot")
        elif zi < -threshold:
            labels.append("Coldspot")
        else:
            labels.append("Not significant")
    return labels

# -----------------------
# Run Gi* per year and save
# -----------------------
excel_path = "J:/Research/Research/WorldBMI/ABSDATA/GiStar_Hotspot_Obesity_AllCI.xlsx"
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    for year in all_years:
        df_year = df[df['Year'] == year].copy().reset_index(drop=True)
        z_score, p_value = gi_star(df_year[target].values, W)
        
        # Labels for each confidence level
        label_90 = label_hot_cold(z_score, 0.10)
        label_95 = label_hot_cold(z_score, 0.05)
        label_99 = label_hot_cold(z_score, 0.01)
        
        df_res = pd.DataFrame({
            'Country': df_year['Country'],
            'GiZscore': z_score,
            'GiPvalue': p_value,
            'Hot-Col_90': label_90,
            'Hot-Col_95': label_95,
            'Hot-Col_99': label_99
        })
        
        df_res.to_excel(writer, sheet_name=str(year), index=False)

print("Done. Gi* hotspot/coldspot analysis saved to Excel for all confidence levels.")



import pandas as pd
import numpy as np
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

target = 'overweight'

# -----------------------
# Impute missing years
# -----------------------
all_years = list(range(1990, 2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country'] == country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)

df = df.sort_values(['Year', 'Country']).reset_index(drop=True)

# -----------------------
# Spatial weights (fully connected minus self)
# -----------------------
N = len(all_countries)
W = np.ones((N, N)) - np.eye(N)

# -----------------------
# Gi* calculation
# -----------------------
def gi_star(x, W):
    x = np.array(x)
    W = np.array(W)
    x_mean = x.mean()
    x_var = x.var(ddof=0)
    n = len(x)
    sum_w = W.sum(axis=1)
    
    num = W @ x - x_mean * sum_w
    denom = np.sqrt(x_var * ((n * (W**2).sum(axis=1) - sum_w**2) / (n - 1)))
    denom[denom == 0] = 1e-10
    z = num / denom
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return z, p

# -----------------------
# Hotspot/coldspot label function for each CI
# -----------------------
def label_hot_cold(z, alpha):
    threshold = stats.norm.ppf(1 - alpha/2)
    labels = []
    for zi in z:
        if zi > threshold:
            labels.append("Hotspot")
        elif zi < -threshold:
            labels.append("Coldspot")
        else:
            labels.append("Not significant")
    return labels

# -----------------------
# Run Gi* per year and save
# -----------------------
excel_path = "J:/Research/Research/WorldBMI/ABSDATA/overweight_GiStar_Hotspot_Obesity_AllCI.xlsx"
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    for year in all_years:
        df_year = df[df['Year'] == year].copy().reset_index(drop=True)
        z_score, p_value = gi_star(df_year[target].values, W)
        
        # Labels for each confidence level
        label_90 = label_hot_cold(z_score, 0.10)
        label_95 = label_hot_cold(z_score, 0.05)
        label_99 = label_hot_cold(z_score, 0.01)
        
        df_res = pd.DataFrame({
            'Country': df_year['Country'],
            'GiZscore': z_score,
            'GiPvalue': p_value,
            'Hot-Col_90': label_90,
            'Hot-Col_95': label_95,
            'Hot-Col_99': label_99
        })
        
        df_res.to_excel(writer, sheet_name=str(year), index=False)

print("Done. Gi* hotspot/coldspot analysis saved to Excel for all confidence levels.")



import pandas as pd
import numpy as np
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

target = 'underweight'

# -----------------------
# Impute missing years
# -----------------------
all_years = list(range(1990, 2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country'] == country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)

df = df.sort_values(['Year', 'Country']).reset_index(drop=True)

# -----------------------
# Spatial weights (fully connected minus self)
# -----------------------
N = len(all_countries)
W = np.ones((N, N)) - np.eye(N)

# -----------------------
# Gi* calculation
# -----------------------
def gi_star(x, W):
    x = np.array(x)
    W = np.array(W)
    x_mean = x.mean()
    x_var = x.var(ddof=0)
    n = len(x)
    sum_w = W.sum(axis=1)
    
    num = W @ x - x_mean * sum_w
    denom = np.sqrt(x_var * ((n * (W**2).sum(axis=1) - sum_w**2) / (n - 1)))
    denom[denom == 0] = 1e-10
    z = num / denom
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return z, p

# -----------------------
# Hotspot/coldspot label function for each CI
# -----------------------
def label_hot_cold(z, alpha):
    threshold = stats.norm.ppf(1 - alpha/2)
    labels = []
    for zi in z:
        if zi > threshold:
            labels.append("Hotspot")
        elif zi < -threshold:
            labels.append("Coldspot")
        else:
            labels.append("Not significant")
    return labels

# -----------------------
# Run Gi* per year and save
# -----------------------
excel_path = "J:/Research/Research/WorldBMI/ABSDATA/underweight_GiStar_Hotspot_Obesity_AllCI.xlsx"
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    for year in all_years:
        df_year = df[df['Year'] == year].copy().reset_index(drop=True)
        z_score, p_value = gi_star(df_year[target].values, W)
        
        # Labels for each confidence level
        label_90 = label_hot_cold(z_score, 0.10)
        label_95 = label_hot_cold(z_score, 0.05)
        label_99 = label_hot_cold(z_score, 0.01)
        
        df_res = pd.DataFrame({
            'Country': df_year['Country'],
            'GiZscore': z_score,
            'GiPvalue': p_value,
            'Hot-Col_90': label_90,
            'Hot-Col_95': label_95,
            'Hot-Col_99': label_99
        })
        
        df_res.to_excel(writer, sheet_name=str(year), index=False)

print("Done. Gi* hotspot/coldspot analysis saved to Excel for all confidence levels.")



import pandas as pd
import numpy as np
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

target = 'obesity'

# -----------------------
# Impute missing years
# -----------------------
all_years = list(range(1990, 2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country'] == country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)

df = df.sort_values(['Year', 'Country']).reset_index(drop=True)

# -----------------------
# Aggregate over years (average per country)
# -----------------------
df_agg = df.groupby('Country')[target].mean().reset_index()

# -----------------------
# Spatial weights (fully connected minus self)
# -----------------------
N = len(df_agg)
W = np.ones((N, N)) - np.eye(N)

# -----------------------
# Gi* calculation
# -----------------------
def gi_star(x, W):
    x = np.array(x)
    W = np.array(W)
    x_mean = x.mean()
    x_var = x.var(ddof=0)
    n = len(x)
    sum_w = W.sum(axis=1)
    
    num = W @ x - x_mean * sum_w
    denom = np.sqrt(x_var * ((n * (W**2).sum(axis=1) - sum_w**2) / (n - 1)))
    denom[denom == 0] = 1e-10
    z = num / denom
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return z, p

# -----------------------
# Hotspot/coldspot labels for confidence levels
# -----------------------
def label_hot_cold(z, alpha):
    threshold = stats.norm.ppf(1 - alpha/2)
    labels = []
    for zi in z:
        if zi > threshold:
            labels.append("Hotspot")
        elif zi < -threshold:
            labels.append("Coldspot")
        else:
            labels.append("Not significant")
    return labels

# -----------------------
# Apply Gi* for overall years
# -----------------------
z_score, p_value = gi_star(df_agg[target].values, W)

# Labels for different confidence levels
label_90 = label_hot_cold(z_score, 0.10)
label_95 = label_hot_cold(z_score, 0.05)
label_99 = label_hot_cold(z_score, 0.01)

# -----------------------
# Save results to Excel
# -----------------------
df_res = pd.DataFrame({
    'Country': df_agg['Country'],
    'GiZscore': z_score,
    'GiPvalue': p_value,
    'Hot-Col_90': label_90,
    'Hot-Col_95': label_95,
    'Hot-Col_99': label_99
})

df_res.to_excel("J:/Research/Research/WorldBMI/ABSDATA/GiStar_Hotspot_Obesity_Overall.xlsx", index=False)

print("Done. Overall Gi* hotspot/coldspot analysis saved to Excel for all confidence levels.")



import pandas as pd
import numpy as np
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

target = 'overweight'

# -----------------------
# Impute missing years
# -----------------------
all_years = list(range(1990, 2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country'] == country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)

df = df.sort_values(['Year', 'Country']).reset_index(drop=True)

# -----------------------
# Aggregate over years (average per country)
# -----------------------
df_agg = df.groupby('Country')[target].mean().reset_index()

# -----------------------
# Spatial weights (fully connected minus self)
# -----------------------
N = len(df_agg)
W = np.ones((N, N)) - np.eye(N)

# -----------------------
# Gi* calculation
# -----------------------
def gi_star(x, W):
    x = np.array(x)
    W = np.array(W)
    x_mean = x.mean()
    x_var = x.var(ddof=0)
    n = len(x)
    sum_w = W.sum(axis=1)
    
    num = W @ x - x_mean * sum_w
    denom = np.sqrt(x_var * ((n * (W**2).sum(axis=1) - sum_w**2) / (n - 1)))
    denom[denom == 0] = 1e-10
    z = num / denom
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return z, p

# -----------------------
# Hotspot/coldspot labels for confidence levels
# -----------------------
def label_hot_cold(z, alpha):
    threshold = stats.norm.ppf(1 - alpha/2)
    labels = []
    for zi in z:
        if zi > threshold:
            labels.append("Hotspot")
        elif zi < -threshold:
            labels.append("Coldspot")
        else:
            labels.append("Not significant")
    return labels

# -----------------------
# Apply Gi* for overall years
# -----------------------
z_score, p_value = gi_star(df_agg[target].values, W)

# Labels for different confidence levels
label_90 = label_hot_cold(z_score, 0.10)
label_95 = label_hot_cold(z_score, 0.05)
label_99 = label_hot_cold(z_score, 0.01)

# -----------------------
# Save results to Excel
# -----------------------
df_res = pd.DataFrame({
    'Country': df_agg['Country'],
    'GiZscore': z_score,
    'GiPvalue': p_value,
    'Hot-Col_90': label_90,
    'Hot-Col_95': label_95,
    'Hot-Col_99': label_99
})

df_res.to_excel("J:/Research/Research/WorldBMI/ABSDATA/GiStar_Hotspot_overweight_Overall.xlsx", index=False)

print("Done. Overall Gi* hotspot/coldspot analysis saved to Excel for all confidence levels.")


import pandas as pd
import numpy as np
from scipy import stats

# -----------------------
# Load dataset
# -----------------------
data_path = "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
df = pd.read_excel(data_path)
df['Year'] = df['Year'].astype(int)

target = 'underweight'

# -----------------------
# Impute missing years
# -----------------------
all_years = list(range(1990, 2022))
all_countries = df['Country'].unique()
imputed_rows = []

for country in all_countries:
    df_country = df[df['Country'] == country].copy()
    country_mean = df_country[target].mean()
    for year in all_years:
        if year not in df_country['Year'].values:
            row = {'Country': country, 'Year': year, target: country_mean}
            imputed_rows.append(row)

if imputed_rows:
    df = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)

df = df.sort_values(['Year', 'Country']).reset_index(drop=True)

# -----------------------
# Aggregate over years (average per country)
# -----------------------
df_agg = df.groupby('Country')[target].mean().reset_index()

# -----------------------
# Spatial weights (fully connected minus self)
# -----------------------
N = len(df_agg)
W = np.ones((N, N)) - np.eye(N)

# -----------------------
# Gi* calculation
# -----------------------
def gi_star(x, W):
    x = np.array(x)
    W = np.array(W)
    x_mean = x.mean()
    x_var = x.var(ddof=0)
    n = len(x)
    sum_w = W.sum(axis=1)
    
    num = W @ x - x_mean * sum_w
    denom = np.sqrt(x_var * ((n * (W**2).sum(axis=1) - sum_w**2) / (n - 1)))
    denom[denom == 0] = 1e-10
    z = num / denom
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return z, p

# -----------------------
# Hotspot/coldspot labels for confidence levels
# -----------------------
def label_hot_cold(z, alpha):
    threshold = stats.norm.ppf(1 - alpha/2)
    labels = []
    for zi in z:
        if zi > threshold:
            labels.append("Hotspot")
        elif zi < -threshold:
            labels.append("Coldspot")
        else:
            labels.append("Not significant")
    return labels

# -----------------------
# Apply Gi* for overall years
# -----------------------
z_score, p_value = gi_star(df_agg[target].values, W)

# Labels for different confidence levels
label_90 = label_hot_cold(z_score, 0.10)
label_95 = label_hot_cold(z_score, 0.05)
label_99 = label_hot_cold(z_score, 0.01)

# -----------------------
# Save results to Excel
# -----------------------
df_res = pd.DataFrame({
    'Country': df_agg['Country'],
    'GiZscore': z_score,
    'GiPvalue': p_value,
    'Hot-Col_90': label_90,
    'Hot-Col_95': label_95,
    'Hot-Col_99': label_99
})

df_res.to_excel("J:/Research/Research/WorldBMI/ABSDATA/GiStar_Hotspot_underweight_Overall.xlsx", index=False)

print("Done. Overall Gi* hotspot/coldspot analysis saved to Excel for all confidence levels.")


