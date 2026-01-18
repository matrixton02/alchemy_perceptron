import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import mlp_model as mlp
import matplotlib.pyplot as plt
import pickle
def process_qm9(csv_path,target='gap',vector_size=2048):
    print(f"Loading {csv_path}")
    df=pd.read_csv(csv_path)
    smiles_col=[c for c in df.columns if 'smile' in c.lower()][0]

    if target not in df.columns:
        print(f"Warning: '{target}' column not found. Available columns:")
        print(df.columns.tolist())
        return None, None
    
    print(f"Input: {smiles_col} | Target: {target} (Molecular Orbital Energy)")

    X_data=[]
    y_data=[]
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=vector_size)
    for idx,row in df.iterrows():
        mol=Chem.MolFromSmiles(row[smiles_col])
        if mol:
            fp_array=morgan_gen.GetFingerprintAsNumPy(mol).astype(np.float32)
            X_data.append(fp_array)
            y_data.append(row[target])
    
    X = np.array(X_data)
    y = np.array(y_data).reshape(-1, 1)

    y_norm = (y - y.mean()) / y.std()  #we normalized the oirbital energies
    print(f"Ready for Training! X: {X.shape}, y: {y_norm.shape}")
    return X, y_norm

def split_data(X,y,train_ratio=0.8,seed=42):
    np.random.seed(seed)

    indices=np.arange(X.shape[0])
    np.random.shuffle(indices)

    split_idx=int(X.shape[0]*train_ratio)

    train_idx,test_idx=indices[:split_idx],indices[split_idx:]

    X_train,X_test=X[train_idx],X[test_idx]
    y_train,y_test=y[train_idx],y[test_idx]

    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    return X_train,y_train,X_test,y_test

def evaluate_rmse(x,y,parameters,config):
    AL,_=mlp.forward_propagation(x,parameters,config)
    mse=np.mean((AL-y)**2)
    rmse=np.sqrt(mse)
    return rmse

def train_model(X_train,y_train,X_test,y_test,hidden,lr,iterations):
    print(f"Input Shape (Transposed): {X_train.shape}")
    input_size=X_train.shape[0]
    output_size=1

    layers=[]
    layers.append(input_size)
    layers.extend(hidden)
    layers.append(output_size)
    print(f"Network Architecture: {layers}")
    config={
        "activation_function":"relu",
        "output_activation_function":"linear",
        "loss":"mse"
        }
    
    print("Starting Training...")
    parameters,cost_history=mlp.model(X_train,y_train,layers,config,lr,iterations)

    train_rmse = evaluate_rmse(X_train, y_train, parameters, config)
    test_rmse = evaluate_rmse(X_test, y_test, parameters, config)
    
    print(f"\nFinal Results:")
    print(f"Train RMSE: {train_rmse:.5f}")
    print(f"Test RMSE:  {test_rmse:.5f}")
    
    return parameters, cost_history,config

def plot_parity(X_test,y_test,parameters,config,y_mean,y_std):
    AL,_=mlp.forward_propagation(X_test,parameters,config)

    preds_real=(AL*y_std)+y_mean
    actual_real=(y_test*y_std)+y_mean

    real_mse=np.mean((preds_real-actual_real)**2)
    real_rmse=np.sqrt(real_mse)

    rmse_ev=real_rmse*27.2114  # since 1 hartree =27.2114 eV


    print(f"Final RMSE: {real_rmse:.4f} Ha")
    print(f"Final RMSE: {rmse_ev:.4f} eV")

    plt.figure(figsize=(8, 8),dpi=100)

    plt.scatter(actual_real.flatten(), preds_real.flatten(), alpha=0.5, s=5, color='#2980b9', label='Test Molecules')
    min_val = min(actual_real.min(), preds_real.min())
    max_val = max(actual_real.max(), preds_real.max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             color='#c0392b', linestyle='--', linewidth=2, label='Ideal Prediction')
    
    plt.xlabel(f"Actual Band Gap (Hartree)")
    plt.ylabel(f"Predicted Band Gap (Hartree)")
    plt.title(f'Neural Network from Scratch: QM9 Band Gap Prediction\nRMSE = {real_rmse:.4f} Ha ({rmse_ev:.2f} eV)', fontsize=14)
    lims = [min(actual_real), max(actual_real)]
    plt.plot(lims, lims, 'k--', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("final_parity_plot.png")
    print("Plot saved as 'final_parity_plot.png'")
    plt.show()


X,y_norm=process_qm9("qm9.csv",target="gap")
X_train,y_train,X_test,y_test=split_data(X,y_norm,train_ratio=0.8)
X_train=X_train.T
y_train=y_train.T
X_test=X_test.T
y_test=y_test.T

input_size=2048
layers=[512,128,64]
output_size=1

epochs=10
learning_rate=0.01
batch_size=64

parameters, costs,config = train_model(
    X_train, y_train, 
    X_test, y_test, 
    layers, 
    0.005, 
    1500
)
with open("qm9_mlp_scratch.pkl", "wb") as f:
    pickle.dump(parameters, f)
print("Model saved successfully!")

plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("MSE Loss")
plt.title("Training Loss - HOMO/LUMO Gap Prediction")
plt.show()

y_mean=np.mean(y_test)
y_std=np.std(y_test)
plot_parity(X_test,y_test,parameters,config,y_mean,y_std)

