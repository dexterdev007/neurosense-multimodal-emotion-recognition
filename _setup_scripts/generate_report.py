import os
import joblib
import pandas as pd
from datetime import datetime

def generate_summary():
    print("="*60)
    print("      NEUROSENSE PROJECT - TECHNICAL SUMMARY REPORT")
    print("      Generated on:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)
    
    artifacts_dir = "NeuroSense/artifacts"
    modalities = ['eeg', 'meg', 'mri', 'speech', 'face', 'fusion']
    
    report_data = []
    
    for mod in modalities:
        model_path = os.path.join(artifacts_dir, mod, f"{mod}_model.pkl")
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                # Note: Scikit-learn models don't store training accuracy by default in the pkl,
                # but we can report on the model parameters/type.
                params = model.get_params()
                status = "✅ Trained & Loaded"
                m_type = type(model).__name__
            except:
                status = "⚠️ Load Error"
                m_type = "Unknown"
        else:
            status = "❌ Not Found (Run Notebook)"
            m_type = "N/A"
            
        report_data.append({"Modality": mod.upper(), "Status": status, "Model Type": m_type})
    
    df = pd.DataFrame(report_data)
    print("\n[MODALITY STATUS]")
    print(df.to_string(index=False))
    
    print("\n" + "="*60)
    print("INSTRUCTIONS FOR VIVA/REPORT:")
    print("1. Open notebooks in Jupyter / Google Colab.")
    print("2. Run all cells to generate latest plots and accuracy metrics.")
    print("3. File -> Save as -> PDF (or Print to PDF) for professional reporting.")
    print("4. Use the 'Theory' section in the UI to explain the scientific basis.")
    print("="*60)

if __name__ == "__main__":
    generate_summary()
