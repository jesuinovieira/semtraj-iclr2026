# A simple script to convert all .xlsx files of this folder to .csv
import os
import pandas as pd


def xlsx_to_csv(path):
    for filename in os.listdir(path):
        if filename.endswith(".xlsx"):
            src = os.path.join(path, filename)
            df = pd.read_excel(src)
            
            # Convert column names to lowercase
            df.columns = [col.lower() for col in df.columns]

            dst = os.path.join(path, filename.replace(".xlsx", ".csv"))
            df.to_csv(dst, index=False)
            print(f"{dst}")


if __name__ == "__main__":
    xlsx_to_csv(os.path.dirname(os.path.abspath(__file__)))
    print("All xlsx files converted to csv")
