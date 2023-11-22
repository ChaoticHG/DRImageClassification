import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

# Define the dataset dictionary list
datasets = [
 {"name": "EfficientV2_M with circlecrop and normalaug"             ,"csv_path": r".\Effv2m\Effv2m_ranger21_300circlecrop_normalaug"},
 {"name": "EfficientV2_M with circlecrop and randaug"               ,"csv_path": r".\Effv2m\Effv2m_ranger21_300circlecrop_randaug"},
 {"name": "EfficientV2_M with circlecrop and trivialaug"            ,"csv_path": r".\Effv2m\Effv2m_ranger21_300circlecrop_trivialaug"},
 {"name": "EfficientV2_M with circlecrop and augmix"                ,"csv_path": r".\Effv2m\Effv2m_ranger21_300circlecrop_augmix"},
 {"name": "EfficientV2_M with circlecrop and fencemask"             ,"csv_path": r".\Effv2m\Effv2m_ranger21_300circlecrop_fencemask"},
 {"name": "EfficientV2_M with circlecrop and gridmask"              ,"csv_path": r".\Effv2m\Effv2m_ranger21_300circlecrop_gridmask"},
 {"name": "EfficientV2_M with circlecrop and resample_normalaug"    ,"csv_path": r".\Effv2m\Effv2m_ranger21_300circlecrop_resample_normalaug"},
 {"name": "EfficientV2_M with circlecrop and resample_randaug"      ,"csv_path": r".\Effv2m\Effv2m_ranger21_300circlecrop_resample_randaug"},
 {"name": "EfficientV2_M with clahe and normalaug"                  ,"csv_path": r".\Effv2m\Effv2m_ranger21_300clahe_normalaug"},
 {"name": "EfficientV2_M with clahe and randaug"                    ,"csv_path": r".\Effv2m\Effv2m_ranger21_300clahe_randaug"},
 {"name": "EfficientV2_M with clahe and trivialaug"                 ,"csv_path": r".\Effv2m\Effv2m_ranger21_300clahe_trivialaug"},
 {"name": "EfficientV2_M with clahe and augmix"                     ,"csv_path": r".\Effv2m\Effv2m_ranger21_300clahe_augmix"},
 {"name": "EfficientV2_M with clahe and fencemask"                  ,"csv_path": r".\Effv2m\Effv2m_ranger21_300clahe_fencemask"},
 {"name": "EfficientV2_M with clahe and gridmask"                   ,"csv_path": r".\Effv2m\Effv2m_ranger21_300clahe_gridmask"},
 {"name": "EfficientV2_M with defaultben and normalaug"             ,"csv_path": r".\Effv2m\Effv2m_ranger21_300defaultben_normalaug"},
 {"name": "EfficientV2_M with defaultben and randaug"               ,"csv_path": r".\Effv2m\Effv2m_ranger21_300defaultben_randaug"},
 {"name": "EfficientV2_M with defaultben and trivialaug"            ,"csv_path": r".\Effv2m\Effv2m_ranger21_300defaultben_trivialaug"},
#HBPASMwEffV2
 {"name": "HBPASM with EFFv2 with clahe and normalaug2"           ,"csv_path": r".\HBPASMwithEffv2\HBPASMwEFFv2_ranger21_300clahe_normalaug2"},
 {"name": "HBPASM with EFFv2 with clahe and randaug2"             ,"csv_path": r".\HBPASMwithEffv2\HBPASMwEFFv2_ranger21_300clahe_randaug2"},
 {"name": "HBPASM with EFFv2 with clahe and trivialaug2"          ,"csv_path": r".\HBPASMwithEffv2\HBPASMwEFFv2_ranger21_300clahe_trivialaug2"},
 {"name": "HBPASM with EFFv2 with clahe and augmix2"              ,"csv_path": r".\HBPASMwithEffv2\HBPASMwEFFv2_ranger21_300clahe_augmix2"},
 {"name": "HBPASM with Effv2 with clahe and fencemask2"           ,"csv_path": r".\HBPASMwithEffv2\HBPASMwEffv2_ranger21_300clahe_fencemask2"},
 {"name": "HBPASM with EFFv2 with clahe and gridmask2"            ,"csv_path": r".\HBPASMwithEffv2\HBPASMwEFFv2_ranger21_300clahe_gridmask2"},
 {"name": "HBPASM with Effv2 with circlecrop and normalaug"    ,"csv_path": r".\HBPASMwithEffv2\HBPASMwithEffv2_ranger21_300circlecrop_normalaug"},
 {"name": "HBPASM with Effv2 with circlecrop and randaug"      ,"csv_path": r".\HBPASMwithEffv2\HBPASMwithEffv2_ranger21_300circlecrop_randaug"},
 {"name": "HBPASM with Effv2 with circlecrop and trivialaug"   ,"csv_path": r".\HBPASMwithEffv2\HBPASMwithEffv2_ranger21_300circlecrop_trivialaug"},
 {"name": "HBPASM with Effv2 with clahe and normalaug"         ,"csv_path": r".\HBPASMwithEffv2\HBPASMwithEffv2_ranger21_300clahe_normalaug"},
 {"name": "HBPASM with Effv2 with clahe and randaug"           ,"csv_path": r".\HBPASMwithEffv2\HBPASMwithEffv2_ranger21_300clahe_randaug"},
 {"name": "HBPASM with Effv2 with clahe and trivialaug"        ,"csv_path": r".\HBPASMwithEffv2\HBPASMwithEffv2_ranger21_300clahe_trivialaug"},
#HBPASMDSOD
 {"name": "DSODHBPASM with circlecrop and normalaug2"        ,"csv_path": r".\HBPASMDSOD\DSODHBPASM_ranger21_300circlecrop_normalaug2"},
 {"name": "DSODHBPASM with circlecrop_resample and normalaug","csv_path": r".\HBPASMDSOD\DSODHBPASM_ranger21_300circlecrop_resample_normalaug"},
 {"name": "DSODHBPASM with circlecrop_resample and randaug"  ,"csv_path": r".\HBPASMDSOD\DSODHBPASM_ranger21_300circlecrop_resample_randaug"},
 {"name": "DSODHBPASM with circlecrop and normalaug"          ,"csv_path": r".\HBPASMDSOD\DSODHBPASM_ranger21_300circlecropnormalaug"},
 {"name": "DSODHBPASM with circlecrop and randaug"            ,"csv_path": r".\HBPASMDSOD\DSODHBPASM_ranger21_300circlecroprandaug"},
 {"name": "DSODHBPASM with circlecrop and trivialaug"         ,"csv_path": r".\HBPASMDSOD\DSODHBPASM_ranger21_300circlecroptrivialaug"},
 {"name": "DSODHBPASM with circlecrop and augmix"             ,"csv_path": r".\HBPASMDSOD\DSODHBPASM_ranger21_300circlecropaugmix"},
 {"name": "DSODHBPASM with circlecrop and fencemask"          ,"csv_path": r".\HBPASMDSOD\DSODHBPASM_ranger21_300circlecropfencemask"},
 {"name": "DSODHBPASM with circlecrop and gridmask"           ,"csv_path": r".\HBPASMDSOD\DSODHBPASM_ranger21_300circlecropgridmask"},
 {"name": "DSODHBPASM with clahe and normalaug"              ,"csv_path": r".\HBPASMDSOD\DSODHBPASM_ranger21_300clahe_normalaug"},
 {"name": "DSODHBPASM with clahe and randaug"                ,"csv_path": r".\HBPASMDSOD\DSODHBPASM_ranger21_300clahe_randaug"},
 {"name": "DSODHBPASM with clahe and trivialaug"             ,"csv_path": r".\HBPASMDSOD\DSODHBPASM_ranger21_300clahe_trivialaug"},
 {"name": "DSODHBPASM with clahe and augmix"                 ,"csv_path": r".\HBPASMDSOD\DSODHBPASM_ranger21_300clahe_augmix"},
 {"name": "DSODHBPASM with clahe and fencemask"              ,"csv_path": r".\HBPASMDSOD\DSODHBPASM_ranger21_300clahe_fencemask"},
 {"name": "DSODHBPASM with clahe and gridmask"               ,"csv_path": r".\HBPASMDSOD\DSODHBPASM_ranger21_300clahe_gridmask"}
]

# Load the selected dataset CSV file
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

# Main Streamlit app
def main():
    st.title("Dashboard for Dataset Visualization")

    # Select dataset from the dropdown
    dataset_selection = st.sidebar.selectbox("Select Dataset", [dataset["name"] for dataset in datasets])
    st.header("Dataset Train and Test")
    # Load the selected dataset
    selected_dataset = next((dataset for dataset in datasets if dataset["name"] == dataset_selection), None)
    if selected_dataset is not None:
        df = load_data(selected_dataset["csv_path"]+".csv")
        if("efficientv2_m" in selected_dataset["name"].lower()):
            st.sidebar.write("Model : EfficientV2_M")
        elif("hbpasmweffv2" in selected_dataset["name"].lower()):
            st.sidebar.write("Model : HBPASM with EfficientNetV2")
        elif("dsodhbpasm" in selected_dataset["name"].lower()):
            st.sidebar.write("Model : HBPASM with DSOD")

        if("circlecrop" in selected_dataset["name"].lower()):
            st.sidebar.write("Preprocessing : Circle Crop")
        elif("clahe" in selected_dataset["name"].lower()):
            st.sidebar.write("Preprocessing : Clahe")
        elif("defaultben" in selected_dataset["name"].lower()):
            st.sidebar.write("Preprocessing : Default Ben's Preprocessing")

        if("normalaug" in selected_dataset["name"].lower()):
            st.sidebar.write("Augmentation : Normal Augment")
        elif("randaug" in selected_dataset["name"].lower()):
            st.sidebar.write("Augmentation : RandAugment")
        elif("trivialaug" in selected_dataset["name"].lower()):
            st.sidebar.write("Augmentation : TrivialAugment")
        elif("augmix" in selected_dataset["name"].lower()):
            st.sidebar.write("Augmentation : Augmix")
        elif("gridmask" in selected_dataset["name"].lower()):
            st.sidebar.write("Augmentation : GridMask")
        elif("fencemask" in selected_dataset["name"].lower()):
            st.sidebar.write("Augmentation : FenceMask")


        metrics = set()
        num_metrics = 6  # Excluding "Train Confusion Matrix" and "Test Confusion Matrix"
        subplot_rows = 2
        subplot_cols = 3

        fig, axs = plt.subplots(subplot_rows, subplot_cols, figsize=(10, 6))

        subplot_idx = 0  # Index to keep track of subplots

        for col in df.columns:
            if col != "Train Confusion Matrix" and col != "Test Confusion Matrix":
                metric_name = col.split(" ")[-1]
                if metric_name not in metrics:
                    metrics.add(metric_name)
                    row = subplot_idx // subplot_cols
                    col = subplot_idx % subplot_cols
                    ax = axs[row, col]
                    ax.plot(df["Train " + metric_name], label="Train")
                    ax.plot(df["Test " + metric_name], label="Test")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Value")
                    ax.set_title(metric_name)
                    ax.legend()

                    subplot_idx += 1

        plt.tight_layout()
        st.pyplot(fig)

        max_qwk_index = np.argmax(df['Test QWK'].values)

        # Get the corresponding confusion matrix data for train and test sets
        train_confmat = df.loc[max_qwk_index, "Train Confusion Matrix"]
        test_confmat = df.loc[max_qwk_index, "Test Confusion Matrix"]
        train_confmat = re.sub(r'[^\d\s]', '', train_confmat)
        test_confmat = re.sub(r'[^\d\s]', '', test_confmat)
        train_confmat = np.array([list(map(int, row.split())) for row in train_confmat.strip().split("\n")])
        test_confmat = np.array([list(map(int, row.split())) for row in test_confmat.strip().split("\n")])
        col1, col2 = st.columns(2)

        # Plot the train confusion matrix
        with col1:
            st.subheader("Train Confusion Matrix")
            fig_train_confmat = plt.figure(figsize=(8, 6))
            sns.heatmap(train_confmat, annot=True, cmap="Blues", fmt="d", annot_kws={"fontsize": 25})
            plt.title("Train Confusion Matrix (QWK={:.4f})".format(df.loc[max_qwk_index, "Test QWK"]), fontsize=25)
            plt.xlabel("Predicted", fontsize=25)
            plt.ylabel("True", fontsize=25)
            st.pyplot(fig_train_confmat)

        # Plot the test confusion matrix
        with col2:
            st.subheader("Test Confusion Matrix")
            fig_test_confmat = plt.figure(figsize=(8, 6))
            sns.heatmap(test_confmat, annot=True, cmap="Blues", fmt="d", annot_kws={"fontsize": 25})
            plt.title("Test Confusion Matrix (QWK={:.4f})".format(df.loc[max_qwk_index, "Test QWK"]), fontsize=25)
            plt.xlabel("Predicted", fontsize=25)
            plt.ylabel("True", fontsize=25)
            st.pyplot(fig_test_confmat)

        st.header("Cross Dataset Testing")
        df2 = load_data(selected_dataset["csv_path"]+"_generalisationtest.csv")
        metrics = set()
        num_metrics = 6  # Excluding "Train Confusion Matrix" and "Test Confusion Matrix"
        subplot_rows = 2
        subplot_cols = 3

        fig, axs = plt.subplots(subplot_rows, subplot_cols, figsize=(10, 6))

        subplot_idx = 0 

        dataset_colors = ['violet', 'blue', 'green', 'orange', 'purple','gold']

        for i, column in enumerate(df2.columns):
            if column != "Test Confusion Matrix" and column != 'Dataset Name':
                row = subplot_idx // subplot_cols
                col = subplot_idx % subplot_cols
                ax = axs[row, col]
                bars = ax.bar(df2['Dataset Name'], df2[column], color=dataset_colors)
                ax.set_xlabel('Dataset Name')
                ax.set_ylabel(column)
                ax.set_title(f'Bar Graph for {column}')
                ax.tick_params(axis='x', rotation=45)
                ax.tick_params(axis='x', which='both', labelbottom=True)
                ax.tick_params(axis='y', which='both', labelleft=True)

                # Assign color to each dataset name
                for bar, color in zip(bars, dataset_colors):
                    bar.set_color(color)

                subplot_idx += 1

        plt.tight_layout()
        st.pyplot(fig)
        #= re.sub(r'[^\d\s]', '', train_confmat)
        # = np.array([list(map(int, row.split())) for row in train_confmat.strip().split("\n")])
        aptos2019_confmat = df.loc[0, "Test Confusion Matrix"]
        aptos2019_confmat = re.sub(r'[^\d\s]', '', aptos2019_confmat)
        aptos2019_confmat = np.array([list(map(int, row.split())) for row in aptos2019_confmat.strip().split("\n")])
        ddrtest_confmat = df.loc[1, "Test Confusion Matrix"]
        ddrtest_confmat = re.sub(r'[^\d\s]', '', ddrtest_confmat)
        ddrtest_confmat = np.array([list(map(int, row.split())) for row in ddrtest_confmat.strip().split("\n")])
        fgadr_confmat = df.loc[2, "Test Confusion Matrix"]
        fgadr_confmat = re.sub(r'[^\d\s]', '', fgadr_confmat)
        fgadr_confmat = np.array([list(map(int, row.split())) for row in fgadr_confmat.strip().split("\n")])
        idridtest_confmat = df.loc[3, "Test Confusion Matrix"]
        idridtest_confmat = re.sub(r'[^\d\s]', '', idridtest_confmat)
        idridtest_confmat = np.array([list(map(int, row.split())) for row in idridtest_confmat.strip().split("\n")])
        retinallesion_confmat = df.loc[4, "Test Confusion Matrix"]
        retinallesion_confmat = re.sub(r'[^\d\s]', '', retinallesion_confmat)
        retinallesion_confmat = np.array([list(map(int, row.split())) for row in retinallesion_confmat.strip().split("\n")])
        kaggletest_confmat = df.loc[5, "Test Confusion Matrix"]
        kaggletest_confmat = re.sub(r'[^\d\s]', '', kaggletest_confmat)
        kaggletest_confmat = np.array([list(map(int, row.split())) for row in kaggletest_confmat.strip().split("\n")])

        col3,col4,col5=st.columns(3)
        #col5,col6=st.columns(2)
        col6,col7,col8=st.columns(3)
        with col3:
            st.subheader("Aptos 2019 Confusion Matrix")
            fig_crosstest1 = plt.figure(figsize=(8, 6))
            sns.heatmap(train_confmat, annot=True, cmap="Blues", fmt="d", annot_kws={"fontsize": 25})
            plt.title("Train Confusion Matrix (QWK={:.4f})".format(df2.loc[0, "Test QWK"]), fontsize=25)
            plt.xlabel("Predicted", fontsize=25)
            plt.ylabel("True", fontsize=25)
            st.pyplot(fig_crosstest1)
        with col4:
            st.subheader("DDR Test Confusion Matrix")
            fig_crosstest2 = plt.figure(figsize=(8, 6))
            sns.heatmap(ddrtest_confmat, annot=True, cmap="Blues", fmt="d", annot_kws={"fontsize": 25})
            plt.title("Train Confusion Matrix (QWK={:.4f})".format(df2.loc[1, "Test QWK"]), fontsize=25)
            plt.xlabel("Predicted", fontsize=25)
            plt.ylabel("True", fontsize=25)
            st.pyplot(fig_crosstest2)
        with col5:
            st.subheader("FGADR Confusion Matrix")
            fig_crosstest3 = plt.figure(figsize=(8, 6))
            sns.heatmap(fgadr_confmat, annot=True, cmap="Blues", fmt="d", annot_kws={"fontsize": 25})
            plt.title("Train Confusion Matrix (QWK={:.4f})".format(df2.loc[2, "Test QWK"]), fontsize=25)
            plt.xlabel("Predicted", fontsize=25)
            plt.ylabel("True", fontsize=25)
            st.pyplot(fig_crosstest3)
        with col6:
            st.subheader("IDRID Confusion Matrix")
            fig_crosstest4 = plt.figure(figsize=(8, 6))
            sns.heatmap(idridtest_confmat, annot=True, cmap="Blues", fmt="d", annot_kws={"fontsize": 25})
            plt.title("Train Confusion Matrix (QWK={:.4f})".format(df2.loc[3, "Test QWK"]), fontsize=25)
            plt.xlabel("Predicted", fontsize=25)
            plt.ylabel("True", fontsize=25)
            st.pyplot(fig_crosstest4)
        with col7:
            st.subheader("Retinal-lesion Confusion Matrix")
            fig_crosstest5 = plt.figure(figsize=(8, 6))
            sns.heatmap(retinallesion_confmat, annot=True, cmap="Blues", fmt="d", annot_kws={"fontsize": 25})
            plt.title("Train Confusion Matrix (QWK={:.4f})".format(df2.loc[4, "Test QWK"]), fontsize=25)
            plt.xlabel("Predicted", fontsize=25)
            plt.ylabel("True", fontsize=25)
            st.pyplot(fig_crosstest5)
        with col8:
            st.subheader("Kaggle Test Confusion Matrix")
            fig_crosstest6 = plt.figure(figsize=(8, 6))
            sns.heatmap(kaggletest_confmat, annot=True, cmap="Blues", fmt="d", annot_kws={"fontsize": 25})
            plt.title("Train Confusion Matrix (QWK={:.4f})".format(df2.loc[5, "Test QWK"]), fontsize=25)
            plt.xlabel("Predicted", fontsize=25)
            plt.ylabel("True", fontsize=25)
            st.pyplot(fig_crosstest6)
# Run the app
if __name__ == "__main__":
    main()
