import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def combine_images(
    img_paths, output_name, title, subplot_titles=None, width_ratios=None
):
    """Combine images horizontally into a single figure."""
    n = len(img_paths)
    fig_width = 16
    fig_height = 8

    # Load images
    images = []
    for p in img_paths:
        if os.path.exists(p):
            images.append(mpimg.imread(p))
        else:
            print(f"Warning: {p} not found.")
            return

    # Create figure
    if width_ratios:
        fig, axes = plt.subplots(
            1,
            n,
            figsize=(fig_width, fig_height),
            gridspec_kw={"width_ratios": width_ratios},
        )
    else:
        fig, axes = plt.subplots(1, n, figsize=(fig_width, fig_height))

    if n == 1:
        axes = [axes]

    for i in range(n):
        axes[i].imshow(images[i])
        axes[i].axis("off")
        if subplot_titles:
            axes[i].set_title(subplot_titles[i], fontsize=14, pad=10)

    plt.tight_layout()
    output_path = os.path.join("../reports/figures", output_name)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Created: {output_path}")


# --- 1. Data Analysis Composition ---
combine_images(
    [
        "../reports/figures/correlation_combined.png"
    ],  # Use the already combined one if it looks good, or re-stitch
    "final_data_analysis.png",
    "Analyse des Correlations",
)

# --- 2. Classification Composition ---
combine_images(
    [
        "../reports/figures/torch_confusion_matrix.png",
        "../reports/figures/shap_classification_summary.png",
    ],
    "final_classification.png",
    "Performance Classification",
    ["Matrice de Confusion", "Importance des Features (SHAP)"],
    width_ratios=[1, 1.2],
)

# --- 3. Regression Composition ---
combine_images(
    [
        "../reports/figures/torch_reg_predictions.png",
        "../reports/figures/shap_regression_project_grade.png",
    ],
    "final_regression.png",
    "Performance Régression",
    ["Prédictions vs Réalité (Project Grade)", "Importance des Features (SHAP)"],
    width_ratios=[1.2, 1],
)

# --- 4. Accuracy Comparison (Stand-alone but standardized) ---
combine_images(
    ["../reports/figures/sklearn_accuracy_comparison.png"],
    "final_accuracy_comparison.png",
    "Comparaison des Accuracy",
)
