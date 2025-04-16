import pandas as pd
import numpy as np


def read_csv_data(filepath):
    return pd.read_csv(filepath)


def calculate_entropy(series):
    """Calculates entropy and returns the value and its formula string."""
    probabilities = series.value_counts(normalize=True)
    entropy_val = 0
    formula_terms = []
    prob_dict = probabilities.to_dict()

    # Handle log2(0) issue and build formula string
    for class_val, prob in prob_dict.items():
        if prob > 0:
            term = prob * np.log2(prob)
            entropy_val -= term
            # Using 4 decimal places for probabilities in formula for readability
            formula_terms.append(f"{prob:.4f}*log2({prob:.4f})")
        # else: p*log2(p) -> 0 as p->0, so we can ignore prob=0 cases

    # Construct the formula string
    if not formula_terms:
        # Case: Empty series or series with values occurring zero times (should not happen with value_counts)
        formula_str = "0 (empty/no positive prob)"
        entropy_val = 0  # Entropy is 0 for empty set
    elif len(formula_terms) == 1 and entropy_val == 0:
        # Case: Single class (entropy is 0)
        formula_str = f"0 (single class: {list(prob_dict.keys())[0]})"
        entropy_val = 0
    else:
        formula_str = "- [" + " + ".join(formula_terms) + "]"

    # Return entropy value, formula string, and probabilities dict
    return entropy_val, formula_str, prob_dict


def conditional_entropy(df, attribute_col, target_col):
    total_rows = len(df)
    entropy_sum = 0
    # Components: (attr_value, prob_attr_value, subset_entropy, subset_entropy_formula, subset_probs)
    components = []
    probabilities = df[attribute_col].value_counts(normalize=True)

    for attr_value, prob in probabilities.items():
        subset = df[df[attribute_col] == attr_value][target_col]

        # Calculate entropy for the subset
        if subset.empty:
            entropy_of_subset = 0
            subset_formula_str = "0 (empty subset)"
            subset_probs = {}
        # Don't need explicit nunique check, calculate_entropy handles single class case
        else:
            entropy_of_subset, subset_formula_str, subset_probs = calculate_entropy(
                subset
            )

        entropy_sum += prob * entropy_of_subset
        components.append(
            (attr_value, prob, entropy_of_subset, subset_formula_str, subset_probs)
        )

    # Return both the sum and the detailed components
    return entropy_sum, components


def generate_entropy_table(df, target_col):
    table_data = []
    # Calculate the total entropy of the target variable correctly
    # Use calculate_entropy directly on the target column Series
    total_entropy_val, _, _ = calculate_entropy(df[target_col])

    for column in df.columns:
        if column == target_col:
            continue

        field_probabilities = df[column].value_counts(normalize=True).to_dict()
        # Get conditional entropy and its components
        cond_entropy_val, components = conditional_entropy(df, column, target_col)
        # Calculate Information Gain
        gain_val = total_entropy_val - cond_entropy_val

        table_data.append(
            {
                "Field Attribute": column,
                "Field Probabilities": field_probabilities,
                "Total Entropy": round(total_entropy_val, 4),  # Pass total entropy
                "Conditional Entropy": round(cond_entropy_val, 4),
                "Conditional Entropy Components": components,
                "Information Gain": round(gain_val, 4),
            }
        )

    result_df = pd.DataFrame(table_data)
    # Sort by Information Gain in descending order
    result_df = result_df.sort_values(by="Information Gain", ascending=False)
    return result_df


def print_pretty_table(result_df):
    # Convert the Field Probabilities dictionaries to formatted strings
    def format_probabilities(prob_dict):
        formatted = ", ".join(f"{k}: {v:.2f}" for k, v in prob_dict.items())
        return f"{{ {formatted} }}"

    # Format the conditional entropy to show detailed subset entropy calculation
    def format_cond_entropy_detailed(components, final_value):
        terms_dict = {}
        # Components: (val, p_val, e_subset, formula_subset, probs_subset)
        for val, p_val, e_subset, formula_subset, _ in components:
            # Format: P(Val) * ( [Entropy Formula] = Entropy Value )
            calc_str = f"{p_val:.2f}*({formula_subset} = {e_subset:.4f})"
            terms_dict[str(val)] = calc_str

        terms_str = ", ".join(f"{k}: {v}" for k, v in terms_dict.items())
        # Final format: { Value1: P(V1)*([H(S|V1) Calc]=H(S|V1)), ... } = Final Cond Entropy
        return f"{{ {terms_str} }} = {final_value:.4f}"

    # Format the conditional entropy sum formula
    def format_cond_entropy_sum(components, final_value):
        # Components: (val, p_val, e_subset, formula_subset, probs_subset)
        terms = []
        for _, p_val, e_subset, _, _ in components:
            terms.append(f"{p_val:.2f}*{e_subset:.4f}")
        formula = " + ".join(terms)
        return f"{formula} = {final_value:.4f}"

    # Format the Information Gain formula
    def format_gain_formula(total_entropy, cond_entropy, gain):
        return f"{total_entropy:.4f} - {cond_entropy:.4f} = {gain:.4f}"

    # Prepare DataFrame for printing
    print_df = result_df.copy()
    print_df["Formatted Probabilities"] = print_df["Field Probabilities"].apply(
        format_probabilities
    )
    # Apply the detailed formula formatting
    print_df["Formatted Conditional Entropy Detailed"] = print_df.apply(
        lambda row: format_cond_entropy_detailed(
            row["Conditional Entropy Components"], row["Conditional Entropy"]
        ),
        axis=1,
    )
    # Apply the sum formula formatting
    print_df["Formatted Conditional Entropy Sum"] = print_df.apply(
        lambda row: format_cond_entropy_sum(
            row["Conditional Entropy Components"], row["Conditional Entropy"]
        ),
        axis=1,
    )
    # Apply the gain formula formatting
    print_df["Formatted Information Gain"] = print_df.apply(
        lambda row: format_gain_formula(
            row["Total Entropy"], row["Conditional Entropy"], row["Information Gain"]
        ),
        axis=1,
    )

    # Define column headers with custom widths
    header_attr = "Field Attribute"
    header_prob = "Field Probabilities"
    header_cond_detailed = (
        "Cond Entropy Detail ({Value: P(Val)*[Calc=H(S|Val)]...}=Result)"  # Abbreviated
    )
    header_cond_sum = "Cond Entropy Sum (Σ P(Val)*H(S|Val)=Result)"
    header_gain = "Information Gain (H(T) - H(T|A) = Gain)"  # Updated Header

    # Setting column widths
    width_attr = 18
    width_prob = 40
    width_cond_detailed = 100  # Slightly reduced detailed width
    width_cond_sum = 50  # Slightly reduced sum width
    width_gain = 45  # Width for gain formula

    # Update header line and separator to include the new column
    header_line = (
        f"{header_attr:<{width_attr}} | "
        f"{header_prob:<{width_prob}} | "
        f"{header_cond_detailed:<{width_cond_detailed}} | "
        f"{header_cond_sum:<{width_cond_sum}} | "
        f"{header_gain:<{width_gain}}"  # Use updated gain header and width
    )
    separator = "-" * len(header_line)

    print(header_line)
    print(separator)

    # Iterate through the DataFrame rows for printing
    for index, row in print_df.iterrows():
        attr = str(row["Field Attribute"])
        prob = row["Formatted Probabilities"]
        cond_formula_detailed = row["Formatted Conditional Entropy Detailed"]
        cond_formula_sum = row["Formatted Conditional Entropy Sum"]
        gain_formula = row["Formatted Information Gain"]  # Get formatted gain

        # Update line format to include the new formatted gain column
        line = (
            f"{attr:<{width_attr}} | "
            f"{prob:<{width_prob}} | "
            f"{cond_formula_detailed:<{width_cond_detailed}} | "
            f"{cond_formula_sum:<{width_cond_sum}} | "
            f"{gain_formula:<{width_gain}}"  # Use formatted gain
        )
        print(line)
        print()  # Add a blank line for spacing between rows


# Example Usage
if __name__ == "__main__":
    # Load your CSV data
    df = read_csv_data("resturant.csv")

    # Set your target column name
    target_column = "Customer Wait"

    entropy_table = generate_entropy_table(df, target_column)

    # Corrected call: pass the DataFrame directly, not its string representation
    print_pretty_table(entropy_table)
