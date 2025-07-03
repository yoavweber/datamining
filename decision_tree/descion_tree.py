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
    total_entropy_val, _, _ = calculate_entropy(df[target_col])

    for column in df.columns:
        if column == target_col:
            continue

        field_probabilities = df[column].value_counts(normalize=True).to_dict()
        num_options = len(field_probabilities)  # Get number of unique values

        # Get conditional entropy and its components
        cond_entropy_val, components = conditional_entropy(df, column, target_col)
        # Calculate Information Gain
        gain_val = total_entropy_val - cond_entropy_val

        table_data.append(
            {
                "Field Attribute": column,
                "Field Probabilities": field_probabilities,
                "Num Options": num_options,  # Store number of options
                "Total Entropy": round(total_entropy_val, 4),
                "Conditional Entropy": round(cond_entropy_val, 4),
                "Conditional Entropy Components": components,
                "Information Gain": round(gain_val, 4),
            }
        )

    result_df = pd.DataFrame(table_data)
    # Sort first by Information Gain (desc), then by Num Options (asc)
    # result_df = result_df.sort_values(
    #     by=["Information Gain", "Num Options"], ascending=[False, True]
    # )
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
        # Print headline for the column/attribute
        print(f"--- Entropy Calculation for: {attr} ---")
        prob = row["Formatted Probabilities"]
        cond_formula_detailed = row["Formatted Conditional Entropy Detailed"]
        cond_formula_sum = row["Formatted Conditional Entropy Sum"]
        gain_formula = row["Formatted Information Gain"]  # Get formatted gain

        # Print each section on a new line with a label
        print(f"Attribute: {attr}")
        print(f"Probabilities: {prob}")
        # Show how probabilities were calculated
        # Get the original probability dict for this attribute
        prob_dict = row["Field Probabilities"]
        total_count = sum([v for v in prob_dict.values()])
        prob_calc_lines = []
        for val, p in prob_dict.items():
            count = int(round(p * total_count)) if total_count > 0 else 0
            prob_calc_lines.append(f"P({val}) = {count}/{int(total_count)} = {p:.2f}")
        # print(f"Cond Entropy: {cond_formula_detailed}")
        # Print conditional entropy details as a markdown table
        print("| Value | P(Value) | Entropy Formula | Entropy Value |")
        print("|-------|----------|-----------------|---------------|")
        for val, p_val, e_subset, formula_subset, _ in row[
            "Conditional Entropy Components"
        ]:
            print(f"| {val} | {p_val:.2f} | {formula_subset} | {e_subset:.4f} |")
        print(f"Cond Entropy Sum: {cond_formula_sum}")
        print(f"Gain: {gain_formula}")
        print()  # Add a blank line for spacing between columns


def filter_dataframe_by_value(df, column_name, value):
    """Filters the DataFrame to keep only rows where column_name equals value.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to filter on.
        value: The specific value to filter by in the specified column.

    Returns:
        pd.DataFrame: A new DataFrame containing only the filtered rows.
    """
    filtered_df = df[df[column_name] == value].copy()
    return filtered_df


def build_decision_tree_level(
    current_df, target_column, available_attributes, level=0, max_depth=2
):
    """Recursively calculates and prints entropy tables for decision tree levels."""
    indent = "  " * level  # Indentation for printing
    print(
        f"\n{indent}--- Analyzing Level {level} --- Subset Size: {len(current_df)} ---"
    )

    # --- Base Cases ---
    if current_df.empty:
        print(f"{indent}Subset is empty. Stopping branch.")
        return

    if level >= max_depth:
        print(f"{indent}Reached max depth ({max_depth}). Stopping branch.")
        # Determine majority class if needed (optional)
        if not current_df.empty:
            majority_class = current_df[target_column].mode()[0]
            print(f"{indent}Majority class in this subset: {majority_class}")
        return

    subset_target_entropy, _, _ = calculate_entropy(current_df[target_column])
    print(
        f"{indent}Entropy of target ('{target_column}') in this subset: {subset_target_entropy:.4f}"
    )

    if subset_target_entropy == 0:
        class_label = current_df[target_column].iloc[0]  # Faster than unique()[0]
        print(
            f"{indent}Subset is PURE. All instances have Target = '{class_label}'. Stopping branch."
        )
        return

    if not available_attributes:
        print(f"{indent}No more attributes to split on.")
        majority_class = current_df[target_column].mode()[0]
        print(f"{indent}Majority class in this leaf: {majority_class}")
        return

    # --- Recursive Step ---
    print(f"{indent}Calculating entropy table for impure subset...")
    # Create a temporary DataFrame with only available attributes + target for table generation
    df_for_table = current_df[available_attributes + [target_column]].copy()
    entropy_table = generate_entropy_table(df_for_table, target_column)

    if entropy_table.empty:
        print(
            f"{indent}Entropy table is empty (likely only target column remains or constant attributes)."
        )
        majority_class = current_df[target_column].mode()[0]
        print(f"{indent}Majority class in this leaf: {majority_class}")
        return

    print(f"\n{indent}" + "=" * 70)
    print(f"{indent}--- Entropy Table (Level {level}, Subset) ---")
    print_pretty_table(entropy_table)
    print(f"{indent}" + "=" * 70)
    print(f"{indent}" + "-" * 50)

    best_attribute = entropy_table.iloc[0]["Field Attribute"]
    print(f"{indent}Best Attribute to Split On at Level {level}: {best_attribute}")

    # Prepare for next level
    next_available_attributes = [
        attr for attr in available_attributes if attr != best_attribute
    ]
    attribute_values = list(entropy_table.iloc[0]["Field Probabilities"].keys())

    for value in attribute_values:
        print(f"{indent}--> Branching on '{best_attribute}' == '{value}'")
        # Filter original subset (current_df) based on the split
        filtered_data = filter_dataframe_by_value(current_df, best_attribute, value)
        # Recursive call
        build_decision_tree_level(
            filtered_data,
            target_column,
            next_available_attributes,
            level + 1,
            max_depth,
        )


# --- Main Execution Block --- (Replaces previous main block)
if __name__ == "__main__":
    # --- Setup ---
    filepath = "exams/2024/78/dataset.csv"
    target_column = "סוג דיאטה"
    # Drop the first column (index 0) instead of by name
    # id_column = "Customer No. " # No longer needed
    max_recursion_depth = 4  # Adjust as needed

    print(f"Loading data from: {filepath}")
    df = read_csv_data(filepath)
    print(f"Initial DataFrame shape: {df.shape}")

    # --- Preprocessing ---
    if df.shape[1] > 1:  # Ensure there's more than one column
        first_col_name = df.columns[0]
        print(f"Dropping the first column (index 0): '{first_col_name}'")
        df = df.iloc[:, 1:]  # Select all rows, and columns from index 1 onwards
        print(f"DataFrame shape after dropping first column: {df.shape}")
    else:
        print(
            "Warning: DataFrame has only one column or is empty. Cannot drop first column."
        )

    # Check if target column still exists after potentially dropping columns
    if target_column not in df.columns:
        print(
            f"Error: Target column '{target_column}' not found in the DataFrame after preprocessing."
        )
    else:
        # Get initial list of attributes to consider (all columns except target)
        initial_attributes = [col for col in df.columns if col != target_column]

        if not initial_attributes:
            print(
                "Error: No attributes available for analysis after dropping first column and target column."
            )
        else:
            print(f"Target Column: '{target_column}'")
            print(f"Initial Attributes for Analysis: {initial_attributes}")
            print("=" * 60)

            # --- Start Recursive Tree Build Process ---
            build_decision_tree_level(
                df, target_column, initial_attributes, max_depth=max_recursion_depth
            )

            print("\n" + "=" * 60)
            print("Decision tree analysis process finished.")
