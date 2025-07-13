import pandas as pd
import numpy as np
import io
import contextlib
from fractions import Fraction

# Helper: pretty-print the decision tree structure once built
def _print_tree(node, indent=""):
    """Recursively prints the learned decision-tree structure."""
    if not node:
        return
    # Leaf node
    if "label" in node:
        print(f"{indent}└─ Leaf → {node['label']}")
        return

    attr = node.get("attribute")
    if attr is None:
        return
    for val, child in node.get("branches", {}).items():
        print(f"{indent}└─ {attr} = {val}")
        _print_tree(child, indent + "   ")

# Helper: compute maximum depth of built tree (root depth = 0)
def _tree_max_depth(node):
    if not node or "label" in node:
        return 0
    return 1 + max((_tree_max_depth(c) for c in node.get("branches", {}).values()), default=0)


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
            frac_str = str(Fraction(prob).limit_denominator())
            formula_terms.append(f"{frac_str} X log2({frac_str})")
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
    # Rank attributes: 1) lowest conditional entropy (better split)
    #                   2) if tie, attribute with MORE distinct values to favour deeper trees
    result_df = result_df.sort_values(
        by=["Conditional Entropy", "Num Options"],
        ascending=[True, False],
    )

    return result_df


def print_pretty_table(result_df, df, target_col, parent_label="ROOT"):
    # Convert the Field Probabilities dictionaries to formatted strings
    def format_probabilities(prob_dict):
        formatted_items = []
        for k, v in prob_dict.items():
            frac = Fraction(v).limit_denominator()
            formatted_items.append(f"{k}: {frac}")
        return f"{{ {', '.join(formatted_items)} }}"

    # Format the conditional entropy to show detailed subset entropy calculation
    def format_cond_entropy_detailed(components, final_value):
        terms_dict = {}
        # Components: (val, p_val, e_subset, formula_subset, probs_subset)
        for val, p_val, e_subset, formula_subset, _ in components:
            # Format: P(Val) * ( [Entropy Formula] = Entropy Value )
            frac_str = str(Fraction(p_val).limit_denominator())
            calc_str = f"{frac_str}*({formula_subset} = {e_subset:.4f})"
            terms_dict[str(val)] = calc_str

        terms_str = ", ".join(f"{k}: {v}" for k, v in terms_dict.items())
        # Final format: { Value1: P(V1)*([H(S|V1) Calc]=H(S|V1)), ... } = Final Cond Entropy
        return f"{{ {terms_str} }} = {final_value:.4f}"

    # Format the conditional entropy sum formula
    def format_cond_entropy_sum(components, final_value):
        # Components: (val, p_val, e_subset, formula_subset, probs_subset)
        terms = []
        for _, p_val, e_subset, _, _ in components:
            terms.append(f"{str(Fraction(p_val).limit_denominator())}X{e_subset:.4f}")
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

    # NOTE: Information Gain is no longer required, so we skip its formatting.

    # Collect summary for all attributes
    summary_rows = []
    for index, row in print_df.iterrows():
        attr = str(row["Field Attribute"])
        # Print headline including parent context
        print(f"--- Entropy Calculation for: {attr} (Parent: {parent_label}) ---")
        prob = row["Formatted Probabilities"]
        cond_formula_detailed = row["Formatted Conditional Entropy Detailed"]
        cond_formula_sum = row["Formatted Conditional Entropy Sum"]
        cond_entropy_val = row["Conditional Entropy"]

        # Print each section on a new line with a label
        print(f"Attribute: {attr}")
        print(f"Probabilities: {prob} \n")
        # Show how probabilities were calculated
        prob_dict = row["Field Probabilities"]
        total_count = sum([v for v in prob_dict.values()])
        prob_calc_lines = []
        for val, p in prob_dict.items():
            count = round(p * total_count, 2) if total_count > 0 else 0
            prob_calc_lines.append(f"P({val}) = {str(Fraction(p).limit_denominator())}")
        # Print summary for each value of the attribute as a table
        # Get all possible classes in the target column for this attribute
        all_classes = sorted(df[target_col].unique())
        # Calculate total for this attribute (sum of all counts for its values)
        total_for_attribute = sum(
            [len(df[df[attr] == val]) for val in row["Field Probabilities"].keys()]
        )
        # Print table header
        header = [
            "Value",
            'סה"כ',
            "Probability (Fraction)",
            "Probability (Decimal)",
        ] + all_classes
        print("| " + " | ".join(header) + " |")
        print("|" + "-------|" * len(header))
        for val, _ in row["Field Probabilities"].items():
            subset = df[df[attr] == val]
            total = len(subset)
            # Probability as fraction and decimal
            prob_fraction = (
                f"{total}/{total_for_attribute}" if total_for_attribute > 0 else "0/0"
            )
            prob_decimal = (
                f"{(total/total_for_attribute):.2f}"
                if total_for_attribute > 0
                else "0.00"
            )
            value_counts = subset[target_col].value_counts()
            row_vals = [val, str(total), prob_fraction, prob_decimal] + [
                str(value_counts.get(cls, 0)) for cls in all_classes
            ]
            print("| " + " | ".join(row_vals) + " |")
        print("")
        # Print conditional entropy details as a markdown table
        print("| Value | P(Value) | Entropy Formula | Entropy Value |")
        print("|-------|----------|-----------------|---------------|")
        for val, p_val, e_subset, formula_subset, _ in row[
            "Conditional Entropy Components"
        ]:
            print(f"| {val} | {p_val:.2f} | {formula_subset} | {e_subset:.4f} |")
        print("\n")
        print(f"Cond Entropy Sum: {cond_formula_sum}")
        # Print the conditional entropy value for this attribute
        print(f"Conditional Entropy: {cond_entropy_val:.4f}")
        print()  # Add a blank line for spacing between columns
        # Collect for summary
        summary_rows.append((attr, cond_entropy_val))

    # Print summary table
    print(f"Summary of Conditional Entropy for this level (Parent: {parent_label}): \n")
    print("| Attribute | Conditional Entropy |")
    print("|-----------|------------------|")
    for attr, cond in summary_rows:
        print(f"| {attr} | {cond:.4f} |")
    print()


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
    current_df,
    target_column,
    available_attributes,
    level=0,
    max_depth=2,
    tree_node=None,
    rule="ROOT",
    forced_root_attr=None,
):
    """Recursively calculates and prints entropy tables for decision tree levels."""
    indent = "  " * level  # Indentation for printing

    # Initialise the node holder on first call
    if tree_node is None:
        tree_node = {}

    print(
        f"\n{indent}--- Analyzing Level {level} --- Subset Size: {len(current_df)} (Parent: {rule}) ---"
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

    # Calculate and print entropy details for the target column
    subset_target_entropy, entropy_formula, prob_dict = calculate_entropy(
        current_df[target_column]
    )
    # Print counts for each class
    value_counts = current_df[target_column].value_counts()
    print(f"{indent}Target value counts: {dict(value_counts)}")
    # Print probabilities for each class
    print(
        f"{indent}Target probabilities: {', '.join([f'{k}: {v:.4f}' for k, v in prob_dict.items()])}"
    )
    # Print the entropy formula
    print(f"{indent}Entropy formula: {entropy_formula}")
    print(
        f"{indent}Entropy of target ('{target_column}') in this subset: {subset_target_entropy:.4f}"
    )

    if subset_target_entropy == 0:
        class_label = current_df[target_column].iloc[0]
        tree_node["label"] = class_label  # mark leaf
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
    # Create a temporary DataFrame with only available attributes + target for table generation
    df_for_table = current_df[available_attributes + [target_column]].copy()
    entropy_table = generate_entropy_table(df_for_table, target_column)

    # If at root and a specific attribute is forced, move it to the top
    if level == 0 and forced_root_attr is not None:
        if forced_root_attr in entropy_table["Field Attribute"].values:
            idx = entropy_table.index[entropy_table["Field Attribute"] == forced_root_attr][0]
            entropy_table = pd.concat([entropy_table.loc[[idx]], entropy_table.drop(idx)], ignore_index=True)

    if entropy_table.empty:
        print(
            f"{indent}Entropy table is empty (likely only target column remains or constant attributes)."
        )
        majority_class = current_df[target_column].mode()[0]
        print(f"{indent}Majority class in this leaf: {majority_class}")
        return

    print(f"\n{indent}" + "=" * 70)
    print(f"{indent}--- Entropy Table (Level {level}) ---")
    print_pretty_table(entropy_table, current_df, target_column, parent_label=rule)
    print(f"{indent}" + "=" * 70)
    print(f"{indent}" + "-" * 50)

    best_attribute = entropy_table.iloc[0]["Field Attribute"]
    tree_node["attribute"] = best_attribute
    tree_node["branches"] = {}
    print(f"{indent}Best Attribute to Split On at Level {level}: {best_attribute}")

    print(f"{indent}Moving to the next level of the decision tree: {level}")
    # Prepare for next level
    next_available_attributes = [
        attr for attr in available_attributes if attr != best_attribute
    ]
    attribute_values = list(entropy_table.iloc[0]["Field Probabilities"].keys())
    print(f"{indent}Attribute Values: {attribute_values}")
    for value in attribute_values:
        print(f"{indent}--> Branching on '{best_attribute}' == '{value}'")
        # Filter original subset (current_df) based on the split
        filtered_data = filter_dataframe_by_value(current_df, best_attribute, value)
        # Recursive call
        branch_node = {}
        tree_node["branches"][value] = branch_node
        child_rule = f"{best_attribute}={value}"
        build_decision_tree_level(
            filtered_data,
            target_column,
            next_available_attributes,
            level + 1,
            max_depth,
            branch_node,
            rule=child_rule,
        )


# --- Main Execution Block --- (Replaces previous main block)
def main():
    """Main entry point for Poetry script."""
    # --- Setup ---
    filepath = "exams/2025/data_dec.csv"
    target_column = "לחץ דם"
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

            # --- Determine candidate root attributes with equal minimal entropy ---
            root_entropy_table = generate_entropy_table(df[initial_attributes + [target_column]], target_column)
            min_ce = root_entropy_table["Conditional Entropy"].min()
            EPS = 1e-6
            candidates = root_entropy_table[
                (root_entropy_table["Conditional Entropy"] - min_ce).abs() < EPS
            ]["Field Attribute"].tolist()

            best_depth = -1
            best_log = ""
            best_tree = {}

            for cand in candidates:
                tree_structure = {}
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    # Reorder attributes so cand appears first but others remain
                    reordered = [cand] + [a for a in initial_attributes if a != cand]
                    build_decision_tree_level(
                        df,
                        target_column,
                        reordered,
                        max_depth=max_recursion_depth,
                        tree_node=tree_structure,
                        forced_root_attr=cand,
                    )

                depth = _tree_max_depth(tree_structure)
                if depth > best_depth:
                    best_depth = depth
                    best_log = buffer.getvalue()
                    best_tree = tree_structure

            # ----- Print only the deepest tree logs -----
            print(best_log)
            print("\n" + "=" * 60)
            print("Decision tree analysis process finished.")

            print("\nFull Decision-Tree Structure (deepest):")
            _print_tree(best_tree)

if __name__ == "__main__":
    main()
