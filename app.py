import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Add the current directory and src to Python path
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

try:
    from src.scaling_laws import ALL_SCALING_LAWS
except ImportError:
    # Fallback import method
    import scaling_laws
    ALL_SCALING_LAWS = scaling_laws.ALL_SCALING_LAWS

st.set_page_config(
    page_title="Scaling Laws", page_icon="📊", layout="wide"
)

st.title("Scaling Laws Explorer")
st.markdown(
    "An intuitive tool for understanding and interpreting scaling laws for natural language AI models"
)

# Create tabs for the two main features
tab1, tab2 = st.tabs(
    ["📊 Compute Budget Visualization", "📈 Fit Your Own Scaling Law"]
)

# Tab 1: Compute Budget Visualization
with tab1:
    st.header("Visualize Scaling Laws from Notable Papers")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Parameters")

        # Required inputs
        input_mode = st.radio(
            "Choose input mode:", ["Compute Budget (FLOPs)", "Target Loss"]
        )

        if input_mode == "Compute Budget (FLOPs)":
            compute_budget = st.number_input(
                "Training Compute Budget C (FLOPs)",
                min_value=1e10,
                max_value=1e30,
                value=1e20,
                format="%.2e",
                help="Enter the compute budget in FLOPs",
            )
            target_loss = None
        else:
            target_loss = st.number_input(
                "Target Loss L",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help="Enter the target loss value",
            )
            compute_budget = None

        # Optional inputs
        with st.expander("Advanced Options"):
            inference_tokens = st.number_input(
                "Inference Compute Budget (test-time tokens D_test)",
                min_value=0,
                value=0,
                format="%d",
                help="Optional: Number of tokens for inference",
            )

            language = st.selectbox(
                "Language",
                ["English", "Spanish", "French", "German", "Chinese", "Japanese"],
                index=0,
                help="Select the language for scaling law recommendations",
            )

            total_tokens = st.number_input(
                "Total Training Tokens (for data constraints)",
                min_value=0,
                value=0,
                format="%d",
                help="Optional: Total unique tokens in training set",
            )

            mfu_inference = st.slider(
                "Model FLOP Utilization at Inference",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help="Model FLOP Utilization (MFU) for inference optimization",
            )

        # Select which scaling laws to display
        st.subheader("Select Scaling Laws")
        available_laws = list(ALL_SCALING_LAWS.keys())
        selected_laws = st.multiselect(
            "Choose scaling laws to compare:",
            available_laws,
            default=available_laws[:2] if len(available_laws) >= 2 else available_laws,
        )

        # Calculate button
        calculate_clicked = st.button(
            "Calculate Optimal Allocation", type="primary", use_container_width=True
        )

    with col2:
        if calculate_clicked and selected_laws:
            st.subheader("Results")

            # Calculate optimal allocations for each selected law
            results = []

            for law_name in selected_laws:
                law_wrapper = ALL_SCALING_LAWS[law_name]
                scaling_law = law_wrapper.scaling_law

                try:
                    if compute_budget is not None:
                        # Use compute budget
                        if inference_tokens > 0:
                            # Handle data-constrained case for inference
                            if "U" in law_wrapper.extra_args and total_tokens > 0:
                                result = scaling_law.compute_optimal_allocation_inference(
                                    compute_budget=compute_budget,
                                    D_inference=inference_tokens,
                                    mfu=mfu_inference,
                                    U=total_tokens,
                                )
                            elif "U" in law_wrapper.extra_args:
                                # Skip data-constrained law if no unique tokens provided
                                continue
                            else:
                                result = scaling_law.compute_optimal_allocation_inference(
                                    compute_budget=compute_budget,
                                    D_inference=inference_tokens,
                                    mfu=mfu_inference,
                                )
                        else:
                            # Handle data-constrained case for basic allocation
                            if "U" in law_wrapper.extra_args and total_tokens > 0:
                                result = scaling_law.compute_optimal_allocation(
                                    C=compute_budget, U=total_tokens
                                )
                            elif "U" in law_wrapper.extra_args:
                                # Skip data-constrained law if no unique tokens provided
                                st.info(f"Skipping {law_name}: requires 'Total Training Tokens' to be specified")
                                continue
                            else:
                                result = scaling_law.compute_optimal_allocation(
                                    C=compute_budget
                                )

                        results.append(
                            {
                                "Scaling Law": law_name,
                                "Optimal N*": f"{result['model']:.2e}",
                                "Optimal D*": f"{result['data']:.2e}",
                                "Loss": f"{result['loss']:.4f}",
                                "Paper": law_wrapper.paper,
                            }
                        )
                    else:
                        # Use target loss - need to implement this
                        st.warning(
                            f"Target loss mode not yet implemented for {law_name}"
                        )

                except Exception as e:
                    st.error(f"Error calculating {law_name}: {str(e)}")

            # Display results table
            if results:
                st.markdown("### Optimal Model Size (N*) and Training Tokens (D*)")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)

                # Create plots
                st.markdown("### Visualizations")

                # Plot 1: Optimal scaling lines
                fig1, ax1 = plt.subplots(figsize=(8, 6))

                # Generate range of compute budgets for optimal scaling curves
                compute_range = np.logspace(15, 25, 100)
                
                for law_name in selected_laws:
                    law_wrapper = ALL_SCALING_LAWS[law_name]
                    scaling_law = law_wrapper.scaling_law

                    # Skip data-constrained laws if no unique tokens provided
                    if "U" in law_wrapper.extra_args and total_tokens == 0:
                        continue

                    # Plot optimal scaling curve: optimal (N*, D*) for different compute budgets
                    N_optimal = []
                    D_optimal = []
                    
                    for C in compute_range:
                        try:
                            if "U" in law_wrapper.extra_args and total_tokens > 0:
                                result = scaling_law.compute_optimal_allocation(C=C, U=total_tokens)
                            else:
                                result = scaling_law.compute_optimal_allocation(C=C)
                            N_optimal.append(result['model'])
                            D_optimal.append(result['data'])
                        except Exception:
                            N_optimal.append(np.nan)
                            D_optimal.append(np.nan)
                    
                    ax1.loglog(D_optimal, N_optimal, label=f"{law_name} (optimal)", linewidth=2)

                # Add constraint line and intersection points
                if compute_budget:
                    # Add iso-compute constraint line
                    N_constraint = np.logspace(6, 12, 100)
                    D_constraint = [compute_budget / (6.0 * N) for N in N_constraint]
                    ax1.loglog(D_constraint, N_constraint, 
                             'k--', alpha=0.7, linewidth=1, 
                             label=f'Compute Budget = {compute_budget:.1e} FLOPs')
                    
                    # Mark intersection points (current optimal allocations)
                    for law_name in selected_laws:
                        law_wrapper = ALL_SCALING_LAWS[law_name]
                        scaling_law = law_wrapper.scaling_law
                        
                        # Skip data-constrained laws if no unique tokens provided
                        if "U" in law_wrapper.extra_args and total_tokens == 0:
                            continue
                            
                        try:
                            if "U" in law_wrapper.extra_args and total_tokens > 0:
                                result = scaling_law.compute_optimal_allocation(C=compute_budget, U=total_tokens)
                            else:
                                result = scaling_law.compute_optimal_allocation(C=compute_budget)
                            ax1.scatter(result['data'], result['model'], 
                                      s=100, alpha=0.8, zorder=5)
                        except Exception:
                            pass
                elif target_loss:
                    # For target loss mode, show iso-loss curves
                    for law_name in selected_laws:
                        law_wrapper = ALL_SCALING_LAWS[law_name]
                        scaling_law = law_wrapper.scaling_law
                        
                        # Skip data-constrained laws if no unique tokens provided
                        if "U" in law_wrapper.extra_args and total_tokens == 0:
                            continue
                        
                        # Plot iso-loss curve for target loss
                        N_isoloss = np.logspace(6, 12, 100)
                        D_isoloss = []
                        
                        for N in N_isoloss:
                            try:
                                if "U" in law_wrapper.extra_args and total_tokens > 0:
                                    D = scaling_law.N_to_D(N, target_loss, U=total_tokens)
                                else:
                                    D = scaling_law.N_to_D(N, target_loss)
                                D_isoloss.append(D)
                            except Exception:
                                D_isoloss.append(np.nan)
                        
                        ax1.loglog(D_isoloss, N_isoloss, 
                                 '--', alpha=0.7, linewidth=1,
                                 label=f'{law_name} iso-loss = {target_loss}')

                ax1.set_xlabel("Training Tokens (D)")
                ax1.set_ylabel("Model Size (N)")
                ax1.set_title("Optimal Scaling Lines")
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                st.pyplot(fig1)

                # Plot 2: Tokens per parameter ratio
                fig2, ax2 = plt.subplots(figsize=(8, 6))

                compute_budgets = np.logspace(15, 25, 100)

                for law_name in selected_laws:
                    law_wrapper = ALL_SCALING_LAWS[law_name]
                    scaling_law = law_wrapper.scaling_law

                    # Skip data-constrained laws if no unique tokens provided
                    if "U" in law_wrapper.extra_args and total_tokens == 0:
                        continue

                    ratios = []
                    for C in compute_budgets:
                        try:
                            if "U" in law_wrapper.extra_args and total_tokens > 0:
                                result = scaling_law.compute_optimal_allocation(C=C, U=total_tokens)
                            else:
                                result = scaling_law.compute_optimal_allocation(C=C)
                            ratio = result["data"] / result["model"]
                            ratios.append(ratio)
                        except Exception:
                            ratios.append(np.nan)

                    ax2.loglog(compute_budgets, ratios, label=law_name, linewidth=2)

                ax2.set_xlabel("Compute Budget (FLOPs)")
                ax2.set_ylabel("Tokens per Parameter Ratio (D/N)")
                ax2.set_title("Optimal Tokens per Parameter Ratio vs Compute Budget")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                st.pyplot(fig2)

                # Recommendations
                st.markdown("### Recommendations")
                st.info(
                    """
                **Key Insights:**
                - The scaling laws above are derived from specific papers
                - Consider your compute budget vs paper's tested range
                - Architecture and data quality affect optimal allocation
                - Multilingual models may scale differently

                **References:**
                """
                )

                for law_name in selected_laws:
                    law_wrapper = ALL_SCALING_LAWS[law_name]
                    st.markdown(
                        f"- **{law_name}**: [{law_wrapper.paper}]({law_wrapper.paper})"
                    )

# Tab 2: Fit Your Own Scaling Law
with tab2:
    st.header("Fit Your Own Scaling Law")
    st.info(
        "This feature is under development. Upload (N, D, Loss) data soon."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Upload Your Data")

        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file with [N, D, Loss] columns",
            type="csv",
            help="File should contain columns: N, D, Loss",
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} data points")

                # Show preview
                st.markdown("#### Data Preview")
                st.dataframe(df.head(), use_container_width=True)

                # Select scaling law type
                law_type = st.selectbox(
                    "Select Scaling Law Type",
                    ["Basic Scaling Law", "Data-Constrained Scaling Law"],
                    help="Choose the type of scaling law to fit",
                )

                # Additional columns for data-constrained
                if law_type == "Data-Constrained Scaling Law":
                    if "U" in df.columns:
                        st.success("Found column 'U' (unique tokens) in data")
                    else:
                        st.warning(
                            "Data-Constrained Law requires column 'U'"
                        )

                # Fit button
                fit_clicked = st.button(
                    "Fit Scaling Law", type="primary", use_container_width=True
                )

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        else:
            st.info("Please upload a CSV file with your scaling data")

    with col2:
        if uploaded_file is not None and fit_clicked:
            st.subheader("Fitting Results")
            st.warning(
                "Fitting functionality is not yet implemented. This will include:"
            )

            st.markdown(
                """
            **Coming Soon:**
            - Full fitted formula with parameters
            - R-squared and evaluation metrics
            - Cross-validation results
            - Optimal scaling line plot
            - Tokens per parameter ratio plot
            - Export functionality for fitted parameters
            """
            )

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit | Multilingual Scaling Laws Research Application")
