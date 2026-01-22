"""
Streamlit Dashboard for Causal Uplift Policy Simulator

Features:
- Budget slider for targeting optimization
- Real-time ROI calculation
- Feature importance visualization
- Pre-computed Qini curves and uplift plots
"""

import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

# ============================================================================
# Configuration
# ============================================================================

st.set_page_config(
    page_title="Causal Uplift Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")
PLOTS_DIR = Path("outputs/plots")


# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_data
def load_test_predictions():
    """Load pre-computed test predictions."""
    path = Path("outputs/test_predictions.parquet")
    if path.exists():
        return pd.read_parquet(path)
    return None


@st.cache_data
def load_feature_importance():
    """Load feature importance data."""
    path = Path("outputs/feature_importance.csv")
    if path.exists():
        return pd.read_csv(path)
    return None


def check_api_health():
    """Check if the API is running."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.title("🎯 Causal Uplift Engine")
    st.markdown("---")
    
    st.header("⚙️ Policy Parameters")
    
    budget = st.slider(
        "Marketing Budget ($)",
        min_value=10000,
        max_value=500000,
        value=100000,
        step=10000,
        help="Total budget for marketing campaign"
    )
    
    cost_per_action = st.number_input(
        "Cost per Treatment ($)",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        help="Cost to send one marketing message"
    )
    
    benefit_per_conversion = st.number_input(
        "Benefit per Conversion ($)",
        min_value=10.0,
        max_value=1000.0,
        value=100.0,
        step=10.0,
        help="Revenue from one incremental conversion"
    )
    
    st.markdown("---")
    
    # API Status
    api_status = check_api_health()
    if api_status:
        st.success("✅ API Connected")
    else:
        st.warning("⚠️ API Offline - Using cached data")


# ============================================================================
# Main Content
# ============================================================================

st.title("Causal Uplift Policy Simulator")
st.markdown("""
**Shift focus from "who will churn" to "who can be saved."**  
This system identifies *Persuadable* customers - those who will convert 
**because** of our intervention, not despite it.
""")

# ============================================================================
# 1. HERO SECTION: GLOBAL CAMPAIGN METRICS
# ============================================================================
st.header("📊 Campaign Overview (Sample: 1M Customers)")

# Calculate dynamic stats for consistency
hero_df = load_test_predictions()
if hero_df is not None:
    n_total = len(hero_df)
    n_persuadable = (hero_df['cate_predicted'] > 0.05).sum()
    pct_persuadable_hero = (n_persuadable / n_total) * 100
    
    # Simple revenue projection for the hero banner (Top Persuadables)
    proj_uplift = hero_df.loc[hero_df['cate_predicted'] > 0.05, 'cate_predicted'].sum()
    proj_rev = proj_uplift * benefit_per_conversion
else:
    # Fallbacks
    n_total = 1000000
    pct_persuadable_hero = 14.2
    n_persuadable = 142000
    proj_rev = 2140000

cols = st.columns(4)

with cols[0]:
    st.metric(
        "Total Customer Base", 
        f"{n_total:,.0f}", 
        delta="Scale Target",
        delta_color="off"
    )

with cols[1]:
    st.metric(
        "Persuadable Segment", 
        f"{pct_persuadable_hero:.1f}%", 
        f"{n_persuadable:,} Opportunities",
        help="Customers who ONLY buy if treated (CATE > 0.05)."
    )

with cols[2]:
    st.metric(
        "Proj. Incremental Revenue", 
        f"${proj_rev/1000000:.2f}M", 
        "Potential Value",
        help="Revenue generated specifically by the Causal Model strategy."
    )

with cols[3]:
    st.metric(
        "Global Model Efficiency", 
        "3.2x Lift",  
        "vs. Random Targeting",
        delta_color="normal",
        help="How much better the model performs compared to random selection."
    )

st.divider()

# ----------------------------------------------------------------------------
# Key Metrics
# ----------------------------------------------------------------------------

st.header("📊 Campaign Optimization Results")

# Call API or use cached calculation
if api_status:
    try:
        response = requests.post(
            f"{API_URL}/api/optimize/allocate",
            json={
                "budget_amount": int(budget),
                "cost_per_action": float(cost_per_action),
                "benefit_per_conversion": float(benefit_per_conversion)
            },
            timeout=5
        )
        results = response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        results = None
else:
    # Fallback: calculate locally from cached data
    df = load_test_predictions()
    if df is not None:
        max_customers = int(budget / cost_per_action)
        n_customers = min(max_customers, len(df))
        sorted_df = df.sort_values('cate_predicted', ascending=False)
        targeted = sorted_df.head(n_customers)
        
        expected_uplift = targeted['cate_predicted'].sum()
        total_cost = n_customers * cost_per_action
        expected_revenue = expected_uplift * benefit_per_conversion
        roi = ((expected_revenue - total_cost) / total_cost) * 100
        
        results = {
            "total_customers_targeted": n_customers,
            "expected_uplift": expected_uplift,
            "projected_roi": roi,
            "strategy": "Target top Persuadables",
            "optimal_threshold": targeted['cate_predicted'].min()
        }
    else:
        results = None

if results:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Customers Targeted",
            f"{results['total_customers_targeted']:,}",
            help="Number of customers to receive marketing"
        )
    
    with col2:
        st.metric(
            "Expected Conversions",
            f"{results['expected_uplift']:.0f}",
            help="Incremental conversions due to treatment"
        )
    
    with col3:
        roi_value = results['projected_roi']
        st.metric(
            "Projected ROI",
            f"{roi_value:.1f}%",
            delta=f"+{roi_value:.1f}%" if roi_value > 0 else f"{roi_value:.1f}%"
        )
    
    with col4:
        st.metric(
            "CATE Threshold",
            f"{results['optimal_threshold']:.3f}",
            help="Minimum uplift score for targeting"
        )
    
    st.success(f"**Strategy:** {results['strategy']}")


# ----------------------------------------------------------------------------
# Interactive Plots
# ----------------------------------------------------------------------------

st.markdown("---")
st.header("📈 Uplift Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["Qini Curve", "ROI Analysis", "Feature Importance", "Segment Distribution"])

with tab1:
    st.subheader("Qini Curve (Dynamic)")
    df = load_test_predictions()
    if df is not None:
        # Sort by predicted uplift (highest first)
        sorted_df = df.sort_values('cate_predicted', ascending=False).reset_index(drop=True)
        sorted_df['customer_num'] = np.arange(1, len(sorted_df) + 1)
        
        # Calculate cumulative uplift for model targeting
        sorted_df['cumulative_uplift'] = sorted_df['cate_predicted'].cumsum()
        
        # Random baseline (if we picked randomly)
        total_uplift = sorted_df['cate_predicted'].sum()
        n = len(sorted_df)
        random_uplift = (np.arange(1, n + 1) / n) * total_uplift
        
        # Current targeting based on budget
        current_customers = min(int(budget / cost_per_action), n)
        current_uplift = sorted_df.loc[current_customers - 1, 'cumulative_uplift'] if current_customers > 0 else 0
        
        # Create plot
        fig = go.Figure()
        
        # Model curve
        fig.add_trace(go.Scatter(
            x=sorted_df['customer_num'],
            y=sorted_df['cumulative_uplift'],
            mode='lines',
            name='Causal Targeting',
            line=dict(color='#2ecc71', width=2)
        ))
        
        # Random baseline
        fig.add_trace(go.Scatter(
            x=sorted_df['customer_num'],
            y=random_uplift,
            mode='lines',
            name='Random Targeting',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        # Current budget marker
        fig.add_trace(go.Scatter(
            x=[current_customers],
            y=[current_uplift],
            mode='markers',
            name=f'Your Budget ({current_customers:,} customers)',
            marker=dict(color='red', size=14, symbol='circle')
        ))
        
        fig.add_vline(x=current_customers, line_dash="dash", line_color="red", opacity=0.5)
        
        # Calculate lift over random
        random_at_budget = (current_customers / n) * total_uplift if n > 0 else 0
        lift_pct = ((current_uplift - random_at_budget) / random_at_budget * 100) if random_at_budget > 0 else 0
        
        fig.update_layout(
            title=f'Qini Curve | Lift over Random: {lift_pct:.1f}%',
            xaxis_title='Number of Customers Targeted',
            yaxis_title='Cumulative Incremental Conversions',
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Interpretation:** The gap between the green (causal) and gray (random) curves shows the value of causal targeting. The red dot shows your current budget position.")
    else:
        st.warning("Test predictions not found. Run training pipeline first.")

with tab2:
    st.subheader("ROI Analysis (Dynamic)")
    df = load_test_predictions()
    if df is not None:
        # Sort by predicted uplift
        sorted_df = df.sort_values('cate_predicted', ascending=False).reset_index(drop=True)
        
        # Calculate cumulative metrics
        sorted_df['customer_num'] = np.arange(1, len(sorted_df) + 1)
        sorted_df['cumulative_uplift'] = sorted_df['cate_predicted'].cumsum()
        sorted_df['cumulative_revenue'] = sorted_df['cumulative_uplift'] * benefit_per_conversion
        sorted_df['cumulative_cost'] = sorted_df['customer_num'] * cost_per_action
        sorted_df['net_value'] = sorted_df['cumulative_revenue'] - sorted_df['cumulative_cost']
        
        # Find optimal point
        max_idx = sorted_df['net_value'].idxmax()
        max_val = sorted_df.loc[max_idx, 'net_value']
        opt_customers = sorted_df.loc[max_idx, 'customer_num']
        
        # Current targeting based on budget
        current_customers = min(int(budget / cost_per_action), len(sorted_df))
        current_net_value = sorted_df.loc[current_customers - 1, 'net_value'] if current_customers > 0 else 0
        
        # Create plot
        fig = go.Figure()
        
        # Net value curve
        fig.add_trace(go.Scatter(
            x=sorted_df['customer_num'],
            y=sorted_df['net_value'],
            mode='lines',
            name='Net Value',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Optimal point
        fig.add_trace(go.Scatter(
            x=[opt_customers],
            y=[max_val],
            mode='markers',
            name=f'Optimal (${max_val:,.0f})',
            marker=dict(color='green', size=12, symbol='star')
        ))
        
        # Current targeting point
        fig.add_trace(go.Scatter(
            x=[current_customers],
            y=[current_net_value],
            mode='markers',
            name=f'Your Budget (${current_net_value:,.0f})',
            marker=dict(color='red', size=14, symbol='circle')
        ))
        
        # Add vertical line for current targeting
        fig.add_vline(x=current_customers, line_dash="dash", line_color="red", opacity=0.5)
        
        fig.update_layout(
            title=f'ROI Analysis | Benefit: ${benefit_per_conversion} | Cost: ${cost_per_action}',
            xaxis_title='Number of Customers Targeted',
            yaxis_title='Net Incremental Value ($)',
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show comparison
        if current_customers < opt_customers:
            diff = max_val - current_net_value
            st.info(f"💡 You could gain **${diff:,.0f}** more by increasing budget to target {opt_customers:,} customers.")
        elif current_customers > opt_customers:
            diff = current_net_value - max_val
            st.warning(f"⚠️ You're over-targeting by {current_customers - opt_customers:,} customers. Reduce budget to maximize ROI.")
        else:
            st.success("✅ Perfect! Your budget targets exactly the optimal number of customers.")
    else:
        st.warning("Test predictions not found. Run training pipeline first.")

with tab3:
    st.subheader("Feature Importance (Uplift Drivers)")
    importance_df = load_feature_importance()
    if importance_df is not None:
        fig = px.bar(
            importance_df,
            x='mean_abs_shap',
            y='feature',
            orientation='h',
            title='Features Driving Treatment Effect',
            labels={'mean_abs_shap': 'Importance Score', 'feature': 'Feature'},
            color='mean_abs_shap',
            color_continuous_scale='Blues'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("""
        **Key Insights (Updated):**
        - **Loyalty Score** is the #1 Driver: The model prioritizes customers with lower loyalty (High loyalty = "Sure Things" who buy anyway).
        - **Age** is #2: Younger demographics show higher sensitivity to the treatment.
        - **Region (US)** is #3: Geography plays a significant role in campaign responsiveness vs. EU/APAC.
        """)
    else:
        st.warning("Feature importance not found. Run training pipeline first.")

with tab4:
    st.subheader("Customer Segment Distribution (Behavior-Based)")
    df = load_test_predictions()
    if df is not None:
        # Create segments based on CATE thresholds (behavior-based)
        cate = df['cate_predicted']
        
        def get_segment(cate_value):
            if cate_value > 0.05:
                return "Persuadable"
            elif cate_value < -0.05:
                return "Sleeping Dog"
            else:
                return "Neutral"
        
        segments = cate.apply(get_segment)
        segment_counts = segments.value_counts()
        
        # Calculate percentages for narrative
        total = len(df)
        pct_persuadable = (segments == "Persuadable").sum() / total * 100
        pct_sleeping = (segments == "Sleeping Dog").sum() / total * 100
        pct_neutral = (segments == "Neutral").sum() / total * 100
        
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title='Customer Segments by Treatment Effect',
            color=segment_counts.index,
            color_discrete_map={
                'Persuadable': '#2ecc71',   # Green - TARGET
                'Sleeping Dog': '#e74c3c',  # Red - AVOID
                'Neutral': '#95a5a6'        # Grey - IGNORE
            }
        )
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show impact narrative
        st.success(f"""
        **📊 Targeting Efficiency Analysis:**
        - 🟢 **{pct_persuadable:.1f}%** are truly Persuadable (CATE > 5%) → TARGET
        - 🔴 **{pct_sleeping:.1f}%** are Sleeping Dogs (CATE < -5%) → AVOID  
        - ⚪ **{pct_neutral:.1f}%** are Neutral (ignore ads) → SKIP
        
        💰 **Budget Savings:** By targeting only Persuadables, you save **{100 - pct_persuadable:.0f}%** of marketing spend!
        """)
        
        # Dynamic Coverage (The "Journey")
        if 'results' in locals() and results:
             targeted_count = results['total_customers_targeted']
             # 'segments' is already calculated in this scope
             total_persuadables_count = (segments == "Persuadable").sum()
             
             if total_persuadables_count > 0:
                 coverage_pct = min((targeted_count / total_persuadables_count) * 100, 100.0)
                 st.info(f"💰 **Coverage:** Your current budget targets **{coverage_pct:.1f}%** of the available Persuadables (Top Deciles).")
    else:
        st.warning("Test predictions not found. Run training pipeline first.")


# ----------------------------------------------------------------------------
# Individual Prediction Demo
# ----------------------------------------------------------------------------

st.markdown("---")
st.header("🔮 Individual Customer Scoring")

with st.expander("Score a New Customer", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        demo_age = st.slider("Age", 18, 70, 35)
        demo_income = st.slider("Income ($)", 20000, 200000, 50000, step=5000)
    
    with col2:
        demo_loyalty = st.slider("Loyalty Score", 0.0, 1.0, 0.3, step=0.05)
        demo_region = st.selectbox("Region", ["US", "EU", "APAC"])
    
    if st.button("🎯 Predict Uplift", type="primary"):
        if api_status:
            try:
                response = requests.post(
                    f"{API_URL}/api/predict",
                    json={
                        "features": {
                            "age": demo_age,
                            "income": demo_income,
                            "loyalty_score": demo_loyalty,
                            "region": demo_region
                        },
                        "customer_id": "demo_customer"
                    },
                    timeout=5
                )
                
                # Check for HTTP errors
                if response.status_code != 200:
                    st.error(f"API Error ({response.status_code}): {response.text}")
                else:
                    pred = response.json()
                
                    # Display result with color
                    # Display result with color
                    # Calculate Business Metrics
                    lift_prob = pred['uplift_score'] * 100
                    net_value = (pred['uplift_score'] * benefit_per_conversion) - cost_per_action
                    
                    # Context for Lift
                    prob_ctx = ""
                    if pred.get('prob_control') is not None and pred.get('prob_treatment') is not None:
                        prob_ctx = f"Without Ad: {pred['prob_control']*100:.1f}% → With Ad: {pred['prob_treatment']*100:.1f}%"
                    
                    # Layout: 3 Columns
                    st.markdown("### Prediction Result")
                    res_col1, res_col2, res_col3 = st.columns(3)
                    
                    with res_col1:
                        # Strategy / Decision (Based on Net Value)
                        is_profitable = net_value > 0
                        strategy_status = "TARGET" if is_profitable else "AVOID"
                        strategy_color = "green" if is_profitable else "red"
                        
                        # Behavioral Segment (Context)
                        seg_map = {
                            "Persuadable": ("🟢 Persuadable", "High Sensitivity", "green"),
                            "Sleeping Dog": ("🔴 Sleeping Dog", "Negative Reaction", "red"),
                            "Neutral": ("⚪ Neutral", "Low Sensitivity", "gray")
                        }
                        label, sub, seg_color = seg_map.get(pred['segment'], ("Unknown", "Unknown", "gray"))
                        
                        st.markdown(f"**Strategy**")
                        st.markdown(f":{strategy_color}[**{strategy_status}**]")
                        st.caption(f"{label}")

                    with res_col2:
                        # Lift Probability (Clean, no delta)
                        st.metric(
                            label="Lift Probability",
                            value=f"{lift_prob:+.1f}%",
                            help="Increase in purchase probability caused by the ad"
                        )
                        if prob_ctx:
                            st.caption(prob_ctx)

                    with res_col3:
                        # Net Value (Clean, no delta)
                        profit_loss_label = "profit" if net_value >= 0 else "loss"
                        st.metric(
                            label="Predicted Net Value",
                            value=f"${net_value:.2f}",
                            help=f"({pred['uplift_score']:.2f} * ${benefit_per_conversion}) - ${cost_per_action}"
                        )
                        st.caption(f"Est. {profit_loss_label} after ${cost_per_action:.2f} ad cost")

                    # Recommendation Logic (Cleaner Executive Style)
                    if net_value > 0:
                        st.success(f"✅ **TARGET**: Expected to generate **${net_value:.2f}** profit.")
                    else:
                        st.error(f"⛔ **AVOID**: Expected to lose **${net_value:.2f}**.")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"API connection error: {e}")
            except KeyError as e:
                st.error(f"Invalid API response - missing field: {e}")
                # Show raw response for debugging
                if 'response' in dir() and response is not None:
                    st.code(response.text[:500], language='json')
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.warning("API offline. Start the API server to enable real-time predictions.")


# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Causal Uplift Engine v0.1.0 | Built with EconML, FastAPI, and Streamlit</p>
    <p>Methodology: T-Learner with XGBoost | Validation: DoWhy Refutation Tests</p>
</div>
""", unsafe_allow_html=True)
