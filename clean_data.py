import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

plt.style.use("dark_background")
COLORS = ["#00ffcc","#ff6b6b","#ffd93d","#6bcff6","#c77dff","#ff9f43","#54a0ff","#5f27cd"]

df = pd.read_csv("Online-Store-Orders.csv")

print("=" * 60)
print("   ONLINE STORE — FULL DATA ANALYSIS")
print("=" * 60)

# ============================================================
# STEP 1 — Fix Missing CouponCode
# ============================================================
df["CouponCode"] = df["CouponCode"].fillna("NO COUPON")
print("\n✅ Step 1 — Missing CouponCodes filled")

# ============================================================
# STEP 2 — Fix Date Format
# ============================================================
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
print("✅ Step 2 — Date format fixed:", df["Date"].dtype)

# ============================================================
# STEP 3 — Remove Duplicates
# ============================================================
before = len(df)
df = df.drop_duplicates()
print(f"✅ Step 3 — Removed {before - len(df)} duplicates")

# ============================================================
# STEP 4 — Duplicate OrderID check
# ============================================================
dup_orders = df["OrderID"].duplicated().sum()
print(f"✅ Step 4 — Duplicate OrderIDs: {dup_orders}")

# ============================================================
# STEP 5 — Standardize Text Columns
# ============================================================
for col in ["Product","PaymentMethod","OrderStatus","ReferralSource"]:
    df[col] = df[col].str.strip().str.title()
print("✅ Step 5 — Text columns standardized")

# ============================================================
# STEP 6 — Validate TotalPrice
# ============================================================
df["Calculated_Total"] = df["Quantity"] * df["UnitPrice"]
df["Price_Mismatch"] = abs(df["TotalPrice"] - df["Calculated_Total"]) > 0.01
print(f"✅ Step 6 — Price mismatches: {df['Price_Mismatch'].sum()}")

# ============================================================
# STEP 7 — Feature Engineering
# ============================================================
month_order = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

df["Order_Value_Category"] = pd.cut(df["TotalPrice"],
    bins=[0,500,1500,3500], labels=["Low","Medium","High"])
df["Month"]      = df["Date"].dt.month_name()
df["Year"]       = df["Date"].dt.year
df["DayOfWeek"]  = df["Date"].dt.day_name()
df["WeekNum"]    = df["Date"].dt.isocalendar().week.astype(int)
df["YearMonth"]  = df["Date"].dt.to_period("M")
df["Data_Quality"] = np.where(df["CouponCode"] == "NO COUPON", "Partial", "Complete")

# Statistical outlier detection using IQR
Q1 = df["TotalPrice"].quantile(0.25)
Q3 = df["TotalPrice"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Order values can't be negative, so clip the lower bound at 0 for reporting/plotting.
# (We still use the raw lower_bound for the actual outlier test below — clipping it would
# silently hide legitimately low-value orders, which is the wrong fix.)
lower_bound_display = max(lower_bound, 0)
df["Is_Outlier"]         = (df["TotalPrice"] < lower_bound) | (df["TotalPrice"] > upper_bound)
df["Outlier_Direction"]  = np.where(df["TotalPrice"] > upper_bound, "High",
                           np.where(df["TotalPrice"] < lower_bound, "Low", "Normal"))
df["Z_Score"]            = np.abs(stats.zscore(df["TotalPrice"]))
df["Z_Score_Outlier"]    = df["Z_Score"] > 3

# Legacy suspicious flag kept for comparison
df["Suspicious_Order"] = (df["Quantity"] == 5) & (df["UnitPrice"] < 50)

print("✅ Step 7 — Feature engineering complete")

# ============================================================
# STEP 8 — Save Cleaned File
# ============================================================
export_cols = ["Calculated_Total","Price_Mismatch"]
df.drop(columns=export_cols, inplace=True)
df.to_csv("cleaned_orders.csv", index=False)
print("✅ Step 8 — cleaned_orders.csv saved")

# ============================================================
# DATA QUALITY REPORT
# ============================================================
print("\n" + "="*50)
print(" DATA QUALITY REPORT")
print("="*50)
print(f"Total Rows              : {len(df)}")
print(f"Missing Coupons Filled  : {(df['CouponCode'] == 'NO COUPON').sum()}")
print(f"Duplicate OrderIDs      : {dup_orders}")
print(f"IQR Outliers Detected   : {df['Is_Outlier'].sum()}")
print(f"Z-Score Outliers (>3σ)  : {df['Z_Score_Outlier'].sum()}")
print(f"Suspicious Orders       : {df['Suspicious_Order'].sum()}")
print(f"Data Quality — Complete : {(df['Data_Quality']=='Complete').sum()}")
print(f"Data Quality — Partial  : {(df['Data_Quality']=='Partial').sum()}")

# ============================================================
# KPI METRICS
# ============================================================
total_revenue    = df["TotalPrice"].sum()
avg_order        = df["TotalPrice"].mean()
median_order     = df["TotalPrice"].median()
highest_order    = df["TotalPrice"].max()
lowest_order     = df["TotalPrice"].min()
unique_customers = df["CustomerID"].nunique()
unique_products  = df["Product"].nunique()
top_product      = df.groupby("Product")["TotalPrice"].sum().idxmax()
top_payment      = df["PaymentMethod"].value_counts().idxmax()
top_referral     = df["ReferralSource"].value_counts().idxmax()
top_customer     = df.groupby("CustomerID")["TotalPrice"].sum().idxmax()

print("\n" + "="*50)
print(" KPI METRICS")
print("="*50)
print(f"Total Orders         : {len(df)}")
print(f"Unique Customers     : {unique_customers}")
print(f"Unique Products      : {unique_products}")
print(f"Total Revenue        : ₹{total_revenue:,.2f}")
print(f"Average Order Value  : ₹{avg_order:,.2f}")
print(f"Median Order Value   : ₹{median_order:,.2f}")
print(f"Highest Order Value  : ₹{highest_order:,.2f}")
print(f"Lowest Order Value   : ₹{lowest_order:,.2f}")
print(f"Top Product          : {top_product}")
print(f"Top Referral Source  : {top_referral}")
print(f"Top Payment Method   : {top_payment}")
print(f"Top Customer ID      : {top_customer}")
print(f"Date Range           : {df['Date'].min().date()} to {df['Date'].max().date()}")

# ============================================================
# CUSTOMER ANALYTICS
# ============================================================
customer_spend = df.groupby("CustomerID")["TotalPrice"].sum().sort_values(ascending=False)
customer_orders = df.groupby("CustomerID")["OrderID"].count()

print("\n" + "="*50)
print(" CUSTOMER ANALYSIS")
print("="*50)
print(f"Avg Customer Spend   : ₹{customer_spend.mean():,.2f}")
print(f"Highest Customer     : ₹{customer_spend.max():,.2f}")
print(f"Lowest Customer      : ₹{customer_spend.min():,.2f}")
top20_pct = customer_spend.head(int(len(customer_spend)*0.2)).sum() / total_revenue * 100
print(f"Top 20% Revenue Contr: {top20_pct:.1f}%")

# ============================================================
# PRODUCT ANALYTICS
# ============================================================
product_revenue = df.groupby("Product")["TotalPrice"].sum().sort_values(ascending=False)
print("\n" + "="*50)
print(" PRODUCT ANALYSIS")
print("="*50)
print(f"Best Product         : {product_revenue.idxmax()} — ₹{product_revenue.max():,.2f}")
print(f"Worst Product        : {product_revenue.idxmin()} — ₹{product_revenue.min():,.2f}")

# ============================================================
# COUPON ANALYSIS
# ============================================================
coupon_avg = df.groupby("CouponCode")["TotalPrice"].mean()
no_coupon_avg = coupon_avg.get("NO COUPON", 0)
print("\n" + "="*50)
print(" COUPON EFFECTIVENESS")
print("="*50)
for code, avg in coupon_avg.items():
    lift = ((avg - no_coupon_avg) / no_coupon_avg * 100) if no_coupon_avg > 0 and code != "NO COUPON" else 0
    tag = f"  +{lift:.1f}% lift" if lift > 0 else ("  baseline" if code == "NO COUPON" else f"  {lift:.1f}%")
    print(f"  {code:<12}: ₹{avg:,.2f} avg order{tag}")

# ============================================================
# FEATURE 1 — REPEAT VS NEW CUSTOMER SPLIT
# ============================================================
print("\n" + "="*50)
print(" REPEAT VS NEW CUSTOMER SPLIT")
print("="*50)
order_counts = df.groupby("CustomerID")["OrderID"].count()
repeat_customers = order_counts[order_counts > 1].index
new_customers    = order_counts[order_counts == 1].index

repeat_revenue = df[df["CustomerID"].isin(repeat_customers)]["TotalPrice"].sum()
new_revenue    = df[df["CustomerID"].isin(new_customers)]["TotalPrice"].sum()
repeat_pct     = repeat_revenue / total_revenue * 100
new_pct        = new_revenue / total_revenue * 100

print(f"Repeat Customers     : {len(repeat_customers)} ({len(repeat_customers)/unique_customers*100:.1f}%)")
print(f"New Customers        : {len(new_customers)} ({len(new_customers)/unique_customers*100:.1f}%)")
print(f"Repeat Revenue       : ₹{repeat_revenue:,.2f} ({repeat_pct:.1f}% of total)")
print(f"New Revenue          : ₹{new_revenue:,.2f} ({new_pct:.1f}% of total)")

# ============================================================
# FEATURE 2 — DAY-OF-WEEK ANALYSIS
# ============================================================
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow_orders   = df["DayOfWeek"].value_counts().reindex(day_order)
dow_revenue  = df.groupby("DayOfWeek")["TotalPrice"].sum().reindex(day_order)
best_day     = dow_revenue.idxmax()
print("\n" + "="*50)
print(" DAY-OF-WEEK ANALYSIS")
print("="*50)
print(f"Best Day (Revenue)   : {best_day} — ₹{dow_revenue.max():,.2f}")
for day in day_order:
    print(f"  {day:<12}: {dow_orders[day]:>4} orders  ₹{dow_revenue[day]:>10,.2f}")

# ============================================================
# FEATURE 3 — OUTLIER DETECTION REPORT
# ============================================================
print("\n" + "="*50)
print(" OUTLIER DETECTION")
print("="*50)
print(f"IQR Range            : ₹{lower_bound_display:,.2f} — ₹{upper_bound:,.2f}  (lower bound clipped at 0; raw stat = ₹{lower_bound:,.2f})")
print(f"IQR Outliers         : {df['Is_Outlier'].sum()} orders")
print(f"  High outliers      : {(df['Outlier_Direction']=='High').sum()}")
print(f"  Low outliers       : {(df['Outlier_Direction']=='Low').sum()}")
print(f"Z-Score Outliers     : {df['Z_Score_Outlier'].sum()} orders (|z|>3)")
print("\nTop 5 High-Value Outliers:")
print(df[df["Is_Outlier"] & (df["Outlier_Direction"]=="High")][
    ["OrderID","CustomerID","Product","TotalPrice","Z_Score"]
].sort_values("TotalPrice", ascending=False).head(5).to_string(index=False))

# ============================================================
# FEATURE 4 — RFM SEGMENTATION
# ============================================================
print("\n" + "="*50)
print(" RFM SEGMENTATION")
print("="*50)
snapshot_date = df["Date"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    Recency   = ("Date",       lambda x: (snapshot_date - x.max()).days),
    Frequency = ("OrderID",    "count"),
    Monetary  = ("TotalPrice", "sum")
).reset_index()

rfm["R_Score"] = pd.qcut(rfm["Recency"],   4, labels=[4,3,2,1]).astype(int)
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1,2,3,4]).astype(int)
rfm["M_Score"] = pd.qcut(rfm["Monetary"],  4, labels=[1,2,3,4]).astype(int)
rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

def rfm_segment(row):
    r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
    if r >= 4 and f >= 3 and m >= 3: return "Champion"
    elif r >= 3 and f >= 3:           return "Loyal"
    elif r >= 4 and f <= 2:           return "New Customer"
    elif r >= 3 and f <= 2:           return "Potential Loyalist"
    elif r == 2 and f >= 2:           return "At Risk"
    elif r == 1 and f >= 3:           return "Lost Champion"
    elif r == 1:                      return "Lost"
    else:                             return "Needs Attention"

rfm["Segment"] = rfm.apply(rfm_segment, axis=1)
seg_summary = rfm.groupby("Segment").agg(
    Customers = ("CustomerID","count"),
    Avg_Revenue = ("Monetary","mean")
).sort_values("Customers", ascending=False)

print(seg_summary.to_string())
rfm.to_csv("rfm_segments.csv", index=False)
print("\n✅ rfm_segments.csv saved")

# ============================================================
# FEATURE 5 — COHORT ANALYSIS
# ============================================================
print("\n" + "="*50)
print(" COHORT ANALYSIS")
print("="*50)
df["CohortMonth"] = df.groupby("CustomerID")["Date"].transform("min").dt.to_period("M")
df["OrderPeriod"] = df["Date"].dt.to_period("M")
df["CohortIndex"] = (df["OrderPeriod"] - df["CohortMonth"]).apply(lambda x: x.n)

cohort_data = df.groupby(["CohortMonth","CohortIndex"])["CustomerID"].nunique().reset_index()
cohort_pivot = cohort_data.pivot(index="CohortMonth", columns="CohortIndex", values="CustomerID")
cohort_size  = cohort_pivot.iloc[:,0]
cohort_pct   = cohort_pivot.divide(cohort_size, axis=0) * 100

print("Cohort Retention (%) — first 5 months shown:")
print(cohort_pct.iloc[:,:5].round(1).to_string())
cohort_pct.to_csv("cohort_retention.csv")
print("\n✅ cohort_retention.csv saved")

# ============================================================
# FEATURE 6 — TIME SERIES FORECASTING (Moving Average)
# ============================================================
print("\n" + "="*50)
print(" TIME SERIES FORECASTING")
print("="*50)
monthly_rev = df.groupby("YearMonth")["TotalPrice"].sum().reset_index()
monthly_rev["YearMonth_str"] = monthly_rev["YearMonth"].astype(str)
monthly_rev = monthly_rev.sort_values("YearMonth")
monthly_rev["MA_3"]  = monthly_rev["TotalPrice"].rolling(3).mean()
monthly_rev["MA_6"]  = monthly_rev["TotalPrice"].rolling(6).mean()

# Linear trend forecast for next 3 months
x = np.arange(len(monthly_rev))
y = monthly_rev["TotalPrice"].values
slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
forecast_x = np.arange(len(monthly_rev), len(monthly_rev)+3)
forecast_vals = intercept + slope * forecast_x
r_squared = r_val ** 2
# A linear trend with this low an R^2 is not a reliable forecast on its own —
# flag it instead of quietly presenting it as fact.
forecast_reliable = r_squared >= 0.5
print(f"Trend Slope          : ₹{slope:,.2f} per month")
print(f"R² (fit quality)     : {r_squared:.3f}" + ("" if forecast_reliable else "  ⚠️ LOW — trend explains little of the monthly variance"))
print(f"Next 3-month forecast (linear trend — see caveat above):")
for i, val in enumerate(forecast_vals, 1):
    print(f"  Month +{i}           : ₹{val:,.2f}")

# ============================================================
# PRODUCT PROFITABILITY RANKING
# ============================================================
print("\n" + "="*50)
print(" PRODUCT PROFITABILITY RANKING")
print("="*50)
assumed_margin = {"Laptop":0.18,"Monitor":0.15,"Phone":0.12,"Tablet":0.14,
                  "Chair":0.22,"Desk":0.20,"Headphones":0.25,"Keyboard":0.30}
prod_stats = df.groupby("Product").agg(
    Revenue=("TotalPrice","sum"), Orders=("OrderID","count"), AvgPrice=("UnitPrice","mean")
).reset_index()
prod_stats["Est_Margin"] = prod_stats["Product"].map(assumed_margin).fillna(0.20)
prod_stats["Est_Profit"] = prod_stats["Revenue"] * prod_stats["Est_Margin"]
prod_stats = prod_stats.sort_values("Est_Profit", ascending=False)
print(prod_stats[["Product","Revenue","Orders","Est_Margin","Est_Profit"]].to_string(index=False))

# ============================================================
# BUSINESS RECOMMENDATIONS
# ============================================================
print("\n" + "="*50)
print(" BUSINESS RECOMMENDATIONS")
print("="*50)
print(f"1. Focus on '{top_product}' — highest revenue product")
print(f"2. Scale '{top_referral}' referral channel — top source")
print(f"3. Promote '{top_payment}' payments — most preferred method")
print(f"4. Reward top 20% customers — they drive {top20_pct:.0f}% of revenue")
print(f"5. Review {df['Is_Outlier'].sum()} outlier orders for fraud/errors")
print(f"6. Retarget 'At Risk' + 'Lost' RFM segments with win-back campaign")
print(f"7. Best sales day is {best_day} — schedule promotions accordingly")
coupon_lift_codes = [c for c in coupon_avg.index if c != "NO COUPON" and coupon_avg[c] > no_coupon_avg]
if coupon_lift_codes:
    print(f"8. Expand coupon codes with positive lift: {', '.join(coupon_lift_codes)}")

# ============================================================
# ORIGINAL 11 CHARTS (Static PNG)
# ============================================================
print("\n📊 Generating static charts...")

def save_chart(fig, name, msg):
    fig.savefig(name)
    plt.close(fig)
    print(f"  ✅ {msg}")

fig, ax = plt.subplots(figsize=(8,5))
df["OrderStatus"].value_counts().plot(kind="bar", color=COLORS, ax=ax)
ax.set_title("Orders by Status", fontsize=16); ax.set_xlabel(""); ax.set_ylabel("Count")
plt.xticks(rotation=45); plt.tight_layout()
save_chart(fig, "chart1_order_status.png", "Chart 1 — Orders by Status")

fig, ax = plt.subplots(figsize=(8,5))
product_revenue.sort_values().plot(kind="barh", color="#00ffcc", ax=ax)
ax.set_title("Total Revenue by Product", fontsize=16)
plt.tight_layout(); save_chart(fig, "chart2_revenue_by_product.png", "Chart 2 — Revenue by Product")

fig, ax = plt.subplots(figsize=(7,7))
df["PaymentMethod"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=COLORS, ax=ax)
ax.set_title("Payment Method Distribution", fontsize=16); ax.set_ylabel("")
plt.tight_layout(); save_chart(fig, "chart3_payment_methods.png", "Chart 3 — Payment Methods")

fig, ax = plt.subplots(figsize=(10,5))
df["Month"].value_counts().reindex(month_order).plot(kind="line", marker="o", color="#00ffcc", ax=ax)
ax.set_title("Orders by Month", fontsize=16)
plt.xticks(rotation=45); plt.tight_layout()
save_chart(fig, "chart4_orders_by_month.png", "Chart 4 — Orders by Month")

fig, ax = plt.subplots(figsize=(7,5))
df["Order_Value_Category"].value_counts().plot(kind="bar", color=["#ffd93d","#00ffcc","#ff6b6b"], ax=ax)
ax.set_title("Order Value Categories", fontsize=16)
plt.xticks(rotation=0); plt.tight_layout()
save_chart(fig, "chart5_order_value_category.png", "Chart 5 — Order Value Category")

fig, ax = plt.subplots(figsize=(10,5))
customer_spend.head(10).plot(kind="bar", color="#ff6b6b", ax=ax)
ax.set_title("Top 10 Customers by Spending", fontsize=16)
plt.xticks(rotation=45); plt.tight_layout()
save_chart(fig, "chart6_top_customers.png", "Chart 6 — Top 10 Customers")

fig, ax = plt.subplots(figsize=(7,5))
df.groupby("CouponCode")["TotalPrice"].sum().sort_values(ascending=False).head(6).plot(kind="bar", color="#ffd93d", ax=ax)
ax.set_title("Revenue by Coupon Code", fontsize=16)
plt.xticks(rotation=45); plt.tight_layout()
save_chart(fig, "chart7_coupon_revenue.png", "Chart 7 — Coupon Revenue")

fig, ax = plt.subplots(figsize=(8,5))
df.groupby("ReferralSource")["TotalPrice"].sum().sort_values().plot(kind="barh", color="#c77dff", ax=ax)
ax.set_title("Revenue by Referral Source", fontsize=16)
plt.tight_layout(); save_chart(fig, "chart8_referral_revenue.png", "Chart 8 — Referral Revenue")

fig, ax = plt.subplots(figsize=(10,5))
monthly_rev.set_index("YearMonth_str")["TotalPrice"].plot(kind="line", marker="o", color="#00ffcc", ax=ax, label="Actual")
monthly_rev.set_index("YearMonth_str")["MA_3"].plot(ax=ax, color="#ffd93d", linestyle="--", label="3-Month MA")
monthly_rev.set_index("YearMonth_str")["MA_6"].plot(ax=ax, color="#ff6b6b", linestyle="--", label="6-Month MA")
ax.set_title("Monthly Revenue Trend + Moving Averages", fontsize=16)
ax.legend(); plt.xticks(rotation=45); plt.tight_layout()
save_chart(fig, "chart9_monthly_revenue.png", "Chart 9 — Monthly Revenue + MA")

fig, ax = plt.subplots(figsize=(8,5))
df.groupby("OrderStatus")["TotalPrice"].sum().sort_values().plot(kind="bar", color="#ff6b6b", ax=ax)
ax.set_title("Revenue by Order Status", fontsize=16)
plt.xticks(rotation=45); plt.tight_layout()
save_chart(fig, "chart10_status_revenue.png", "Chart 10 — Revenue by Status")

plt.figure(figsize=(8,5))
sns.heatmap(df[["Quantity","UnitPrice","TotalPrice"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap"); plt.tight_layout()
plt.savefig("chart11_correlation_heatmap.png"); plt.close()
print("  ✅ Chart 11 — Correlation Heatmap")

# ============================================================
# NEW CHART 12 — DAY-OF-WEEK ANALYSIS
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14,5))
dow_orders.plot(kind="bar", color="#6bcff6", ax=axes[0])
axes[0].set_title("Orders by Day of Week", fontsize=14)
axes[0].set_xlabel(""); axes[0].set_ylabel("Orders")
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
dow_revenue.plot(kind="bar", color="#c77dff", ax=axes[1])
axes[1].set_title("Revenue by Day of Week", fontsize=14)
axes[1].set_xlabel(""); axes[1].set_ylabel("Revenue (₹)")
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout(); plt.savefig("chart12_day_of_week.png"); plt.close()
print("  ✅ Chart 12 — Day of Week Analysis")

# ============================================================
# NEW CHART 13 — OUTLIER DETECTION
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14,5))
colors_map = df["Outlier_Direction"].map({"Normal":"#00ffcc","High":"#ff6b6b","Low":"#ffd93d"})
axes[0].scatter(df.index, df["TotalPrice"], c=colors_map, alpha=0.6, s=20)
axes[0].axhline(upper_bound, color="#ff6b6b", linestyle="--", label=f"Upper IQR ₹{upper_bound:,.0f}")
axes[0].axhline(lower_bound_display, color="#ffd93d", linestyle="--", label=f"Lower IQR ₹{lower_bound_display:,.0f}")
axes[0].set_title("Outlier Detection (IQR Method)", fontsize=14)
axes[0].set_xlabel("Order Index"); axes[0].set_ylabel("TotalPrice (₹)")
patches = [mpatches.Patch(color="#ff6b6b", label="High Outlier"),
           mpatches.Patch(color="#00ffcc", label="Normal"),
           mpatches.Patch(color="#ffd93d", label="Low Outlier")]
axes[0].legend(handles=patches, fontsize=9)
df["Outlier_Direction"].value_counts().plot(kind="bar", color=["#00ffcc","#ff6b6b","#ffd93d"], ax=axes[1])
axes[1].set_title("Outlier Distribution", fontsize=14)
axes[1].set_xlabel("Category"); axes[1].set_ylabel("Count")
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0)
plt.tight_layout(); plt.savefig("chart13_outlier_detection.png"); plt.close()
print("  ✅ Chart 13 — Outlier Detection")

# ============================================================
# NEW CHART 14 — RFM SEGMENT DISTRIBUTION
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14,5))
seg_counts = rfm["Segment"].value_counts()
seg_counts.plot(kind="bar", color=COLORS[:len(seg_counts)], ax=axes[0])
axes[0].set_title("Customer Count by RFM Segment", fontsize=14)
axes[0].set_xlabel(""); axes[0].set_ylabel("Customers")
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
rfm.groupby("Segment")["Monetary"].mean().sort_values(ascending=False).plot(
    kind="bar", color="#ff9f43", ax=axes[1])
axes[1].set_title("Avg Revenue by RFM Segment", fontsize=14)
axes[1].set_xlabel(""); axes[1].set_ylabel("Avg Revenue (₹)")
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout(); plt.savefig("chart14_rfm_segments.png"); plt.close()
print("  ✅ Chart 14 — RFM Segments")

# ============================================================
# NEW CHART 15 — COHORT RETENTION HEATMAP
# ============================================================
plt.figure(figsize=(14,7))
cohort_display = cohort_pct.iloc[:,:8].copy()
cohort_display.index = cohort_display.index.astype(str)
sns.heatmap(cohort_display, annot=True, fmt=".0f", cmap="YlOrRd",
            linewidths=0.5, vmin=0, vmax=100,
            cbar_kws={"label":"Retention %"})
plt.title("Cohort Retention Analysis — Monthly Retention %", fontsize=16)
plt.xlabel("Months Since First Purchase")
plt.ylabel("Cohort (First Purchase Month)")
plt.tight_layout(); plt.savefig("chart15_cohort_retention.png"); plt.close()
print("  ✅ Chart 15 — Cohort Retention Heatmap")

# ============================================================
# NEW CHART 16 — REPEAT VS NEW CUSTOMER
# FIX: pie chart title was overlapping the slice label when one slice is tiny
# (e.g. repeat revenue ~1.5%). Fixed by adding top padding via tight_layout
# rect, increasing pctdistance so the label sits further from the edge, and
# bumping suptitle vs subplot title spacing so they never collide.
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12,5.5))
fig.suptitle("Repeat vs New Customers", fontsize=15, y=0.98)
axes[0].pie([len(repeat_customers), len(new_customers)],
    labels=["Repeat","New"], autopct="%1.1f%%", pctdistance=0.75,
    colors=["#00ffcc","#ff6b6b"], startangle=90)
axes[0].set_title("Customer Type Split (by count)", fontsize=13, pad=14)
axes[1].pie([repeat_revenue, new_revenue],
    labels=["Repeat","New"], autopct="%1.1f%%", pctdistance=0.75,
    colors=["#00ffcc","#ff6b6b"], startangle=90)
axes[1].set_title("Revenue Split: Repeat vs New", fontsize=13, pad=14)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("chart16_repeat_vs_new.png"); plt.close()
print("  ✅ Chart 16 — Repeat vs New Customers (pie label collision fixed)")

# ============================================================
# NEW CHART 17 — COUPON EFFECTIVENESS
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12,5))
coupon_avg_df = df.groupby("CouponCode").agg(
    Avg_Order=("TotalPrice","mean"), Count=("OrderID","count")).reset_index()
colors_c = ["#ffd93d" if c == "NO COUPON" else "#00ffcc" for c in coupon_avg_df["CouponCode"]]
axes[0].bar(coupon_avg_df["CouponCode"], coupon_avg_df["Avg_Order"], color=colors_c)
axes[0].set_title("Avg Order Value by Coupon", fontsize=14)
axes[0].set_ylabel("Avg Order Value (₹)")
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30)
axes[1].bar(coupon_avg_df["CouponCode"], coupon_avg_df["Count"], color="#6bcff6")
axes[1].set_title("Order Count by Coupon", fontsize=14)
axes[1].set_ylabel("Number of Orders")
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30)
plt.tight_layout(); plt.savefig("chart17_coupon_effectiveness.png"); plt.close()
print("  ✅ Chart 17 — Coupon Effectiveness")

# ============================================================
# NEW CHART 18 — FORECAST
# ============================================================
fig, ax = plt.subplots(figsize=(12,5))
x_vals = np.arange(len(monthly_rev))
actual_y = monthly_rev["TotalPrice"].values
ax.plot(monthly_rev["YearMonth_str"], actual_y, marker="o", color="#00ffcc", linewidth=2, label="Actual Revenue")
trend_y = intercept + slope * x_vals
ax.plot(monthly_rev["YearMonth_str"], trend_y, color="#ffd93d", linestyle="--", linewidth=1.5, label="Trend Line")
forecast_labels = ["Forecast +1","Forecast +2","Forecast +3"]
ax.scatter(forecast_labels, forecast_vals, color="#ff6b6b", s=100, zorder=5, label="Forecast")
for lbl, val in zip(forecast_labels, forecast_vals):
    ax.annotate(f"₹{val:,.0f}", (lbl, val), textcoords="offset points",
                xytext=(0,10), ha="center", fontsize=9, color="#ff6b6b")
all_labels = list(monthly_rev["YearMonth_str"]) + forecast_labels
ax.set_xticks(range(len(all_labels)))
ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)
title_suffix = "" if forecast_reliable else f"  (R²={r_squared:.2f} — low confidence)"
ax.set_title(f"Monthly Revenue + 3-Month Forecast (Linear Trend){title_suffix}", fontsize=13)
ax.set_ylabel("Revenue (₹)"); ax.legend()
plt.tight_layout(); plt.savefig("chart18_forecast.png"); plt.close()
print("  ✅ Chart 18 — Forecast")

# ============================================================
# NEW CHART 19 — PRODUCT PROFITABILITY
# ============================================================
fig, ax = plt.subplots(figsize=(10,5))
prod_stats_sorted = prod_stats.sort_values("Est_Profit", ascending=True)
bars = ax.barh(prod_stats_sorted["Product"], prod_stats_sorted["Est_Profit"], color="#54a0ff")
ax.set_title("Estimated Profit by Product", fontsize=14)
ax.set_xlabel("Estimated Profit (₹)")
for bar, pct in zip(bars, prod_stats_sorted["Est_Margin"]):
    ax.text(bar.get_width()*0.5, bar.get_y()+bar.get_height()/2,
            f"{pct*100:.0f}% margin", va="center", ha="center", fontsize=9, color="black", fontweight="bold")
plt.tight_layout(); plt.savefig("chart19_product_profitability.png"); plt.close()
print("  ✅ Chart 19 — Product Profitability")

# ============================================================
# INTERACTIVE PLOTLY DASHBOARD
# FIX: dashboard pie chart (Repeat vs New Revenue, row 4 col 2) had the same
# title/label collision as chart16. Fixed with domain padding and a textinfo
# override so the percentage label doesn't sit flush against the subplot title.
# ============================================================
print("\n🌐 Building interactive HTML dashboard...")

fig_dash = make_subplots(
    rows=4, cols=2,
    subplot_titles=(
        "Revenue by Product",
        "Orders by Status",
        "Monthly Revenue Trend",
        "Payment Method Distribution",
        "RFM Segment — Customer Count",
        "Day of Week — Orders",
        "Coupon Avg Order Value",
        "Repeat vs New Revenue"
    ),
    specs=[
        [{"type":"bar"},     {"type":"bar"}],
        [{"type":"scatter"}, {"type":"pie"}],
        [{"type":"bar"},     {"type":"bar"}],
        [{"type":"bar"},     {"type":"pie"}]
    ],
    vertical_spacing=0.08
)

fig_dash.add_trace(go.Bar(x=product_revenue.index, y=product_revenue.values,
    marker_color="#00ffcc", name="Revenue"), row=1, col=1)
fig_dash.add_trace(go.Bar(x=df["OrderStatus"].value_counts().index,
    y=df["OrderStatus"].value_counts().values,
    marker_color="#ff6b6b", name="Orders"), row=1, col=2)
fig_dash.add_trace(go.Scatter(x=monthly_rev["YearMonth_str"], y=monthly_rev["TotalPrice"],
    mode="lines+markers", line=dict(color="#00ffcc", width=2),
    name="Monthly Revenue"), row=2, col=1)
fig_dash.add_trace(go.Scatter(x=monthly_rev["YearMonth_str"], y=monthly_rev["MA_3"],
    mode="lines", line=dict(color="#ffd93d", dash="dash"), name="3M MA"), row=2, col=1)
pm_counts = df["PaymentMethod"].value_counts()
fig_dash.add_trace(go.Pie(labels=pm_counts.index, values=pm_counts.values,
    name="Payment", textposition="inside"), row=2, col=2)
seg_c = rfm["Segment"].value_counts()
fig_dash.add_trace(go.Bar(x=seg_c.index, y=seg_c.values,
    marker_color="#c77dff", name="RFM Segments"), row=3, col=1)
fig_dash.add_trace(go.Bar(x=day_order, y=dow_orders.values,
    marker_color="#6bcff6", name="Day Orders"), row=3, col=2)
fig_dash.add_trace(go.Bar(x=coupon_avg_df["CouponCode"], y=coupon_avg_df["Avg_Order"],
    marker_color="#ffd93d", name="Coupon Avg"), row=4, col=1)
fig_dash.add_trace(go.Pie(labels=["Repeat","New"],
    values=[repeat_revenue, new_revenue], name="Customer Type",
    textposition="inside", textinfo="label+percent"), row=4, col=2)

fig_dash.update_layout(
    height=1500,
    title_text="Online Store — Interactive Analytics Dashboard",
    title_font_size=22,
    showlegend=False,
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#16213e",
    font=dict(color="white"),
    margin=dict(t=110)
)
fig_dash.write_html("dashboard.html")
print("  ✅ dashboard.html saved — open in any browser!")

# ============================================================
# GENERATE FULL REPORT
# ============================================================
with open("report.txt", "w", encoding="utf-8") as r:
    r.write("ONLINE STORE — FULL DATA ANALYSIS REPORT\n")
    r.write("=" * 60 + "\n\n")
    r.write("PROJECT SUMMARY\n" + "-"*40 + "\n")
    r.write(f"Total Orders         : {len(df)}\n")
    r.write(f"Unique Customers     : {unique_customers}\n")
    r.write(f"Unique Products      : {unique_products}\n")
    r.write(f"Total Revenue        : Rs.{total_revenue:,.2f}\n")
    r.write(f"Average Order Value  : Rs.{avg_order:,.2f}\n")
    r.write(f"Median Order Value   : Rs.{median_order:,.2f}\n")
    r.write(f"Highest Order Value  : Rs.{highest_order:,.2f}\n")
    r.write(f"Lowest Order Value   : Rs.{lowest_order:,.2f}\n\n")

    # --- Key Insight summary, added up front so a reviewer sees the "so what"
    # before the raw numbers ---
    r.write("KEY INSIGHTS\n" + "-"*40 + "\n")
    r.write(
        f"'{top_product}' is the leading revenue driver, '{top_referral}' is the top\n"
        f"acquisition channel, and the top 20% of customers generate {top20_pct:.0f}% of total\n"
        f"revenue — concentration that justifies a loyalty/retention push. Repeat purchase\n"
        f"rate is very low ({len(repeat_customers)/unique_customers*100:.1f}% of customers, "
        f"{repeat_pct:.1f}% of revenue), which is the\n"
        f"single biggest growth lever available: most customers buy once and do not return.\n"
        f"{best_day} is the strongest day for revenue, useful for timing promotions.\n\n"
    )

    r.write("CUSTOMER ANALYSIS\n" + "-"*40 + "\n")
    r.write(f"Repeat Customers     : {len(repeat_customers)} ({len(repeat_customers)/unique_customers*100:.1f}%)\n")
    r.write(f"New Customers        : {len(new_customers)} ({len(new_customers)/unique_customers*100:.1f}%)\n")
    r.write(f"Repeat Revenue %     : {repeat_pct:.1f}%\n")
    r.write(f"New Revenue %        : {new_pct:.1f}%\n")
    r.write(f"Top 20% Contrib      : {top20_pct:.1f}% of total revenue\n\n")

    r.write("OUTLIER DETECTION\n" + "-"*40 + "\n")
    r.write(f"IQR Range            : Rs.{lower_bound_display:,.2f} - Rs.{upper_bound:,.2f}\n")
    r.write(f"  Note: raw lower IQR bound was Rs.{lower_bound:,.2f} (negative); clipped at 0\n")
    r.write(f"  since order values cannot be negative. Used for display only — the outlier\n")
    r.write(f"  flag below still uses the raw statistical bound.\n")
    r.write(f"IQR Outliers         : {df['Is_Outlier'].sum()}\n")
    r.write(f"Z-Score Outliers     : {df['Z_Score_Outlier'].sum()}\n\n")

    r.write("RFM SEGMENTS\n" + "-"*40 + "\n")
    r.write(seg_summary.to_string() + "\n\n")

    r.write("FORECASTING\n" + "-"*40 + "\n")
    r.write(f"Trend Slope          : Rs.{slope:,.2f}/month\n")
    r.write(f"R-squared            : {r_squared:.3f}\n")
    if not forecast_reliable:
        r.write(
            "  CAVEAT: R-squared is low, meaning the linear trend explains only a small\n"
            "  share of month-to-month variation. Treat the forecast below as a rough,\n"
            "  low-confidence estimate, not a reliable prediction. Consider it more useful\n"
            "  for spotting the slope's direction (down) than for the exact rupee figures.\n"
        )
    for i, val in enumerate(forecast_vals, 1):
        r.write(f"Forecast Month +{i}    : Rs.{val:,.2f}\n")

    r.write("\nMETHODOLOGY & ASSUMPTIONS\n" + "-"*40 + "\n")
    r.write(
        "- Repeat customer = placed more than 1 order in the dataset's full date range\n"
        "  (no rolling time window applied).\n"
        "- Outliers = IQR method (1.5x IQR) on TotalPrice; Z-score method (>3 sigma) shown\n"
        "  alongside for comparison, since the two methods can disagree on skewed data.\n"
        "- RFM scores use quartile-based binning (qcut) on Recency, Frequency, Monetary,\n"
        "  so thresholds are dataset-relative, not fixed business rules.\n"
        "- Forecast uses simple linear regression on monthly revenue totals; no\n"
        "  seasonality component, so caution applies on a low R-squared (see above).\n"
        "- Profit margins per product are assumed estimates (not actual cost data) and\n"
        "  should be replaced with real margin figures if available.\n\n"
    )

    r.write("BUSINESS RECOMMENDATIONS\n" + "-"*40 + "\n")
    r.write(f"1. Focus on '{top_product}' — highest revenue product\n")
    r.write(f"2. Scale '{top_referral}' referral channel\n")
    r.write(f"3. Best sales day is {best_day} — schedule promotions\n")
    r.write(f"4. Retarget At-Risk and Lost RFM segments\n")
    r.write(f"5. Review {df['Is_Outlier'].sum()} outlier orders\n")
    r.write(f"6. Reward top 20% customers — they drive {top20_pct:.0f}% revenue\n")
    r.write(f"7. Investigate why repeat purchase rate is only {len(repeat_customers)/unique_customers*100:.1f}% —\n")
    r.write(f"   this is the largest single opportunity area in the dataset\n")

    r.write("\nCHARTS GENERATED\n" + "-"*40 + "\n")
    charts = [f"chart{i}_{n}.png" for i,n in enumerate([
        "order_status","revenue_by_product","payment_methods","orders_by_month",
        "order_value_category","top_customers","coupon_revenue","referral_revenue",
        "monthly_revenue","status_revenue","correlation_heatmap","day_of_week",
        "outlier_detection","rfm_segments","cohort_retention","repeat_vs_new",
        "coupon_effectiveness","forecast","product_profitability"],1)]
    for c in charts:
        r.write(f"  {c}\n")
    r.write("  dashboard.html (Interactive)\n")

print("\n✅ report.txt saved")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("        FINAL SUMMARY")
print("="*60)
print(f"📦 Total Orders         : {len(df)}")
print(f"👥 Unique Customers     : {unique_customers}")
print(f"🛍️  Unique Products      : {unique_products}")
print(f"💰 Total Revenue        : ₹{total_revenue:,.2f}")
print(f"📊 Avg Order Value      : ₹{avg_order:,.2f}")
print(f"🏆 Top Product          : {top_product}")
print(f"📢 Top Referral         : {top_referral}")
print(f"💳 Top Payment          : {top_payment}")
print(f"🔁 Repeat Customers     : {len(repeat_customers)}")
print(f"📈 Revenue Trend        : ₹{slope:+,.0f}/month")
print(f"⚠️  Outliers Detected    : {df['Is_Outlier'].sum()}")
print(f"🎯 RFM Champions        : {(rfm['Segment']=='Champion').sum()}")
print(f"📅 Best Sales Day       : {best_day}")
print(f"📅 Date Range           : {df['Date'].min().date()} to {df['Date'].max().date()}")
print("="*60)
print("\n✅ All done! Files saved:")
print("   cleaned_orders.csv | rfm_segments.csv | cohort_retention.csv")
print("   report.txt | dashboard.html | 19 PNG charts")