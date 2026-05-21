# ============================================================
#   UNEMPLOYMENT ANALYSIS — CodeAlpha Internship Task 2
#   Author: Your Name
#   Description: Analyze Unemployment trends in India
#                with focus on Covid-19 impact
# ============================================================

# ─────────────────────────────────────────
# STEP 1: Import Libraries
# ─────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   UNEMPLOYMENT ANALYSIS IN INDIA — CodeAlpha Task 2")
print("=" * 60)


# ─────────────────────────────────────────
# STEP 2: Load & Clean Dataset
# ─────────────────────────────────────────
df = pd.read_csv('Unemployment in India.csv')

# Strip extra spaces from column names
df.columns = df.columns.str.strip()

# Rename columns for easier use
df.rename(columns={
    'Estimated Unemployment Rate (%)' : 'Unemployment_Rate',
    'Estimated Employed'              : 'Employed',
    'Estimated Labour Participation Rate (%)' : 'Labour_Participation_Rate',
    'Date'                            : 'Date',
    'Frequency'                       : 'Frequency',
    'Region'                          : 'Region',
    'Area'                            : 'Area'
}, inplace=True)

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Sort by date
df = df.sort_values('Date').reset_index(drop=True)

print("\n📂 Dataset Loaded Successfully!")
print(f"   Shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Period: {df['Date'].min().strftime('%b %Y')} → {df['Date'].max().strftime('%b %Y')}")


# ─────────────────────────────────────────
# STEP 3: Explore the Data
# ─────────────────────────────────────────
print("\n─── First 5 Rows ───────────────────────────────────────")
print(df.head())

print("\n─── Missing Values ─────────────────────────────────────")
print(df.isnull().sum())

print("\n─── Statistical Summary ────────────────────────────────")
print(df[['Unemployment_Rate', 'Employed', 'Labour_Participation_Rate']].describe())

print("\n─── Regions in Dataset ─────────────────────────────────")
print(df['Region'].unique())


# ─────────────────────────────────────────
# STEP 4: Overall Unemployment Trend Over Time
# ─────────────────────────────────────────
print("\n📊 Generating Visualizations...")

monthly_avg = df.groupby('Date')['Unemployment_Rate'].mean().reset_index()

plt.figure(figsize=(14, 6))
plt.plot(monthly_avg['Date'], monthly_avg['Unemployment_Rate'],
         color='#2196F3', linewidth=2.5, marker='o', markersize=4, label='Avg Unemployment Rate')

# Highlight Covid-19 period
covid_start = pd.to_datetime('2020-03-01')
covid_end   = pd.to_datetime('2020-12-31')
plt.axvspan(covid_start, covid_end, alpha=0.15, color='red', label='Covid-19 Period')
plt.axvline(covid_start, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

plt.title("Overall Unemployment Rate Trend in India\n(with Covid-19 Impact Highlighted)",
          fontsize=14, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("unemployment_trend.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: unemployment_trend.png")


# ─────────────────────────────────────────
# STEP 5: Covid-19 Impact Analysis
# ─────────────────────────────────────────
pre_covid  = df[df['Date'] < '2020-03-01']['Unemployment_Rate'].mean()
during_covid = df[(df['Date'] >= '2020-03-01') & (df['Date'] <= '2020-12-31')]['Unemployment_Rate'].mean()
post_covid = df[df['Date'] > '2020-12-31']['Unemployment_Rate'].mean()

print("\n─── Covid-19 Impact ────────────────────────────────────")
print(f"   Pre-Covid  Avg Unemployment Rate : {pre_covid:.2f}%")
print(f"   During Covid Avg Unemployment Rate: {during_covid:.2f}%")
print(f"   Post-Covid Avg Unemployment Rate  : {post_covid:.2f}%")
print(f"   📈 Increase during Covid          : +{during_covid - pre_covid:.2f}%")

# Bar chart — Covid impact
fig, ax = plt.subplots(figsize=(8, 6))
periods = ['Pre-Covid\n(Before Mar 2020)', 'During Covid\n(Mar–Dec 2020)', 'Post-Covid\n(After 2020)']
values  = [pre_covid, during_covid, post_covid]
colors  = ['#4CAF50', '#F44336', '#2196F3']

bars = ax.bar(periods, values, color=colors, edgecolor='white', width=0.5)
ax.set_title("Covid-19 Impact on Unemployment Rate in India",
             fontsize=13, fontweight='bold')
ax.set_ylabel("Average Unemployment Rate (%)")
ax.set_ylim(0, max(values) + 5)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.2f}%", ha='center', fontweight='bold', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("unemployment_covid_impact.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: unemployment_covid_impact.png")


# ─────────────────────────────────────────
# STEP 6: Region-wise Unemployment Analysis
# ─────────────────────────────────────────
region_avg = df.groupby('Region')['Unemployment_Rate'].mean().sort_values(ascending=False).reset_index()

plt.figure(figsize=(14, 7))
bars = plt.barh(region_avg['Region'], region_avg['Unemployment_Rate'],
                color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(region_avg))),
                edgecolor='white')
plt.title("Average Unemployment Rate by Region in India",
          fontsize=14, fontweight='bold')
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("Region")
for bar, val in zip(bars, region_avg['Unemployment_Rate']):
    plt.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
             f"{val:.1f}%", va='center', fontsize=9)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("unemployment_by_region.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: unemployment_by_region.png")


# ─────────────────────────────────────────
# STEP 7: Rural vs Urban Comparison
# ─────────────────────────────────────────
area_monthly = df.groupby(['Date', 'Area'])['Unemployment_Rate'].mean().reset_index()
rural  = area_monthly[area_monthly['Area'] == 'Rural']
urban  = area_monthly[area_monthly['Area'] == 'Urban']

plt.figure(figsize=(14, 6))
plt.plot(rural['Date'], rural['Unemployment_Rate'],
         color='#4CAF50', linewidth=2.5, marker='o', markersize=3, label='Rural')
plt.plot(urban['Date'], urban['Unemployment_Rate'],
         color='#FF5722', linewidth=2.5, marker='s', markersize=3, label='Urban')
plt.axvspan(covid_start, covid_end, alpha=0.1, color='red', label='Covid-19 Period')
plt.title("Rural vs Urban Unemployment Rate Over Time",
          fontsize=14, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("unemployment_rural_vs_urban.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: unemployment_rural_vs_urban.png")


# ─────────────────────────────────────────
# STEP 8: Heatmap — Region vs Month
# ─────────────────────────────────────────
df['Month'] = df['Date'].dt.strftime('%b %Y')
heatmap_data = df.groupby(['Region', 'Month'])['Unemployment_Rate'].mean().unstack()

# Sort columns by date
heatmap_data = heatmap_data[sorted(heatmap_data.columns,
                key=lambda x: pd.to_datetime(x, format='%b %Y'))]

plt.figure(figsize=(18, 10))
sns.heatmap(heatmap_data, cmap='YlOrRd', linewidths=0.3,
            annot=False, cbar_kws={'label': 'Unemployment Rate (%)'})
plt.title("Unemployment Rate Heatmap — Region vs Month",
          fontsize=14, fontweight='bold')
plt.xlabel("Month")
plt.ylabel("Region")
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig("unemployment_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: unemployment_heatmap.png")


# ─────────────────────────────────────────
# STEP 9: Top 5 Most & Least Affected Regions
# ─────────────────────────────────────────
print("\n─── Top 5 Most Affected Regions (Highest Unemployment) ───")
print(region_avg.head(5).to_string(index=False))

print("\n─── Top 5 Least Affected Regions (Lowest Unemployment) ───")
print(region_avg.tail(5).to_string(index=False))


# ─────────────────────────────────────────
# STEP 10: Key Insights Summary
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("   📋 KEY INSIGHTS")
print("=" * 60)
print(f"   • Overall avg unemployment rate : {df['Unemployment_Rate'].mean():.2f}%")
print(f"   • Highest unemployment rate     : {df['Unemployment_Rate'].max():.2f}%")
print(f"   • Lowest unemployment rate      : {df['Unemployment_Rate'].min():.2f}%")
print(f"   • Most affected region          : {region_avg.iloc[0]['Region']}")
print(f"   • Least affected region         : {region_avg.iloc[-1]['Region']}")
print(f"   • Covid-19 spike                : +{during_covid - pre_covid:.2f}% increase")
rural_avg  = df[df['Area'] == 'Rural']['Unemployment_Rate'].mean()
urban_avg  = df[df['Area'] == 'Urban']['Unemployment_Rate'].mean()
print(f"   • Rural avg unemployment        : {rural_avg:.2f}%")
print(f"   • Urban avg unemployment        : {urban_avg:.2f}%")

print("\n" + "=" * 60)
print("   ✅ ALL TASKS COMPLETED SUCCESSFULLY!")
print("   📁 Output Files Generated:")
print("      • unemployment_trend.png")
print("      • unemployment_covid_impact.png")
print("      • unemployment_by_region.png")
print("      • unemployment_rural_vs_urban.png")
print("      • unemployment_heatmap.png")
print("=" * 60)