# 🛵 Taxi Trip & Revenue Analysis Dashboard

An interactive, multi-tab Python dashboard for analysing bike-taxi trip data, rider performance, and revenue metrics — built with **Streamlit** (Python frontend) and **Plotly** (interactive charts).

---

## 📁 Project Structure

```
bike rider DA/
├── app.py                   # Main application (frontend + backend combined)
├── https://www.kaggle.com/datasets/itskoustavdas/indian-bike-rider-daily-operational-dataset-2025 # Source dataset (76,502 rows × 18 columns)
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── report.docx              # UI output report with screenshots & analysis
```

---

## 📊 Dataset Overview

| Column | Description |
|---|---|
| `rider_id` | Unique rider identifier |
| `date` | Shift date (YYYY-MM-DD) |
| `location` | City (Mumbai, Kolkata, Chennai, …) |
| `rides` | Total rides in shift |
| `on_time_rides` | Rides completed on time |
| `cancelled_rides` | Rides cancelled |
| `missed_rides` | Rides missed |
| `distance_km` | Total km covered |
| `overhead_distance_km` | Non-revenue km (dead miles) |
| `earnings` | Base earnings (₹) |
| `peak_bonus` | Peak hour bonus (₹) |
| `waiting_earnings` | Waiting time earnings (₹) |
| `fuel_cost` | Fuel cost (₹) |
| `idle_minutes` | Minutes idle during shift |
| `work_hours` | Total shift duration (hrs) |
| `customer_rating` | Average customer rating (0–5) |
| `avg_speed_kmph` | Average speed km/h |
| `network_downtime` | App network downtime (mins) |

**Derived columns computed at runtime:**
- `net_earnings` = earnings + peak_bonus + waiting_earnings − fuel_cost
- `revenue_per_km` = earnings / distance_km
- `profit_margin` = net_earnings / gross_revenue × 100
- `on_time_rate`, `cancel_rate`, `overhead_ratio`

---

## 🖥️ Dashboard Tabs

### 📊 Tab 1 — Overview
- **8 KPI cards**: Total Rides, Gross Revenue, Net Profit, Avg Rating, Total Distance, Unique Riders, Total Fuel Cost, Avg Rides/Day
- **Bar chart**: Gross Revenue by City
- **Donut chart**: Ride share by City
- **Data preview table**: First 200 filtered rows

### 💰 Tab 2 — Revenue Analysis
- **Stacked + waterfall bar**: Earnings components (base, peak bonus, waiting, fuel) by City
- **Horizontal bar**: Average profit margin by City
- **Box plot**: Revenue per km distribution by City
- **Scatter plot**: Net Earnings vs Fuel Cost (bubble = rides count)
- **Grouped bar**: Peak bonus vs no-bonus earnings comparison
- **Histogram**: Profit margin distribution

### 🚗 Tab 3 — Trip Analysis
- **Grouped bar**: On-time / Cancelled / Missed rides by City
- **Grouped bar**: On-Time Rate vs Cancellation Rate (%)
- **Histogram**: Distance distribution by City
- **Violin plot**: Speed distribution by City
- **Box plot**: Idle minutes by City
- **Scatter + OLS trendline**: Overhead vs actual distance

### 🏆 Tab 4 — Rider Performance
- **Horizontal bar**: Top N riders by net earnings (adjustable slider)
- **Scatter**: Customer rating vs Net earnings (bubble = rides)
- **Histogram**: Rating distribution by City
- **Histogram**: Cancellation rate distribution across riders
- **Scatter + trendline**: Avg speed vs Gross Revenue
- **Radar chart**: City-level performance radar (normalised metrics)
- **Leaderboard table**: Sortable top-N rider stats

### 📅 Tab 5 — Time Trends
- **Dual-panel time-series**: Daily Revenue + Net Profit + Fuel Cost (line) + Daily Rides (bar)
- **Bar**: Average revenue by day of week
- **Line**: Average rides & rating by day of week
- **Heatmap**: Monthly net earnings by City × Month
- **Scatter + OLS**: Work hours vs net earnings correlation

---

## 🚀 Setup & Run

### 1. Clone / navigate to project folder

```bash
cd "bike rider DA"
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

The dashboard opens automatically at **http://localhost:8501**

---

## 🔧 Sidebar Filters

| Filter | Description |
|---|---|
| Date Range | Restricts analysis to selected date window |
| City / Location | Multi-select for one or more cities |
| Customer Rating | Slider to filter by min–max rating |

All charts and KPIs update instantly when filters change.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│              Streamlit UI                │
│  (Tabs, Sidebar, KPI Cards, Charts)     │
├─────────────────────────────────────────┤
│           Plotly Express / GO            │
│   (Interactive charts, radar, heatmap)  │
├─────────────────────────────────────────┤
│          Pandas + NumPy Backend          │
│  (Data loading, cleaning, aggregation)  │
├─────────────────────────────────────────┤
│         BikeRidersInfoSheet.csv          │
│           (76,502 records)               │
└─────────────────────────────────────────┘
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥1.33.0 | Python web UI framework |
| `pandas` | ≥2.0.0 | Data loading and transformation |
| `numpy` | ≥1.24.0 | Numerical operations |
| `plotly` | ≥5.20.0 | Interactive visualisations |
| `statsmodels` | ≥0.14.0 | OLS trendlines in scatter plots |

---

## 📄 Key Findings (from full dataset)

- **76,502 shift records** across multiple Indian cities
- Net earnings = gross earnings minus fuel cost; peak bonus significantly improves margins
- Higher customer ratings correlate positively with net earnings
- Weekends show elevated ride counts and revenue
- Overhead (dead-mile) distance is a key cost driver

---

## 📝 License

For educational / IBM SkillBuild internship use only.
