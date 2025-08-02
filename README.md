# Data Analysis Agentic System

An intelligent data analysis system that demonstrates the principles of agentic AI using specialized agents, tool integration, and multi-agent orchestration.

## Overview

This project implements an agentic AI system for data analysis that orchestrates specialized components to perform complex analytical tasks. The system follows a sequential workflow architecture with a central controller that coordinates the process, specialized agents for different aspects of data analysis, and a custom Insight Generator tool that automatically discovers meaningful patterns and translates them into actionable business intelligence.

### Key Features

- **Multi-agent Orchestration**: A controller agent that orchestrates the workflow and synthesizes results
- **Specialized Agents**: Purpose-built agents for data collection, processing, analysis, and visualization
- **Built-in Tool Integration**: Tools for data retrieval, cleaning, and statistical analysis
- **Custom Insight Generator**: Automatically identifies patterns and generates actionable recommendations
- **Error Handling & Memory Management**: Robust error recovery and context preservation across the workflow

## System Architecture

The system follows a hierarchical architecture with agents and tools organized as follows:

```
Controller Agent
├── Data Collection Agent
│   └── Data Retrieval Tool
├── Data Processing Agent
│   └── Data Cleaning Tool
├── Analysis Agent
│   ├── Statistical Analysis Tool
│   └── Custom Insight Generator Tool
└── Visualization Agent
    └── Visualization Tool
```

## Agent Roles and Responsibilities

### Controller Agent
Orchestrates the end-to-end data analysis process and synthesizes results. Responsible for task delegation, workflow management, error handling, and final report compilation.

### Data Collection Agent
Gathers high-quality data from various sources and validates its structure. Handles different data formats and performs initial quality assessment.

### Data Processing Agent
Cleans and transforms raw data into analysis-ready datasets. Handles missing values, outliers, duplicates, and performs normalization.

### Analysis Agent
Discovers meaningful patterns and insights in data. Performs statistical analysis and generates actionable insights prioritized by significance.

### Visualization Agent
Creates clear, informative visualizations that communicate key insights. Selects appropriate visualization types for different data patterns.

## Tool Integration

### Data Retrieval Tool
Fetches data from various sources including files (CSV, Excel, JSON), APIs, and URLs. Performs basic validation and returns a standardized data structure.

### Data Cleaning Tool
Handles missing values, duplicates, outliers, and performs normalization. Uses statistical methods for outlier detection and implements different cleaning strategies based on data types.

### Statistical Analysis Tool
Performs descriptive statistics, correlation analysis, regression analysis, and hypothesis testing. Provides detailed statistical results with significance measures.

### Visualization Tool
Creates various visualizations including bar charts, line plots, scatter plots, histograms, box plots, and heatmaps. Automatically formats and labels visualizations.

## Custom Insight Generator

The Insight Generator Tool is the core innovation of this project. It bridges the gap between raw statistical analysis and actionable business intelligence by:

1. **Automatically Discovering Patterns**: Identifies trends, correlations, outliers, and segment differences without requiring manual specification
2. **Generating Natural Language Explanations**: Translates complex statistical findings into clear, business-friendly language
3. **Providing Actionable Recommendations**: Suggests specific next steps based on the insights discovered
4. **Prioritizing by Significance**: Ranks insights by statistical significance and business relevance

The tool implements specialized algorithms for:
- Trend detection in time series data
- Correlation discovery with strength classification
- Outlier detection with contextual explanation
- Segment difference identification
- Recommendation generation based on findings

## Performance Evaluation

The system has been thoroughly evaluated across multiple dimensions:

### Accuracy Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Correlation Identification | 50% | Correctly identified half of the statistically significant correlations |
| Trend Detection | 100% | Perfectly identified all meaningful trends in time series data |
| Outlier Detection | 0% | System needs improvement in identifying statistical outliers |
| Segment Difference Detection | 100% | Perfectly identified all meaningful segment differences |

### Efficiency Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Average Processing Time | 0.93s per 100 rows | Linear scaling with data size |
| Memory Usage | ~2.45MB + 0.43KB per row | Highly efficient memory utilization |
| CPU Utilization | 0.71% average | Minimal CPU resource requirements |

### Reliability Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Error Rate | 0.00% | No errors encountered during standard operation |
| Recovery Rate | 50.00% | Half of error conditions successfully recovered |
| Consistency | 75.00% | High consistency of results across multiple runs |

## Installation

### Prerequisites

- Python 3.9+
- Required Python packages (see requirements.txt)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/TejaChowdary19/Agentic-Systems.git
cd data-analysis-agentic-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from data_analysis_system.simplified_main import run_data_analysis

# Define arguments
class Args:
    source_type = "file"
    data_source = "path/to/your/data.csv"
    objective = "Analyze sales performance"
    business_context = "Retail analysis"
    target_column = "sales"
    output_dir = "results"

# Run analysis
result = run_data_analysis(Args)

# Print results
if result["status"] == "success":
    print(result["summary"])
```

### Sample Output

```json
{
  "status": "success",
  "analysis_objective": "Analyze sales performance",
  "data_summary": {
    "original_data": [100, 6],
    "cleaned_data": [99, 6]
  },
  "key_findings": [
    {
      "type": "correlation",
      "finding": "Strong positive correlation (0.96) between price and competitor_price"
    },
    {
      "type": "trend",
      "finding": "marketing_spend is increasing over time (by 253.05%)"
    },
    {
      "type": "segment",
      "finding": "customer_segment segment 'High Value' has 139.7% higher sales than segment 'Low Value'"
    }
  ],
  "recommendations": [
    {
      "type": "correlation_investigation",
      "recommendation": "Investigate potential causal relationship between price and competitor_price",
      "priority": "high"
    },
    {
      "type": "trend_analysis",
      "recommendation": "Conduct deeper time-series analysis on marketing_spend",
      "priority": "high"
    }
  ]
}
```

## Evaluation and Testing

The system includes comprehensive evaluation scripts to measure performance metrics:

1. **Accuracy Evaluation**:
```bash
python evaluation_metrics.py
```

2. **Performance and Reliability Evaluation**:
```bash
python performance_metrics.py
```

## Limitations and Future Work

### Current Limitations

- Outlier detection needs improvement (0% accuracy in current evaluation)
- Limited to structured tabular data (CSV, Excel, JSON)
- Fixed sequential workflow regardless of data characteristics
- No advanced machine learning capabilities

### Planned Improvements

1. **Enhanced Outlier Detection**:
   - Implement multiple detection algorithms (IQR, isolation forests)
   - Improve contextual awareness for outlier significance

2. **Adaptive Workflow**:
   - Implement decision points to customize analysis based on data
   - Create alternative analysis paths for different data types

3. **Learning System**:
   - Develop mechanisms to track successful analysis patterns
   - Implement strategy optimization based on historical performance

4. **Advanced Analysis Capabilities**:
   - Add support for machine learning models
   - Implement causal inference methods

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was developed as part of the Building Agentic Systems assignment
- Thanks to the creators of the libraries used in this project (pandas, numpy, matplotlib, etc.)
