# usability_test.py

import pandas as pd
import numpy as np
import os
import json
import re
from data_analysis_system.simplified_main import run_data_analysis

def create_business_dataset():
    """Create a realistic business dataset for usability testing."""
    np.random.seed(42)  # For reproducibility
    
    # Create a dataset with business metrics
    n = 500
    
    # Create date range - past 500 days
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='D')
    
    # Basic metrics with seasonal patterns and trends
    base_sales = 10000 + np.linspace(0, 5000, n)  # Increasing trend
    seasonality = 2000 * np.sin(np.linspace(0, 4*np.pi, n))  # Seasonal pattern
    daily_sales = base_sales + seasonality + np.random.normal(0, 1000, n)
    
    # Create marketing spend with relationship to sales (with 7-day lag)
    base_marketing = 2000 + np.linspace(0, 1000, n)  # Increasing trend
    marketing_spend = base_marketing + np.random.normal(0, 200, n)
    
    # Create customer metrics
    new_customers = daily_sales * 0.05 + np.random.normal(0, 20, n)
    repeat_customers = daily_sales * 0.15 + np.random.normal(0, 30, n)
    
    # Create product categories
    categories = ['Electronics', 'Clothing', 'Home Goods', 'Books', 'Toys']
    product_category = np.random.choice(categories, n)
    
    # Create sales channels
    channels = ['Online', 'Store', 'Phone', 'Partner']
    sales_channel = np.random.choice(channels, n, p=[0.6, 0.3, 0.05, 0.05])
    
    # Make online sales higher
    daily_sales[sales_channel == 'Online'] *= 1.2
    
    # Make electronics sales higher
    daily_sales[product_category == 'Electronics'] *= 1.5
    
    # Calculate profit margin based on category and channel
    base_margin = 0.3  # 30% base margin
    category_margins = {
        'Electronics': 0.25,
        'Clothing': 0.4,
        'Home Goods': 0.35,
        'Books': 0.2,
        'Toys': 0.45
    }
    
    channel_margin_multipliers = {
        'Online': 1.1,  # 10% higher margin online
        'Store': 0.9,   # 10% lower margin in store
        'Phone': 1.0,
        'Partner': 0.7  # 30% lower margin with partners
    }
    
    profit_margin = np.zeros(n)
    for i in range(n):
        cat_margin = category_margins[product_category[i]]
        channel_mult = channel_margin_multipliers[sales_channel[i]]
        profit_margin[i] = cat_margin * channel_mult
    
    # Calculate profit
    profit = daily_sales * profit_margin
    
    # Calculate customer satisfaction (higher for certain categories and channels)
    base_satisfaction = np.random.normal(80, 5, n)  # Base score out of 100
    satisfaction_adjustment = np.zeros(n)
    
    # Electronics have lower satisfaction
    satisfaction_adjustment[product_category == 'Electronics'] -= 5
    
    # Books have higher satisfaction
    satisfaction_adjustment[product_category == 'Books'] += 8
    
    # Online has lower satisfaction
    satisfaction_adjustment[sales_channel == 'Online'] -= 3
    
    # Store has higher satisfaction
    satisfaction_adjustment[sales_channel == 'Store'] += 5
    
    customer_satisfaction = np.clip(base_satisfaction + satisfaction_adjustment, 0, 100)
    
    # Create the DataFrame
    data = {
        'date': dates,
        'daily_sales': daily_sales,
        'marketing_spend': marketing_spend,
        'new_customers': new_customers,
        'repeat_customers': repeat_customers,
        'product_category': product_category,
        'sales_channel': sales_channel,
        'profit_margin': profit_margin,
        'profit': profit,
        'customer_satisfaction': customer_satisfaction
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs('usability_data', exist_ok=True)
    test_file = 'usability_data/business_dataset.csv'
    df.to_csv(test_file, index=False)
    
    print(f"Created business dataset for usability testing: {test_file}")
    print(f"Data shape: {df.shape}")
    print("Data preview:")
    print(df.head())
    
    return test_file

def evaluate_insight_clarity(insight):
    """Evaluate the clarity of an insight."""
    # Check for key components of clear insights
    has_metric = any(metric in insight.lower() for metric in ['sales', 'profit', 'margin', 'customers', 'marketing', 'satisfaction'])
    has_direction = any(direction in insight.lower() for direction in ['increase', 'decrease', 'higher', 'lower', 'more', 'less', 'growth', 'decline'])
    has_magnitude = bool(re.search(r'\d+(\.\d+)?%', insight))  # Contains percentage
    has_segment = any(segment_word in insight.lower() for segment_word in ['category', 'channel', 'segment', 'group'])
    
    # Calculate clarity score (0-1)
    components = [has_metric, has_direction, has_magnitude, has_segment]
    clarity_score = sum(components) / len(components)
    
    return {
        "clarity_score": clarity_score,
        "has_metric": has_metric,
        "has_direction": has_direction,
        "has_magnitude": has_magnitude,
        "has_segment": has_segment
    }

def evaluate_insight_actionability(insight, context="business"):
    """Evaluate how actionable an insight is."""
    # Keywords that suggest actionable insights
    actionable_keywords = {
        'business': ['opportunity', 'improve', 'optimize', 'increase', 'decrease', 'strategy', 
                    'recommend', 'action', 'focus', 'prioritize', 'target', 'allocate'],
        'scientific': ['investigate', 'experiment', 'test', 'hypothesis', 'theory', 'model', 
                      'predict', 'validate', 'evidence', 'mechanism']
    }
    
    # Check for actionable components
    has_actionable_keyword = any(keyword in insight.lower() for keyword in actionable_keywords.get(context, []))
    has_comparison = any(comp in insight.lower() for comp in ['than', 'compared to', 'versus', 'vs', 'more than', 'less than'])
    has_specific_entity = bool(re.search(r'[\'\"]([^\'\"]+)[\'\"]', insight))  # Text in quotes
    
    # Check for causal language
    causal_phrases = ['because', 'due to', 'result of', 'leads to', 'causes', 'affects', 'influences', 'impact']
    has_causal_language = any(phrase in insight.lower() for phrase in causal_phrases)
    
    # Calculate actionability score (0-1)
    components = [has_actionable_keyword, has_comparison, has_specific_entity, has_causal_language]
    actionability_score = sum(components) / len(components)
    
    return {
        "actionability_score": actionability_score,
        "has_actionable_keyword": has_actionable_keyword,
        "has_comparison": has_comparison,
        "has_specific_entity": has_specific_entity,
        "has_causal_language": has_causal_language
    }

def run_usability_test():
    """Run usability test on business dataset and evaluate the insights."""
    print("\n" + "="*50)
    print("USABILITY TESTING")
    print("="*50)
    
    # Create business dataset
    data_path = create_business_dataset()
    
    # Run the analysis
    class Args:
        source_type = "file"
        data_source = data_path
        objective = "Analyze sales performance and identify opportunities for improvement"
        business_context = "Retail business analytics"
        target_column = "daily_sales"
        output_dir = "usability_results"
    
    args = Args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\nRunning analysis on business dataset...")
    result = run_data_analysis(args)
    
    # Evaluate insights if analysis was successful
    if result["status"] == "success":
        print(f"\n✅ Analysis completed successfully")
        
        # Extract insights
        insights = []
        if "key_findings" in result:
            for finding in result["key_findings"]:
                if "finding" in finding:
                    insights.append(finding["finding"])
        
        # Extract recommendations
        recommendations = []
        if "recommendations" in result:
            for rec in result["recommendations"]:
                if "recommendation" in rec:
                    recommendations.append(rec["recommendation"])
        
        print(f"Found {len(insights)} insights and {len(recommendations)} recommendations")
        
        # Evaluate each insight
        insight_evaluations = []
        total_clarity = 0
        total_actionability = 0
        
        print("\nEvaluating insights...")
        for i, insight in enumerate(insights):
            clarity = evaluate_insight_clarity(insight)
            actionability = evaluate_insight_actionability(insight, "business")
            
            total_clarity += clarity["clarity_score"]
            total_actionability += actionability["actionability_score"]
            
            insight_evaluations.append({
                "insight": insight,
                "clarity": clarity,
                "actionability": actionability,
                "overall_score": (clarity["clarity_score"] + actionability["actionability_score"]) / 2
            })
        
        # Evaluate each recommendation
        recommendation_evaluations = []
        total_rec_clarity = 0
        total_rec_actionability = 0
        
        print("Evaluating recommendations...")
        for i, rec in enumerate(recommendations):
            clarity = evaluate_insight_clarity(rec)
            actionability = evaluate_insight_actionability(rec, "business")
            
            total_rec_clarity += clarity["clarity_score"]
            total_rec_actionability += actionability["actionability_score"]
            
            recommendation_evaluations.append({
                "recommendation": rec,
                "clarity": clarity,
                "actionability": actionability,
                "overall_score": (clarity["clarity_score"] + actionability["actionability_score"]) / 2
            })
        
        # Calculate average scores
        avg_clarity = total_clarity / len(insights) if insights else 0
        avg_actionability = total_actionability / len(insights) if insights else 0
        avg_rec_clarity = total_rec_clarity / len(recommendations) if recommendations else 0
        avg_rec_actionability = total_rec_actionability / len(recommendations) if recommendations else 0
        
        # Calculate overall usability score (0-100)
        insight_weight = 0.6
        recommendation_weight = 0.4
        
        insight_score = (avg_clarity + avg_actionability) / 2
        rec_score = (avg_rec_clarity + avg_rec_actionability) / 2
        
        overall_usability = (insight_score * insight_weight + rec_score * recommendation_weight) * 100
        
        # Rank insights by overall score
        sorted_insights = sorted(insight_evaluations, key=lambda x: x["overall_score"], reverse=True)
        
        # Prepare usability report
        usability_report = {
            "metrics": {
                "overall_usability_score": overall_usability,
                "insight_clarity": avg_clarity,
                "insight_actionability": avg_actionability,
                "recommendation_clarity": avg_rec_clarity,
                "recommendation_actionability": avg_rec_actionability,
                "total_insights": len(insights),
                "total_recommendations": len(recommendations)
            },
            "best_insights": sorted_insights[:3] if len(sorted_insights) >= 3 else sorted_insights,
            "all_insights": insight_evaluations,
            "recommendation_evaluations": recommendation_evaluations
        }
        
        # Save report
        with open(os.path.join(args.output_dir, "usability_report.json"), 'w') as f:
            json.dump(usability_report, f, indent=2)
        
        # Print summary
        print("\nUsability Test Results:")
        print(f"Overall Usability Score: {overall_usability:.1f}/100")
        print(f"Insight Clarity: {avg_clarity:.2f}")
        print(f"Insight Actionability: {avg_actionability:.2f}")
        print(f"Recommendation Clarity: {avg_rec_clarity:.2f}")
        print(f"Recommendation Actionability: {avg_rec_actionability:.2f}")
        
        print("\nTop 3 Insights by Usability:")
        for i, insight in enumerate(sorted_insights[:3]):
            print(f"{i+1}. [{insight['overall_score']:.2f}] {insight['insight']}")
        
        # Generate report in HTML format
        generate_usability_report(usability_report, args.output_dir)
        
        return usability_report
    else:
        print(f"\n❌ Analysis failed: {result.get('error_message', 'Unknown error')}")
        return None

def generate_usability_report(report, output_dir):
    """Generate an HTML report for usability testing."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Analysis System Usability Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .metric {{ margin-bottom: 10px; }}
            .metric-value {{ font-weight: bold; }}
            .insights {{ margin-bottom: 30px; }}
            .insight {{ background-color: #f8f9fa; padding: 10px; margin-bottom: 10px; border-radius: 5px; }}
            .score-high {{ color: #28a745; }}
            .score-medium {{ color: #fd7e14; }}
            .score-low {{ color: #dc3545; }}
            .recommendation {{ background-color: #e9f7ef; padding: 10px; margin-bottom: 10px; border-radius: 5px; }}
            progress {{ width: 100%; height: 20px; }}
        </style>
    </head>
    <body>
        <h1>Data Analysis System Usability Report</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            
            <div class="metric">
                <div>Overall Usability Score: 
                    <span class="metric-value {get_score_class(report['metrics']['overall_usability_score']/100)}">
                        {report['metrics']['overall_usability_score']:.1f}/100
                    </span>
                </div>
                <progress value="{report['metrics']['overall_usability_score']}" max="100"></progress>
            </div>
            
            <div class="metric">
                <div>Insight Clarity: 
                    <span class="metric-value {get_score_class(report['metrics']['insight_clarity'])}">
                        {report['metrics']['insight_clarity']:.2f}
                    </span>
                </div>
                <progress value="{report['metrics']['insight_clarity']*100}" max="100"></progress>
            </div>
            
            <div class="metric">
                <div>Insight Actionability: 
                    <span class="metric-value {get_score_class(report['metrics']['insight_actionability'])}">
                        {report['metrics']['insight_actionability']:.2f}
                    </span>
                </div>
                <progress value="{report['metrics']['insight_actionability']*100}" max="100"></progress>
            </div>
            
            <div class="metric">
                <div>Recommendation Clarity: 
                    <span class="metric-value {get_score_class(report['metrics']['recommendation_clarity'])}">
                        {report['metrics']['recommendation_clarity']:.2f}
                    </span>
                </div>
                <progress value="{report['metrics']['recommendation_clarity']*100}" max="100"></progress>
            </div>
            
            <div class="metric">
                <div>Recommendation Actionability: 
                    <span class="metric-value {get_score_class(report['metrics']['recommendation_actionability'])}">
                        {report['metrics']['recommendation_actionability']:.2f}
                    </span>
                </div>
                <progress value="{report['metrics']['recommendation_actionability']*100}" max="100"></progress>
            </div>
            
            <p>Total Insights: {report['metrics']['total_insights']}</p>
            <p>Total Recommendations: {report['metrics']['total_recommendations']}</p>
        </div>
        
        <div class="insights">
            <h2>Top Insights</h2>
    """
    
    # Add top insights
    for i, insight in enumerate(report['best_insights']):
        score_class = get_score_class(insight['overall_score'])
        html_content += f"""
            <div class="insight">
                <h3>Insight {i+1} <span class="{score_class}">({insight['overall_score']:.2f})</span></h3>
                <p>{insight['insight']}</p>
                <div>Clarity: {insight['clarity']['clarity_score']:.2f}</div>
                <div>Actionability: {insight['actionability']['actionability_score']:.2f}</div>
                <div>Components:</div>
                <ul>
                    <li>Has metric: {'✅' if insight['clarity']['has_metric'] else '❌'}</li>
                    <li>Has direction: {'✅' if insight['clarity']['has_direction'] else '❌'}</li>
                    <li>Has magnitude: {'✅' if insight['clarity']['has_magnitude'] else '❌'}</li>
                    <li>Has segment: {'✅' if insight['clarity']['has_segment'] else '❌'}</li>
                    <li>Has actionable keyword: {'✅' if insight['actionability']['has_actionable_keyword'] else '❌'}</li>
                    <li>Has comparison: {'✅' if insight['actionability']['has_comparison'] else '❌'}</li>
                    <li>Has specific entity: {'✅' if insight['actionability']['has_specific_entity'] else '❌'}</li>
                    <li>Has causal language: {'✅' if insight['actionability']['has_causal_language'] else '❌'}</li>
                </ul>
            </div>
        """
    
    # Add top recommendations
    html_content += """
        <h2>Top Recommendations</h2>
    """
    
    sorted_recommendations = sorted(report['recommendation_evaluations'], 
                                    key=lambda x: x["overall_score"], 
                                    reverse=True)
    
    top_recommendations = sorted_recommendations[:3] if len(sorted_recommendations) >= 3 else sorted_recommendations
    
    for i, rec in enumerate(top_recommendations):
        score_class = get_score_class(rec['overall_score'])
        html_content += f"""
            <div class="recommendation">
                <h3>Recommendation {i+1} <span class="{score_class}">({rec['overall_score']:.2f})</span></h3>
                <p>{rec['recommendation']}</p>
                <div>Clarity: {rec['clarity']['clarity_score']:.2f}</div>
                <div>Actionability: {rec['actionability']['actionability_score']:.2f}</div>
            </div>
        """
    
    # Complete the HTML
    html_content += """
        </div>
        
        <div>
            <h2>Improvement Suggestions</h2>
            <ul>
    """
    
    # Add improvement suggestions based on scores
    if report['metrics']['insight_clarity'] < 0.7:
        html_content += """
                <li>Improve insight clarity by ensuring insights include specific metrics, clear directions of change, and quantified magnitudes</li>
        """
    
    if report['metrics']['insight_actionability'] < 0.7:
        html_content += """
                <li>Enhance insight actionability by including more specific comparisons, causal relationships, and actionable keywords</li>
        """
    
    if report['metrics']['recommendation_clarity'] < 0.7:
        html_content += """
                <li>Make recommendations more clear by specifying which metrics they will affect and by how much</li>
        """
    
    if report['metrics']['recommendation_actionability'] < 0.7:
        html_content += """
                <li>Improve recommendation actionability by making them more specific and targeted to particular segments or processes</li>
        """
    
    html_content += """
                <li>Consider adding more segment-specific insights to make findings more targeted</li>
                <li>Include more causal language to explain why patterns exist, not just what they are</li>
                <li>Add comparative benchmarks where possible to provide context for insights</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open(os.path.join(output_dir, "usability_report.html"), 'w') as f:
        f.write(html_content)
    
    print(f"\nUsability report saved to {os.path.join(output_dir, 'usability_report.html')}")

def get_score_class(score):
    """Get CSS class based on score."""
    if score >= 0.7:
        return "score-high"
    elif score >= 0.4:
        return "score-medium"
    else:
        return "score-low"

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    run_usability_test()