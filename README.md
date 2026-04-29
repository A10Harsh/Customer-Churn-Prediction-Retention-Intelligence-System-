# Customer-Churn-Prediction-Retention-Intelligence-System




Customer churn is a critical challenge in the telecom industry, directly impacting revenue and customer lifetime value. While traditional machine learning models can accurately predict which customers are likely to churn, they fail to provide actionable insights.

This project bridges that gap by developing a GenAI-powered Next Best Action (NBA) framework that transforms churn predictions into personalized, explainable, and actionable retention strategies.

The system integrates:
Predictive modeling (Random Forest, XGBoost, ANN)
Explainable AI using SHAP
Rule-based decision intelligence
Generative AI for personalized recommendations

Objectives
Predict customer churn using machine learning models
Identify key factors driving churn using SHAP explainability
Map churn drivers to actionable retention strategies
Generate personalized recommendations using Generative AI
Build a decision-support system for customer retention


Methodology
1. Churn Prediction

Multiple models are trained and evaluated:

Random Forest
XGBoost (final selected model)
Artificial Neural Network (ANN)
2. Explainability (SHAP)

SHAP (SHapley Additive exPlanations) is used to:cInterpret model predictions
Identify top features influencing churn for each customer
3. Action Mapping Enginec: Key churn drivers are mapped to predefined business actions:

Feature	Action
MonthlyCharges	Offer discount / alternative plan
tenure	Provide onboarding or engagement offers
Contract	Encourage long-term subscription
TechSupport	Offer free or discounted support
OnlineSecurity	Promote security add-ons
4. Business Rule Filtering

Actions are refined using domain-specific constraints:

Discounts applied only to high-paying customers
Tech support offered only if not already subscribed
Contract upgrade suggested for monthly users
5. GenAI Personalization Layer

Generative AI converts structured actions into:

Human-like retention strategies
Personalized customer messages
Justifications for recommendations

Here are the notebooks 

https://colab.research.google.com/drive/1lqboF6sQISR4MkN1PCFTyzIRVMTt06VH?authuser=1#scrollTo=hXP2QHrAK0FV


ANN+SHAP notebooks : https://colab.research.google.com/drive/1OR6HgEslS8fwK3NZ0tNstKNWR-qVa1uD?authuser=1#scrollTo=gMLKSoK6S-bY

XGBoost+ Shap Notebooks : https://colab.research.google.com/drive/1AnHu-dmiwGw0aEVv2MCGKE4F9Y5QmL1O?authuser=1#scrollTo=AdtyiS5ElKIi

Random Forest + shap notebooks : https://colab.research.google.com/drive/1u1M5FqY9QX9kah3koKSVCkQe0GAchKqd?authuser=1#scrollTo=howyjoYXsrye
